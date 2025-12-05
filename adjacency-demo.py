# Databricks notebook source
# MAGIC %pip install -r requirements.lock
# MAGIC %restart_python

# COMMAND ----------

from collections import defaultdict
from enum import Enum
from typing import Optional

import numpy as np
from ortools.sat.python import cp_model
from pydantic import BaseModel

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define our entities: stands and turns

# COMMAND ----------


class Stand(BaseModel):
    stand_id: str


class Turn(BaseModel):
    turn_id: str
    turn_seq: int
    flight_id: str
    arrival_time: int
    departure_time: int


# COMMAND ----------

# MAGIC %md
# MAGIC ## Test data

# COMMAND ----------

turns: list[Turn] = [
    Turn(turn_id="1", turn_seq=0, flight_id="FR13", arrival_time=20, departure_time=55),
    Turn(turn_id="2", turn_seq=0, flight_id="FR42", arrival_time=10, departure_time=35),
    Turn(turn_id="3", turn_seq=0, flight_id="FR66", arrival_time=35, departure_time=60),
    Turn(turn_id="4", turn_seq=0, flight_id="FR99", arrival_time=25, departure_time=50),
]

stands: list[Stand] = [
    Stand(stand_id="1L"),
    Stand(stand_id="1C"),
    Stand(stand_id="2L"),
    Stand(stand_id="2C"),
    Stand(stand_id="2R"),
]


# COMMAND ----------

# MAGIC %md
# MAGIC ## Feasibility matrix
# MAGIC + This is a matrix of booleans that indicates whether a turn can be assigned to a stand
# MAGIC + It is used to filter out infeasible assignments
# MAGIC + We assume this has been pre-computed from static rules

# COMMAND ----------

feasibility: np.ndarray[bool] = np.ones((len(turns), len(stands)), dtype=bool)

feasibility[0][0] = False
feasibility[1][0] = False
feasibility[3][0] = False
feasibility[3][2] = False
feasibility[3][4] = False

feasibility

# COMMAND ----------

# MAGIC %md
# MAGIC ## "Main" model: turn presence and stand intervals
# MAGIC + This model is used to find the optimal assignment of turns to stands
# MAGIC + This is a simple model that uses (optional) interval variables to represent the time that a turn is assigned to a stand
# MAGIC together with Boolean presence variables to represent the state of the assignment

# COMMAND ----------

model = cp_model.CpModel()

stand_indices: dict[str, int] = {stand.stand_id: i for i, stand in enumerate(stands)}
intervals_per_stand: defaultdict[str, list[cp_model.IntervalVar]] = defaultdict(list)
presence_vars_per_flight: defaultdict[str, list[cp_model.BoolVarT]] = defaultdict(list)
assignments: defaultdict[
    tuple[str, str], dict[str, cp_model.BoolVarT | cp_model.IntervalVar]
] = defaultdict(dict)

# COMMAND ----------

# MAGIC %md
# MAGIC Here we iterate over all turns and stands, and create the variables for the (feasible) assignments

# COMMAND ----------

for t_idx, turn in enumerate(turns):
    for s_idx, stand in enumerate(stands):
        if not feasibility[t_idx][s_idx]:
            continue
        is_present: cp_model.BoolVarT = model.NewBoolVar(
            f"{turn.turn_id}_on_{stand.stand_id}"
        )
        var = model.NewOptionalIntervalVar(
            turn.arrival_time,
            turn.departure_time - turn.arrival_time,
            turn.departure_time,
            is_present,
            name=f"stand_{stand.stand_id}_for_{turn.turn_id}",
        )
        intervals_per_stand[stand.stand_id].append(var)
        presence_vars_per_flight[turn.turn_id].append(is_present)
        assignments[(turn.turn_id, stand.stand_id)] = {
            "is_present": is_present,
            "interval": var,
        }

for f_id, bool_vars in presence_vars_per_flight.items():
    model.AddExactlyOne(bool_vars)

for stand, intervals in intervals_per_stand.items():
    model.AddNoOverlap(intervals)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Define our adjacency rules
# MAGIC + This is a list of rules that define the adjacency between stands
# MAGIC + Each rule is a tuple of (stand_a, stand_b) and two corresponding time windows represnting
# MAGIC the intervals during which the stands must not both be occupied simultaneously
# MAGIC + The time window is a tuple of (start_anchor, start_offset_minutes, end_anchor, end_offset_minutes)
# MAGIC + The start and end anchors are the time of the start or end of the turn
# MAGIC + The start and end offsets are the minutes before or after the anchor that the shadow interval should be

# COMMAND ----------


class TimeAnchor(str, Enum):
    ARRIVAL = "ARRIVAL"
    DEPARTURE = "DEPARTURE"


class TimeWindowDefinition(BaseModel):
    """Defines how to calculate a shadow interval relative to a flight."""

    start_anchor: TimeAnchor
    start_offset_minutes: int = 0
    end_anchor: TimeAnchor
    end_offset_minutes: int = 0


class AdjacencyRule(BaseModel):
    """Represents a conflict rule between two stands."""

    rule_id: str
    name: str
    description: Optional[str] = None
    stand_a: str
    stand_b: str
    time_constraint_a: TimeWindowDefinition
    time_constraint_b: TimeWindowDefinition


# COMMAND ----------

# MAGIC %md
# MAGIC ### Helper function to compute the shadow interval times
# MAGIC + This function is used to compute the shadow interval times for a given turn and time window definition

# COMMAND ----------


def _compute_shadow_times(
    arr: int, dep: int, time_constraint: TimeWindowDefinition
) -> tuple[int, int]:
    """Compute shadow interval start and end times based on a TimeWindowDefinition."""
    base_start = arr if time_constraint.start_anchor == TimeAnchor.ARRIVAL else dep
    s_start = base_start + time_constraint.start_offset_minutes

    base_end = arr if time_constraint.end_anchor == TimeAnchor.ARRIVAL else dep
    s_end = base_end + time_constraint.end_offset_minutes

    return s_start, s_end


# COMMAND ----------

# MAGIC %md
# MAGIC ## Apply the adjacency rules
# MAGIC + This function is used to create a new set of interval variables for each adjacency rule
# MAGIC + These are then added to the model as no-overlap constraints

# COMMAND ----------


def apply_adjacency_rules(
    model: cp_model.CpModel,
    turns: list[Turn],
    assignments: dict[
        tuple[str, str], dict[str, cp_model.BoolVarT | cp_model.IntervalVar]
    ],
    adjacency_rules: list[AdjacencyRule],
) -> None:
    """
    Apply adjacency rules by creating shadow intervals that cannot overlap.

    Args:
        model: The CP-SAT model
        turns: List of Turn objects
        assignments: Dict mapping (turn_id, stand_id) to assignment variables from main model
        adjacency_rules: List of AdjacencyRule objects defining the constraints
    """
    turn_times = {
        turn.turn_id: (turn.arrival_time, turn.departure_time) for turn in turns
    }

    for rule in adjacency_rules:
        shadows = []

        for (t_id, s_id), assignment in assignments.items():
            if s_id == rule.stand_a:
                arr, dep = turn_times[t_id]
                s_start, s_end = _compute_shadow_times(arr, dep, rule.time_constraint_a)
                shadow_interval = model.NewOptionalIntervalVar(
                    start=s_start,
                    size=(s_end - s_start),
                    end=s_end,
                    is_present=assignment["is_present"],
                    name=f"Shadow_{rule.name}_{t_id}_{s_id}",
                )
                shadows.append(shadow_interval)
            elif s_id == rule.stand_b:
                arr, dep = turn_times[t_id]
                s_start, s_end = _compute_shadow_times(arr, dep, rule.time_constraint_b)
                shadow_interval = model.NewOptionalIntervalVar(
                    start=s_start,
                    size=(s_end - s_start),
                    end=s_end,
                    is_present=assignment["is_present"],
                    name=f"Shadow_{rule.name}_{t_id}_{s_id}",
                )
                shadows.append(shadow_interval)

        if shadows:
            model.AddNoOverlap(shadows)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Example adjacency rules
# MAGIC + This is a list of simple MARS-style example adjacency rules that exclude simultaneous occupation of two adjacent stands


# COMMAND ----------

# Example adjacency rules using Pydantic models
adjacency_rules: list[AdjacencyRule] = [
    AdjacencyRule(
        rule_id="1",
        name="1L_1C_adjacency",
        stand_a="1L",
        stand_b="1C",
        time_constraint_a=TimeWindowDefinition(
            start_anchor=TimeAnchor.ARRIVAL,
            start_offset_minutes=0,
            end_anchor=TimeAnchor.DEPARTURE,
            end_offset_minutes=0,
        ),
        time_constraint_b=TimeWindowDefinition(
            start_anchor=TimeAnchor.ARRIVAL,
            start_offset_minutes=0,
            end_anchor=TimeAnchor.DEPARTURE,
            end_offset_minutes=0,
        ),
    ),
    AdjacencyRule(
        rule_id="2",
        name="2L_2C_adjacency",
        stand_a="2L",
        stand_b="2C",
        time_constraint_a=TimeWindowDefinition(
            start_anchor=TimeAnchor.ARRIVAL,
            start_offset_minutes=0,
            end_anchor=TimeAnchor.DEPARTURE,
            end_offset_minutes=0,
        ),
        time_constraint_b=TimeWindowDefinition(
            start_anchor=TimeAnchor.ARRIVAL,
            start_offset_minutes=0,
            end_anchor=TimeAnchor.DEPARTURE,
            end_offset_minutes=0,
        ),
    ),
]

# COMMAND ----------

apply_adjacency_rules(model, turns, assignments, adjacency_rules)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Solve the model
# MAGIC + Now we can use the CP-SAT solver to find a feasible solution

# COMMAND ----------

solver = cp_model.CpSolver()
status = solver.solve(model)

if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print("Solution Found!")

    for turn in turns:
        for stand in stands:
            if (turn.turn_id, stand.stand_id) in assignments:
                if solver.Value(
                    assignments[(turn.turn_id, stand.stand_id)]["is_present"]
                ):
                    print(
                        f"  Turn {turn.turn_id} (Flight {turn.flight_id}) assigned to -> Stand {stand.stand_id}"
                    )

elif status == cp_model.INFEASIBLE:
    print("No solution found. The model is infeasible.")
else:
    print("Solver status:", solver.StatusName())
