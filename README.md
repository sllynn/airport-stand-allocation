# Airport Stand Allocation with Adjacency Constraints

A demonstration of using Google OR-Tools CP-SAT solver to allocate aircraft to parking stands while respecting adjacency constraints.

## Overview

This demo solves the **stand allocation problem**: assigning flight turns (ground time between arrival and departure) to aircraft parking stands at an airport, subject to:

1. **Feasibility constraints** — Not all aircraft can use all stands (e.g., due to size restrictions)
2. **No-overlap constraints** — A stand cannot have two aircraft at the same time
3. **Adjacency constraints** — Adjacent stands may have restrictions on simultaneous use (e.g., large aircraft wingspan blocking neighboring stands)

## Key Concepts

### Turns
A **turn** represents an aircraft's ground time at the airport:
- `arrival_time` — When the aircraft arrives
- `departure_time` — When it departs
- `flight_id` — The associated flight number

### Stands
A **stand** is a parking position for aircraft, identified by `stand_id` (e.g., "1L", "1C", "2L").

### Adjacency Rules
An **adjacency rule** defines when two stands cannot be simultaneously occupied. Each rule specifies:
- Two stands (`stand_a`, `stand_b`)
- Time windows for each stand, defined relative to arrival/departure times

This allows flexible constraints like:
- "Stand 1L and 1C cannot both be occupied at the same time"
- "Stand 2L blocks Stand 2C from 15 minutes before departure until 10 minutes after"

## How It Works

The solver uses **optional interval variables** to model potential stand assignments:

1. For each feasible (turn, stand) pair, create a Boolean "presence" variable and an interval variable
2. Add `ExactlyOne` constraints so each turn is assigned to exactly one stand
3. Add `NoOverlap` constraints so no stand has overlapping turns
4. For adjacency rules, create "shadow intervals" that inherit the presence variable from the main assignment and add `NoOverlap` constraints between adjacent stands

## Running the Demo

### Prerequisites

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.lock
```

### Run locally

```bash
python adjacency-demo.py
```

### Run on Databricks

The file is formatted as a Databricks notebook and can be imported directly.

## Example Output

```
Solution Found!
  Turn 1 (Flight FR13) assigned to -> Stand 1C
  Turn 2 (Flight FR42) assigned to -> Stand 1L
  Turn 3 (Flight FR66) assigned to -> Stand 2R
  Turn 4 (Flight FR99) assigned to -> Stand 2C
```

## Project Structure

```
├── adjacency-demo.py    # Main demo notebook
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Dependencies

- **ortools** — Google OR-Tools for constraint programming
- **numpy** — Feasibility matrix handling
- **pydantic** — Data validation and type hints

