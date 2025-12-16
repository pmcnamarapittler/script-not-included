# Script Not Included

Fantasy projections assume a neutral game script.  
Real games don’t work like that.

**Script Not Included** is a context-aware fantasy football analysis project that adjusts baseline projections using season trends, game conditions, and interpretable performance signals — helping fantasy managers make better start/sit decisions with explanations they can trust.

---

## Why this exists

Most fantasy projections treat every game as if it unfolds under identical conditions.

In reality:
- Weather changes
- Playing surface matters
- Roles shift week to week
- Fatigue shows up late
- Film tells a different story than the box score

**Script Not Included** makes those assumptions explicit — and adjusts for them.

This project focuses on *interpretation*, not prediction, and prioritizes transparency over black-box scoring.

---

## What it does (v1)

**Script Not Included (RB v1)** supports fantasy analysis for **running backs** only.

Given a fantasy roster, the system:
1. Starts from a baseline projection
2. Applies context-aware adjustments
3. Outputs an adjusted projection with a range and confidence level
4. Explains *why* the projection changed
5. Provides start/sit and flex guidance

---

## Core inputs

### Baseline
- Manual or external fantasy projection (e.g., ESPN)
- League format (PPR / Half / Standard)

### Season & role trends
- Snap share trend
- Carry and target trends
- Red zone usage
- Role stability indicators

### Game context
- Temperature and weather
- Turf vs grass
- Short week / travel
- Declared return-from-injury status (binary only)

### Performance signals (optional, experimental)
- Interpretable movement proxies derived from video clips
  - Burst after handoff
  - Direction change efficiency
  - Late-game fatigue decay

> Note: These signals are used to *contextualize performance*, not to infer health or predict injuries.

---

## Outputs

For each running back, the system produces:
- **Baseline projection**
- **Adjusted projection**
- **Projected range (floor–ceiling)**
- **Confidence level** (Low / Medium / High)
- **Explanation bullets** describing the adjustment
- **Start / Sit / Flex recommendations**

All outputs are explainable and traceable to specific inputs.

---

## Design principles

- **Context-aware**  
  Projections change when conditions change.

- **Explainable by default**  
  Every adjustment includes a reason.

- **Conservative by design**  
  Small deltas, widened ranges, calibrated confidence.

- **No medical inference**  
  This project does not diagnose injuries or assess injury risk.

- **Human judgment first**  
  The system supports decisions — it does not replace them.

---

## What this is *not*

- Not an injury prediction system  
- Not a betting model  
- Not a guarantee of fantasy outcomes  
- Not a replacement for watching the game

---

## Architecture (high level)

![Architecture diagram](docs/architecture.svg)

---

## Roadmap

**v1 (current)**
- Running backs only
- Manual roster input
- Context-aware adjustment engine
- Explainable projections and recommendations

**v2**
- Wide receivers
- Expanded film-based signals
- Weekly evaluation vs baseline projections

**v3**
- Quarterbacks (experimental)
- Cross-position roster optimization
- Longitudinal season narratives

---

## Ethics & limitations

Script Not Included intentionally avoids:
- Medical claims
- Injury risk prediction
- Deterministic language

All insights are framed as *interpretations* based on observed trends and conditions.  
Actual outcomes remain uncertain — as they should.

---

## Status

This project is an active side project and portfolio piece focused on:
- Applied systems design
- Human-centered AI
- Interpretable analytics
- Computer vision as a supporting signal, not a black box

---

## Author
Paige McNamara-Pittler...a fantasy football player who got tired of arguing about the eye test without receipts.
