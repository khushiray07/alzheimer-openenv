# AlzheimerEnv — OpenEnv Submission

## Overview

**AlzheimerEnv** is a real-world OpenEnv reinforcement learning environment that challenges an AI agent to predict and intervene on Alzheimer's disease risk using synthetic gene expression data. The agent observes patient transcriptomic profiles and must correctly classify disease status, identify key biomarkers, and propose targeted gene interventions to reduce patient risk below a clinical threshold.

This environment simulates decision-making at the intersection of genomics and precision medicine, offering graded rewards that reflect clinical accuracy and planning efficiency.

---

## Environment Description

### Observation Space

Each observation is a dictionary with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `patient_id` | `str` | Unique patient identifier (e.g. "PT-001") |
| `risk_score` | `float` | Current Alzheimer's risk score (0–100) |
| `gene_expression` | `dict` | Expression levels for 10 key genes |
| `step` | `int` | Current step within the episode |
| `max_steps` | `int` | Maximum steps allowed for this task |

### Action Space

Actions are strings with one of the following formats:

| Format | Example | Used In |
|--------|---------|---------|
| `classify:LABEL` | `classify:AD` | Task 1 |
| `rank:[G1,G2,G3]` | `rank:[APOE,APP,PSEN1]` | Task 2 |
| `downregulate:GENE` | `downregulate:APOE` | Task 3 |
| `upregulate:GENE` | `upregulate:TREM2` | Task 3 |

### Reward Range

All rewards are floats in the range **[0.0, 1.0]**.

---

## Tasks

| Task ID | Name | Difficulty | Max Steps | Description |
|---------|------|------------|-----------|-------------|
| 1 | `risk_classification` | Easy | 3 | Classify patient as AD or Control based on gene expression |
| 2 | `biomarker_ranking` | Medium | 5 | Rank top 3 Alzheimer's biomarker genes from patient profile |
| 3 | `intervention_planning` | Hard | 8 | Propose gene interventions to reduce risk below 40.0 |

---

## Setup & Installation

```bash
pip install -r requirements.txt
python server.py
```

The server starts on port **7860**.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check — returns running status |
| `GET` | `/health` | Detailed health info with version |
| `GET` | `/tasks` | List all available tasks |
| `GET` | `/state` | Return current environment state |
| `POST` | `/reset` | Reset environment: `{"task_id": 1, "patient_id": null}` |
| `POST` | `/step` | Execute action: `{"action": "classify:AD"}` |

---

## Running Inference

```bash
export API_BASE_URL=https://ap
i.anthropic.com
export MODEL_NAME=claude-haiku-4-5-20251001
export HF_TOKEN=your_token
python inference.py
```

The script runs all 3 tasks sequentially and prints structured logs:

```
[START] task_id=1 patient="PT-001" initial_risk=78.4 max_steps=3
[STEP] step=1 action="classify:AD" reasoning="High APOE expression indicates AD" reward=0.923
[END] task_id=1 total_reward=0.923 avg_reward=0.923 score=92.3 status=SUCCESS
...
=== INFERENCE COMPLETE ===
Tasks run: 3
Total score: 275.4 / 300
Status: PASS
```

---

## Evaluation Criteria

### Task 1 — Risk Classification
- **Correct prediction**: reward sampled from `[0.85, 1.0]`
- **Correct on ambiguous case** (risk 40–65): reward from `[0.78, 0.92]`
- **Wrong prediction**: reward from `[0.0, 0.12]`
- **Invalid action**: reward = `0.05`

### Task 2 — Biomarker Ranking
- Reward = `overlap_score × 0.85 + progress_bonus`
- Overlap is computed over top-3 predicted genes vs gold standard `[APOE, APP, PSEN1, MAPT, BACE1]`
- Progress bonus = `(step / max_steps) × 0.15`

### Task 3 — Intervention Planning
- Reward proportional to risk reduction progress toward target (40.0)
- High-impact genes (`APOE, APP, PSEN1, BACE1, MAPT` downregulation; `TREM2, CLU` upregulation) produce larger risk reductions
- Reaching target yields reward = `1.0`
- Step efficiency bonus applied: earlier success = higher reward

---

## Gene Reference

| Gene | Role | Expression in AD |
|------|------|-----------------|
| APOE | Major genetic risk factor | Elevated |
| APP | Amyloid precursor protein | Elevated |
| PSEN1 | Presenilin-1, amyloid processing | Elevated |
| BACE1 | Beta-secretase | Elevated |
| MAPT | Tau protein | Elevated |
| TREM2 | Microglial immune receptor | Reduced |
| CLU | Clusterin, neuroprotective | Variable |
| BIN1 | Bridging integrator | Variable |
| CR1 | Complement receptor | Elevated |
| PICALM | Clathrin assembly protein | Variable |
