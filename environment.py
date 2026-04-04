"""AlzheimerEnv — core environment for Alzheimer's risk prediction and intervention."""

import random
from typing import Optional

from tasks.task1_classify import Task1RiskClassification
from tasks.task2_biomarker import Task2BiomarkerRanking
from tasks.task3_intervene import Task3InterventionPlanning
from graders.grader1 import Grader1
from graders.grader2 import Grader2
from graders.grader3 import Grader3

# ---------------------------------------------------------------------------
# Hardcoded synthetic patient dataset — no external files needed
# ---------------------------------------------------------------------------
PATIENTS = [
    {
        "id": "PT-001", "label": 1, "risk_score": 78.4, "age": 72, "stage": "Moderate",
        "genes": {
            "APP": 2.31, "BACE1": 1.84, "PSEN1": 2.13, "MAPT": 1.92,
            "APOE": 2.71, "CLU": 1.55, "BIN1": 0.88, "TREM2": 0.72,
            "CR1": 1.43, "PICALM": 1.22,
        },
    },
    {
        "id": "PT-002", "label": 0, "risk_score": 23.1, "age": 68, "stage": "Healthy",
        "genes": {
            "APP": 0.91, "BACE1": 1.05, "PSEN1": 0.82, "MAPT": 1.02,
            "APOE": 1.14, "CLU": 1.01, "BIN1": 0.97, "TREM2": 1.03,
            "CR1": 0.94, "PICALM": 1.01,
        },
    },
    {
        "id": "PT-003", "label": 1, "risk_score": 55.9, "age": 75, "stage": "Early",
        "genes": {
            "APP": 1.63, "BACE1": 1.41, "PSEN1": 1.52, "MAPT": 1.44,
            "APOE": 1.82, "CLU": 1.28, "BIN1": 0.92, "TREM2": 0.84,
            "CR1": 1.31, "PICALM": 1.15,
        },
    },
    {
        "id": "PT-004", "label": 0, "risk_score": 18.7, "age": 65, "stage": "Healthy",
        "genes": {
            "APP": 0.85, "BACE1": 0.97, "PSEN1": 0.79, "MAPT": 0.95,
            "APOE": 1.02, "CLU": 0.98, "BIN1": 1.04, "TREM2": 1.08,
            "CR1": 0.91, "PICALM": 0.99,
        },
    },
    {
        "id": "PT-005", "label": 1, "risk_score": 88.2, "age": 79, "stage": "Severe",
        "genes": {
            "APP": 3.12, "BACE1": 2.67, "PSEN1": 2.89, "MAPT": 2.71,
            "APOE": 3.44, "CLU": 2.01, "BIN1": 0.71, "TREM2": 0.58,
            "CR1": 1.87, "PICALM": 1.64,
        },
    },
]

TASK_REGISTRY = {
    1: {"handler": Task1RiskClassification, "grader": Grader1, "max_steps": 3,
        "name": "risk_classification", "difficulty": "easy"},
    2: {"handler": Task2BiomarkerRanking, "grader": Grader2, "max_steps": 5,
        "name": "biomarker_ranking", "difficulty": "medium"},
    3: {"handler": Task3InterventionPlanning, "grader": Grader3, "max_steps": 8,
        "name": "intervention_planning", "difficulty": "hard"},
}


class AlzheimerEnv:
    """OpenEnv-compatible environment for Alzheimer's disease RL tasks."""

    def __init__(self):
        self.task_id: int = 1
        self.patient: dict = PATIENTS[0]
        self.step_count: int = 0
        self.max_steps: int = 3
        self.current_risk: float = 0.0
        self.cumulative_reward: float = 0.0
        self.done: bool = False
        self.history: list = []
        self._initialized: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, task_id: int = 1, patient_id: Optional[str] = None) -> dict:
        """Reset the environment for a new episode."""
        if task_id not in TASK_REGISTRY:
            raise ValueError(f"Unknown task_id {task_id}. Choose from {list(TASK_REGISTRY.keys())}.")

        task_cfg = TASK_REGISTRY[task_id]
        self.task_id = task_id
        self.max_steps = task_cfg["max_steps"]
        self.step_count = 0
        self.cumulative_reward = 0.0
        self.done = False
        self.history = []

        # Select patient
        if patient_id is not None:
            matches = [p for p in PATIENTS if p["id"] == patient_id]
            if not matches:
                raise ValueError(f"Patient '{patient_id}' not found.")
            self.patient = matches[0]
        else:
            self.patient = random.choice(PATIENTS)

        self.current_risk = float(self.patient["risk_score"])
        self._initialized = True
        return self.state()

    def step(self, action: str) -> dict:
        """Execute one action in the environment."""
        if not self._initialized:
            raise RuntimeError("Call reset() before step().")
        if self.done:
            return {
                "observation": self.state(),
                "reward": 0.0,
                "done": True,
                "info": {"error": "Episode already done. Call reset()."},
            }

        task_cfg = TASK_REGISTRY[self.task_id]
        handler = task_cfg["handler"]
        grader = task_cfg["grader"]

        self.step_count += 1
        reward = 0.0
        info = {}

        if self.task_id == 1:
            parsed = handler.parse_action(action)
            info = handler.compute_info(self.patient, parsed)
            reward = grader.grade(self.patient, parsed)

        elif self.task_id == 2:
            parsed = handler.parse_action(action, available_genes=list(self.patient["genes"].keys()))
            reward = grader.grade(parsed, self.step_count, self.max_steps)
            info = {
                "ranking": parsed.get("ranking", []),
                "invalid_genes": parsed.get("invalid_genes", []),
                "valid": parsed.get("valid", False),
                "overlap": Task2BiomarkerRanking.compute_overlap(
                    parsed.get("ranking", []),
                    ["APOE", "APP", "PSEN1", "MAPT", "BACE1"],
                    k=3,
                ),
            }

        elif self.task_id == 3:
            parsed = handler.parse_action(action)
            old_risk = self.current_risk
            if parsed.get("valid"):
                self.current_risk = handler.apply_intervention(
                    old_risk, parsed["gene"], parsed["direction"], self.patient["genes"]
                )
            new_risk = self.current_risk
            reward = grader.grade(
                old_risk, new_risk, parsed,
                self.step_count, self.max_steps,
                patient_genes=self.patient["genes"],
            )
            info = {
                "old_risk": old_risk,
                "new_risk": new_risk,
                "risk_delta": round(old_risk - new_risk, 2),
                "target_risk": 40.0,
                "gene": parsed.get("gene", ""),
                "direction": parsed.get("direction", ""),
                "valid": parsed.get("valid", False),
            }

        reward = float(round(max(0.0, min(1.0, reward)), 4))
        self.cumulative_reward += reward

        # Determine done
        if self.step_count >= self.max_steps:
            self.done = True
        if self.task_id == 3 and self.current_risk <= 40.0:
            self.done = True

        self.history.append({
            "step": self.step_count,
            "action": action,
            "reward": reward,
            "info": info,
        })

        observation = self._build_observation()
        return {
            "observation": observation,
            "reward": reward,
            "done": self.done,
            "info": info,
        }

    def state(self) -> dict:
        """Return current environment state."""
        return {
            "task_id": self.task_id,
            "patient_id": self.patient["id"],
            "risk_score": self.current_risk,
            "gene_expression": dict(self.patient["genes"]),
            "step": self.step_count,
            "max_steps": self.max_steps,
            "cumulative_reward": round(self.cumulative_reward, 4),
            "done": self.done,
            "history": self.history,
        }

    def list_tasks(self) -> list:
        """Return all task definitions."""
        return [
            {
                "id": tid,
                "name": cfg["name"],
                "difficulty": cfg["difficulty"],
                "max_steps": cfg["max_steps"],
            }
            for tid, cfg in TASK_REGISTRY.items()
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_observation(self) -> dict:
        """Build the observation dict for the current step."""
        if self.task_id == 1:
            return Task1RiskClassification.get_observation(
                self.patient, self.step_count, self.max_steps
            )
        elif self.task_id == 2:
            return Task2BiomarkerRanking.get_observation(
                self.patient, self.step_count, self.max_steps
            )
        elif self.task_id == 3:
            return Task3InterventionPlanning.get_observation(
                self.patient, self.step_count, self.max_steps, self.current_risk
            )
        return self.state()
