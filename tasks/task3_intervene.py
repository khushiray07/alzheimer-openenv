"""Task 3: Intervention Planning — propose gene interventions to reduce AD risk."""

import random

TARGET_RISK = 40.0

HIGH_IMPACT_DOWN = {"APOE", "APP", "PSEN1", "BACE1", "MAPT"}
HIGH_IMPACT_UP = {"TREM2", "CLU"}


class Task3InterventionPlanning:
    task_id = 3
    name = "intervention_planning"
    difficulty = "hard"
    max_steps = 8

    @staticmethod
    def get_observation(patient: dict, step: int, max_steps: int, current_risk: float) -> dict:
        return {
            "patient_id": patient["id"],
            "risk_score": current_risk,
            "target_risk": TARGET_RISK,
            "risk_reduction_needed": max(0.0, round(current_risk - TARGET_RISK, 2)),
            "age": patient["age"],
            "gene_expression": dict(patient["genes"]),
            "step": step,
            "max_steps": max_steps,
            "task": "intervention_planning",
            "available_genes": list(patient["genes"].keys()),
            "instructions": (
                f"Propose a gene intervention to reduce this patient's Alzheimer's risk "
                f"below {TARGET_RISK}. Current risk: {current_risk:.1f}. "
                "Action format: downregulate:GENE  or  upregulate:GENE  "
                "e.g. downregulate:APOE"
            ),
        }

    @staticmethod
    def parse_action(action_str: str) -> dict:
        action_str = action_str.strip()
        for direction in ("downregulate", "upregulate"):
            if action_str.startswith(f"{direction}:"):
                gene = action_str[len(direction) + 1:].strip()
                if gene:
                    return {"direction": direction, "gene": gene, "valid": True}
        return {"direction": "", "gene": "", "valid": False, "error": f"Invalid action: '{action_str}'"}

    @staticmethod
    def apply_intervention(current_risk: float, gene: str, direction: str, patient_genes: dict) -> float:
        gene = gene.upper()
        if direction == "downregulate" and gene in HIGH_IMPACT_DOWN:
            delta = random.uniform(6, 14)
        elif direction == "upregulate" and gene in HIGH_IMPACT_UP:
            delta = random.uniform(4, 10)
        else:
            delta = random.uniform(1, 4)
        new_risk = current_risk - delta
        return round(max(5.0, min(100.0, new_risk)), 2)
