"""Task 2: Biomarker Ranking — rank top Alzheimer's biomarker genes."""

GOLD_RANKING = ["APOE", "APP", "PSEN1", "MAPT", "BACE1", "TREM2", "CLU", "CR1", "BIN1", "PICALM"]


class Task2BiomarkerRanking:
    task_id = 2
    name = "biomarker_ranking"
    difficulty = "medium"
    max_steps = 5

    @staticmethod
    def get_observation(patient: dict, step: int, max_steps: int) -> dict:
        return {
            "patient_id": patient["id"],
            "risk_score": patient["risk_score"],
            "age": patient["age"],
            "gene_expression": dict(patient["genes"]),
            "step": step,
            "max_steps": max_steps,
            "task": "biomarker_ranking",
            "available_genes": list(patient["genes"].keys()),
            "instructions": (
                "Rank the top 3 Alzheimer's biomarker genes from this patient's data "
                "in order of importance. "
                "Action format: rank:[GENE1,GENE2,GENE3]  e.g. rank:[APOE,APP,PSEN1]"
            ),
        }

    @staticmethod
    def parse_action(action_str: str) -> dict:
        action_str = action_str.strip()
        if action_str.startswith("rank:[") and action_str.endswith("]"):
            inner = action_str[len("rank:["):-1]
            genes = [g.strip() for g in inner.split(",") if g.strip()]
            if len(genes) >= 1:
                return {"ranking": genes, "valid": True}
        return {"ranking": [], "valid": False, "error": f"Invalid action: '{action_str}'"}

    @staticmethod
    def compute_overlap(predicted_ranking: list, gold: list, k: int = 3) -> float:
        k = min(k, len(predicted_ranking), len(gold))
        if k == 0:
            return 0.0
        overlap = len(set(predicted_ranking[:k]) & set(gold[:k]))
        return round(overlap / k, 4)
