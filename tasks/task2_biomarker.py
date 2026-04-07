"""
Task 2: Biomarker Ranking — rank top Alzheimer's biomarker genes.

Gold standard is based on population-level GWAS and clinical research evidence,
not patient-specific expression. The agent must apply domain knowledge about
which genes are most diagnostically important across the AD population.
"""

# Population-level gold standard based on GWAS evidence and clinical literature
# (not patient-specific — reflects diagnostic importance across the AD population)
GOLD_RANKING = ["APOE", "APP", "PSEN1", "MAPT", "BACE1", "TREM2", "CLU", "CR1", "BIN1", "PICALM"]


class Task2BiomarkerRanking:
    task_id = 2
    name = "biomarker_ranking"
    difficulty = "medium"
    max_steps = 5

    @staticmethod
    def get_observation(patient: dict, step: int, max_steps: int) -> dict:
        genes = patient["genes"]
        # Sort genes by expression level to help agent reason about which are dysregulated
        sorted_by_expression = sorted(genes.items(), key=lambda x: x[1], reverse=True)

        return {
            "patient_id": patient["id"],
            "risk_score": patient["risk_score"],
            "age": patient["age"],
            "stage": patient["stage"],
            "gene_expression": dict(genes),
            "genes_by_expression_level": [g for g, _ in sorted_by_expression],
            "step": step,
            "max_steps": max_steps,
            "task": "biomarker_ranking",
            "available_genes": list(genes.keys()),
            "instructions": (
                "Rank the top 3 Alzheimer's Disease biomarker genes in order of "
                "population-level diagnostic importance (not just this patient's expression). "
                "Consider known genetic risk factors, amyloid/tau pathways, and GWAS evidence. "
                "Action format: rank:[GENE1,GENE2,GENE3]  e.g. rank:[APOE,APP,PSEN1]  "
                "Genes must be from the available_genes list."
            ),
        }

    @staticmethod
    def parse_action(action_str: str, available_genes: list = None) -> dict:
        action_str = action_str.strip()
        if action_str.startswith("rank:[") and action_str.endswith("]"):
            inner = action_str[len("rank:["):-1]
            genes = [g.strip().upper() for g in inner.split(",") if g.strip()]
            if len(genes) >= 1:
                # Validate genes are from available list if provided
                if available_genes:
                    valid_genes = [g for g in genes if g in [ag.upper() for ag in available_genes]]
                    invalid = [g for g in genes if g not in [ag.upper() for ag in available_genes]]
                else:
                    valid_genes = genes
                    invalid = []
                return {
                    "ranking": valid_genes,
                    "invalid_genes": invalid,
                    "valid": len(valid_genes) >= 1,
                }
        return {
            "ranking": [],
            "valid": False,
            "error": f"Invalid action: '{action_str}'. Use rank:[GENE1,GENE2,GENE3]",
        }

    @staticmethod
    def compute_overlap(predicted_ranking: list, gold: list = None, k: int = 3) -> float:
        """Compute normalized set overlap between predicted top-k and gold top-k."""
        if gold is None:
            gold = GOLD_RANKING
        k = min(k, len(predicted_ranking), len(gold))
        if k == 0:
            return 0.01
        overlap = len(set(predicted_ranking[:k]) & set(gold[:k]))
        raw = overlap / k
        # Clamp to strict (0, 1) interval
        return float(round(max(0.01, min(0.99, raw)), 4))
