"""
Grader 3: Score for Task 3 — Intervention Planning.

Deterministic scoring based on risk reduction progress toward target.
Rewards choosing biologically appropriate genes (high-expression AD genes
benefit most from downregulation; low-expression protective genes from upregulation).
"""

TARGET = 40.0

# Genes where expression level matters for intervention scoring
AD_RISK_GENES = {"APOE", "APP", "PSEN1", "BACE1", "MAPT"}
PROTECTIVE_GENES = {"TREM2", "CLU"}


class Grader3:
    @staticmethod
    def grade(
        old_risk: float,
        new_risk: float,
        parsed_action: dict,
        step: int,
        max_steps: int,
        patient_genes: dict = None,
    ) -> float:
        if not parsed_action.get("valid"):
            return 0.05

        reduction = old_risk - new_risk
        gene = parsed_action.get("gene", "").upper()
        direction = parsed_action.get("direction", "")

        if reduction <= 0:
            # Penalize harmful interventions more than neutral ones
            if new_risk > old_risk:
                return 0.02  # Made things worse
            return 0.05      # No effect

        # Biological appropriateness bonus
        # Choosing the right type of gene for the intervention = extra reward
        bio_bonus = 0.0
        if patient_genes:
            expr = patient_genes.get(gene, 1.0)
            if direction == "downregulate" and gene in AD_RISK_GENES:
                # Higher expression → more benefit from downregulation
                bio_bonus = min(0.10, (expr - 1.0) * 0.08)
            elif direction == "upregulate" and gene in PROTECTIVE_GENES:
                # Lower expression → more benefit from upregulation
                bio_bonus = min(0.10, max(0, (1.5 - expr) * 0.10))

        # Progress toward target
        max_reduction = old_risk - TARGET
        if max_reduction <= 0:
            return 0.99  # Already at or below target

        progress = min(1.0, reduction / max_reduction)

        # Reached target: full reward regardless of steps
        if new_risk <= TARGET:
            return round(max(0.01, min(0.99, 0.95 + bio_bonus)), 3)

        # Step efficiency: earlier progress = better score
        # Step 1 of 8: efficiency=1.0, Step 8 of 8: efficiency=0.88
        efficiency = 1.0 - (step / max_steps) * 0.12

        base = progress * efficiency * 0.90
        return round(max(0.01, min(0.99, base + bio_bonus)), 3)
