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


def _clamp(score: float) -> float:
    """Ensure score is strictly between 0 and 1 (exclusive)."""
    return float(round(max(0.01, min(0.99, score)), 4))


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
            return _clamp(0.05)

        reduction = old_risk - new_risk
        gene = parsed_action.get("gene", "").upper()
        direction = parsed_action.get("direction", "")

        if reduction <= 0:
            if new_risk > old_risk:
                return _clamp(0.02)
            return _clamp(0.05)

        bio_bonus = 0.0
        if patient_genes:
            expr = patient_genes.get(gene, 1.0)
            if direction == "downregulate" and gene in AD_RISK_GENES:
                bio_bonus = min(0.10, (expr - 1.0) * 0.08)
            elif direction == "upregulate" and gene in PROTECTIVE_GENES:
                bio_bonus = min(0.10, max(0, (1.5 - expr) * 0.10))

        max_reduction = old_risk - TARGET
        if max_reduction <= 0:
            return _clamp(0.95)

        progress = min(1.0, reduction / max_reduction)

        if new_risk <= TARGET:
            return _clamp(0.90 + bio_bonus)

        efficiency = 1.0 - (step / max_steps) * 0.12
        base = progress * efficiency * 0.85
        return _clamp(base + bio_bonus)
