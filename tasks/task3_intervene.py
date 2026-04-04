"""
Task 3: Intervention Planning — propose gene interventions to reduce AD risk.

Deterministic: intervention effect is computed from the patient's actual gene
expression level, not random. Higher expression of a risk gene = larger reduction
when downregulated. Lower expression of a protective gene = larger gain when upregulated.
"""

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
        genes = patient["genes"]
        # Surface expression z-scores to guide agent reasoning
        # (how far each gene deviates from healthy baseline of 1.0)
        expression_deviation = {
            g: round(v - 1.0, 3) for g, v in genes.items()
        }
        # Highlight which genes are most dysregulated
        top_dysregulated = sorted(
            expression_deviation.items(), key=lambda x: abs(x[1]), reverse=True
        )[:5]

        return {
            "patient_id": patient["id"],
            "risk_score": current_risk,
            "target_risk": TARGET_RISK,
            "risk_reduction_needed": max(0.0, round(current_risk - TARGET_RISK, 2)),
            "age": patient["age"],
            "stage": patient["stage"],
            "gene_expression": dict(genes),
            "expression_deviation_from_healthy": dict(expression_deviation),
            "top_dysregulated_genes": [g for g, _ in top_dysregulated],
            "step": step,
            "max_steps": max_steps,
            "task": "intervention_planning",
            "instructions": (
                f"Reduce this patient's Alzheimer's risk below {TARGET_RISK}. "
                f"Current risk: {current_risk:.1f}. "
                "Choose a gene intervention based on expression levels: "
                "downregulate overexpressed risk genes (deviation > 0) or "
                "upregulate underexpressed protective genes (deviation < 0). "
                "Effect magnitude scales with expression deviation from baseline. "
                "Action format: downregulate:GENE  or  upregulate:GENE"
            ),
        }

    @staticmethod
    def parse_action(action_str: str) -> dict:
        action_str = action_str.strip()
        for direction in ("downregulate", "upregulate"):
            if action_str.startswith(f"{direction}:"):
                gene = action_str[len(direction) + 1:].strip().upper()
                if gene:
                    return {"direction": direction, "gene": gene, "valid": True}
        return {
            "direction": "",
            "gene": "",
            "valid": False,
            "error": f"Invalid action: '{action_str}'. Use downregulate:GENE or upregulate:GENE",
        }

    @staticmethod
    def apply_intervention(
        current_risk: float, gene: str, direction: str, patient_genes: dict
    ) -> float:
        """
        Deterministic intervention: effect magnitude is derived from the patient's
        actual gene expression level — not random.

        Downregulating an overexpressed risk gene produces larger risk reduction.
        Upregulating an underexpressed protective gene produces larger improvement.
        Biologically inappropriate interventions have minimal effect.
        """
        gene = gene.upper()
        expr = patient_genes.get(gene, 1.0)  # baseline = 1.0 (healthy)

        if direction == "downregulate" and gene in HIGH_IMPACT_DOWN:
            # Effect scales with how overexpressed the gene is (expr > 1.0 = bad)
            deviation = max(0.0, expr - 1.0)
            delta = 5.0 + deviation * 7.0   # range ~5–18 depending on expression
        elif direction == "upregulate" and gene in HIGH_IMPACT_UP:
            # Effect scales with how underexpressed the gene is (expr < 1.0 = bad)
            deviation = max(0.0, 1.5 - expr)
            delta = 3.0 + deviation * 6.0   # range ~3–12 depending on expression
        elif direction == "downregulate":
            # Downregulating a non-risk gene: small effect, can even be harmful
            delta = max(0.5, expr * 1.5)    # range ~0.5–5
        else:
            # Upregulating a non-protective gene: minimal benefit
            delta = max(0.5, (1.0 / max(0.1, expr)) * 1.0)  # range ~0.5–2

        new_risk = current_risk - delta
        return round(max(5.0, min(100.0, new_risk)), 2)
