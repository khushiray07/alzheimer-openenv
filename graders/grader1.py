"""
Grader 1: Score for Task 1 — Risk Classification.

Deterministic scoring: reward is a function of risk_score and correctness,
not random. Same inputs always produce the same reward.
"""


def _clamp(score: float) -> float:
    """Ensure score is strictly between 0 and 1 (exclusive)."""
    return float(round(max(0.01, min(0.99, score)), 4))


class Grader1:
    @staticmethod
    def grade(patient: dict, parsed_action: dict) -> float:
        if not parsed_action.get("valid"):
            return _clamp(0.05)

        true_label = "AD" if patient["label"] == 1 else "Control"
        predicted = parsed_action.get("prediction", "")
        risk = patient["risk_score"]

        if predicted == true_label:
            if true_label == "AD":
                base = 0.75 + min(0.24, (risk - 30) / 400)
            else:
                base = 0.75 + min(0.24, (70 - risk) / 400)
            return _clamp(base)
        else:
            if risk >= 70 or risk <= 25:
                return _clamp(0.02)
            elif 40 < risk < 65:
                return _clamp(0.10)
            else:
                return _clamp(0.05)
