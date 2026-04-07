"""
Grader 1: Score for Task 1 — Risk Classification.

Deterministic scoring: reward is a function of risk_score and correctness,
not random. Same inputs always produce the same reward.
"""


class Grader1:
    @staticmethod
    def grade(patient: dict, parsed_action: dict) -> float:
        if not parsed_action.get("valid"):
            # Partial credit based on how close the format was
            return 0.05

        true_label = "AD" if patient["label"] == 1 else "Control"
        predicted = parsed_action.get("prediction", "")
        risk = patient["risk_score"]

        if predicted == true_label:
            if true_label == "AD":
                # Higher risk = more obvious case = higher reward
                # Risk 88 → ~0.98, Risk 55 → ~0.88, Risk 40 → ~0.82
                base = 0.75 + min(0.24, (risk - 30) / 400)
            else:
                # Lower risk = more obvious Control case = higher reward
                # Risk 18 → ~0.97, Risk 23 → ~0.95
                base = 0.75 + min(0.24, (70 - risk) / 400)
            return round(max(0.01, min(0.99, base)), 3)
        else:
            # Wrong prediction — penalty proportional to how obvious the case was
            if risk >= 70 or risk <= 25:
                return 0.02   # Very obvious case, large penalty
            elif 40 < risk < 65:
                return 0.10   # Ambiguous zone, smaller penalty
            else:
                return 0.05
