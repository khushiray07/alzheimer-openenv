"""Grader 3: Score for Task 3 — Intervention Planning."""

TARGET = 40.0


class Grader3:
    @staticmethod
    def grade(old_risk: float, new_risk: float, parsed_action: dict, step: int, max_steps: int) -> float:
        if not parsed_action.get("valid"):
            return 0.05

        reduction = old_risk - new_risk
        if reduction <= 0:
            return 0.05

        max_reduction = old_risk - TARGET
        if max_reduction <= 0:
            return 1.0  # already at or below target

        progress = min(1.0, reduction / max_reduction)

        if new_risk <= TARGET:
            return 1.0

        efficiency = 1.0 - (step / max_steps) * 0.1
        return round(min(0.99, progress * efficiency), 3)
