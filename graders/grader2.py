"""Grader 2: Score for Task 2 — Biomarker Ranking."""


class Grader2:
    GOLD = ["APOE", "APP", "PSEN1", "MAPT", "BACE1"]

    @staticmethod
    def grade(parsed_action: dict, step: int, max_steps: int) -> float:
        if not parsed_action.get("valid"):
            return 0.05

        ranking = parsed_action.get("ranking", [])
        gold = Grader2.GOLD
        k = min(3, len(ranking))
        overlap = len(set(ranking[:k]) & set(gold[:k])) / k if k > 0 else 0
        progress_bonus = (step / max_steps) * 0.15
        return round(min(1.0, overlap * 0.85 + progress_bonus), 3)
