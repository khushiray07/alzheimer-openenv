"""
Grader 2: Score for Task 2 — Biomarker Ranking.

Deterministic scoring based on overlap with gold standard ranking.
Rewards partial credit: top-1, top-2, top-3 matches each contribute.
"""

GOLD = ["APOE", "APP", "PSEN1", "MAPT", "BACE1"]


def _clamp(score: float) -> float:
    """Ensure score is strictly between 0 and 1 (exclusive)."""
    return float(round(max(0.01, min(0.99, score)), 4))


class Grader2:
    @staticmethod
    def grade(parsed_action: dict, step: int, max_steps: int) -> float:
        if not parsed_action.get("valid"):
            return _clamp(0.05)

        ranking = parsed_action.get("ranking", [])
        if not ranking:
            return _clamp(0.05)

        positional_score = 0.0
        weights = [0.40, 0.25, 0.20]
        for i, weight in enumerate(weights):
            if i < len(ranking) and i < len(GOLD) and ranking[i] == GOLD[i]:
                positional_score += weight

        k = min(3, len(ranking))
        overlap = len(set(ranking[:k]) & set(GOLD[:k])) / k if k > 0 else 0
        overlap_score = overlap * 0.60

        content_score = max(positional_score, overlap_score)
        efficiency_bonus = max(0, (1 - step / max_steps)) * 0.15

        return _clamp(content_score + efficiency_bonus)
