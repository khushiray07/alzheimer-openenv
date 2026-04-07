"""
Grader 2: Score for Task 2 — Biomarker Ranking.

Deterministic scoring based on overlap with gold standard ranking.
Rewards partial credit: top-1, top-2, top-3 matches each contribute.
"""

GOLD = ["APOE", "APP", "PSEN1", "MAPT", "BACE1"]


class Grader2:
    @staticmethod
    def grade(parsed_action: dict, step: int, max_steps: int) -> float:
        if not parsed_action.get("valid"):
            return 0.05

        ranking = parsed_action.get("ranking", [])
        if not ranking:
            return 0.05

        # Positional scoring: exact position match scores more than set overlap
        # Top-1 match = 0.40, top-2 match = 0.25, top-3 match = 0.20 → max 0.85
        positional_score = 0.0
        weights = [0.40, 0.25, 0.20]
        for i, weight in enumerate(weights):
            if i < len(ranking) and i < len(GOLD) and ranking[i] == GOLD[i]:
                positional_score += weight

        # Set overlap bonus for top-3 (partial credit for right genes, wrong order)
        k = min(3, len(ranking))
        overlap = len(set(ranking[:k]) & set(GOLD[:k])) / k if k > 0 else 0
        overlap_score = overlap * 0.60  # max 0.60

        # Take best of positional or overlap (rewards two different strategies)
        content_score = max(positional_score, overlap_score)

        # Step efficiency bonus: earlier correct answer = more reward
        efficiency_bonus = max(0, (1 - step / max_steps)) * 0.15

        return round(min(0.99, content_score + efficiency_bonus), 3)
