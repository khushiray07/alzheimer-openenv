"""Grader 1: Score for Task 1 — Risk Classification."""

import random


class Grader1:
    @staticmethod
    def grade(patient: dict, parsed_action: dict) -> float:
        if not parsed_action.get("valid"):
            return 0.05

        true_label = "AD" if patient["label"] == 1 else "Control"
        predicted = parsed_action.get("prediction", "")

        if predicted == true_label:
            risk = patient["risk_score"]
            if 40 < risk < 65:  # ambiguous zone — harder case
                return round(random.uniform(0.78, 0.92), 3)
            return round(random.uniform(0.85, 1.0), 3)
        else:
            return round(random.uniform(0.0, 0.12), 3)
