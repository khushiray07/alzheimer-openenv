"""Task 1: Risk Classification — classify patient as AD or Control."""


class Task1RiskClassification:
    task_id = 1
    name = "risk_classification"
    difficulty = "easy"
    max_steps = 3

    @staticmethod
    def get_observation(patient: dict, step: int, max_steps: int) -> dict:
        return {
            "patient_id": patient["id"],
            "risk_score": patient["risk_score"],
            "age": patient["age"],
            "stage": patient["stage"],
            "gene_expression": dict(patient["genes"]),
            "step": step,
            "max_steps": max_steps,
            "task": "risk_classification",
            "instructions": (
                "Classify this patient as 'AD' (Alzheimer's Disease) or 'Control' "
                "based on their gene expression profile and risk score. "
                "Action format: classify:AD  or  classify:Control"
            ),
        }

    @staticmethod
    def parse_action(action_str: str) -> dict:
        action_str = action_str.strip()
        if action_str.startswith("classify:"):
            prediction = action_str[len("classify:"):].strip()
            if prediction in ("AD", "Control"):
                return {"prediction": prediction, "valid": True}
        return {"prediction": "", "valid": False, "error": f"Invalid action: '{action_str}'"}

    @staticmethod
    def compute_info(patient: dict, parsed_action: dict) -> dict:
        true_label = "AD" if patient["label"] == 1 else "Control"
        predicted = parsed_action.get("prediction", "")
        correct = predicted == true_label
        return {
            "correct": correct,
            "true_label": true_label,
            "predicted": predicted,
        }
