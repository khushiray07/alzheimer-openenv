import os
import sys
import json

sys.path.insert(0, os.path.dirname(__file__))

from openai import OpenAI
from environment import AlzheimerEnv

# Environment variables with defaults
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

TASKS = [
    {
        "task_id": 1,
        "task_name": "risk_classification",
        "system": "You are an AI agent in AlzheimerEnv. Classify the patient as AD or Control based on gene expression. Reply ONLY with JSON: {\"action\": \"classify:AD\", \"reasoning\": \"brief reason\"}",
        "fallback": "classify:AD"
    },
    {
        "task_id": 2,
        "task_name": "biomarker_ranking",
        "system": "You are an AI agent in AlzheimerEnv. Rank the top 3 Alzheimer's biomarker genes. Reply ONLY with JSON: {\"action\": \"rank:[APOE,APP,PSEN1]\", \"reasoning\": \"brief reason\"}",
        "fallback": "rank:[APOE,APP,PSEN1]"
    },
    {
        "task_id": 3,
        "task_name": "intervention_planning",
        "system": "You are an AI agent in AlzheimerEnv. Propose a gene intervention to reduce Alzheimer's risk below 40. Reply ONLY with JSON: {\"action\": \"downregulate:APOE\", \"reasoning\": \"brief reason\"}",
        "fallback": "downregulate:APOE"
    }
]


def get_action(system_prompt, state, fallback):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Current state: {json.dumps(state)}"}
            ],
            max_tokens=200
        )
        text = response.choices[0].message.content.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(text)
        return parsed.get("action", fallback), None
    except Exception as e:
        return fallback, str(e)


def run_episode(task_config):
    env = AlzheimerEnv()
    task_id = task_config["task_id"]
    task_name = task_config["task_name"]

    obs = env.reset(task_id=task_id)

    print(f"[START] task={task_name} env=AlzheimerEnv model={MODEL_NAME}", flush=True)

    rewards = []
    steps = 0
    success = False
    last_error = None

    try:
        while True:
            state = env.state()
            action, error = get_action(
                task_config["system"],
                state,
                task_config["fallback"]
            )

            result = env.step(action)
            reward = float(result["reward"])
            done = bool(result["done"])
            steps += 1
            rewards.append(reward)

            error_str = error if error else "null"
            done_str = "true" if done else "false"

            print(f"[STEP] step={steps} action={action} reward={reward:.2f} done={done_str} error={error_str}", flush=True)

            if done:
                success = reward > 0.5
                break

    except Exception as e:
        last_error = str(e)

    rewards_str = ",".join([f"{r:.2f}" for r in rewards])
    success_str = "true" if success else "false"
    print(f"[END] success={success_str} steps={steps} rewards={rewards_str}", flush=True)


if __name__ == "__main__":
    for task in TASKS:
        run_episode(task)
        print("", flush=True)
