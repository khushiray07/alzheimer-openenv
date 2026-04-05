"""
AlzheimerEnv inference script.

Runs 3 full task episodes using an LLM agent (via OpenAI-compatible client),
printing [START] / [STEP] / [END] logs to stdout.
"""

import sys
import os
import json
import time

# Ensure local modules are importable
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config from environment variables
# ---------------------------------------------------------------------------
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.anthropic.com")
MODEL_NAME = os.environ.get("MODEL_NAME", "claude-haiku-4-5-20251001")
HF_TOKEN = os.environ.get("HF_TOKEN")          # No default — must be set externally
LOCAL_IMAGE_NAME = os.environ.get("LOCAL_IMAGE_NAME")  # Optional, for from_docker_image()

# Use HF_TOKEN as the API key (passed to OpenAI client)
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY") or HF_TOKEN

# ---------------------------------------------------------------------------
# Task configuration
# ---------------------------------------------------------------------------
TASK_CONFIGS = {
    1: {
        "system_prompt": (
            "You are an AI agent in AlzheimerEnv. Classify the patient as AD or Control based on gene expression. "
            'Reply ONLY with JSON: {"action": "classify:AD", "reasoning": "brief reason under 80 chars"}'
        ),
        "fallback_action": "classify:AD",
        "patient_id": "PT-001",
    },
    2: {
        "system_prompt": (
            "You are an AI agent in AlzheimerEnv. Rank the top 3 Alzheimer's biomarker genes from the patient data. "
            'Reply ONLY with JSON: {"action": "rank:[GENE1,GENE2,GENE3]", "reasoning": "brief reason under 80 chars"}'
        ),
        "fallback_action": "rank:[APOE,APP,PSEN1]",
        "patient_id": "PT-003",
    },
    3: {
        "system_prompt": (
            "You are an AI agent in AlzheimerEnv. Propose a gene intervention to reduce Alzheimer's risk below 40. "
            'Reply ONLY with JSON: {"action": "downregulate:GENE or upregulate:GENE", "reasoning": "brief reason under 80 chars"}'
        ),
        "fallback_action": "downregulate:APOE",
        "patient_id": "PT-005",
    },
}


# ---------------------------------------------------------------------------
# LLM client helper
# ---------------------------------------------------------------------------

def get_llm_client():
    """Return an OpenAI-compatible client pointed at API_BASE_URL."""
    try:
        from openai import OpenAI
        api_key = ANTHROPIC_API_KEY if ANTHROPIC_API_KEY else "dummy-key"
        # Anthropic's API base for OpenAI-compat is /v1
        base_url = API_BASE_URL.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url = base_url + "/v1"
        client = OpenAI(api_key=api_key, base_url=base_url)
        return client
    except Exception:
        return None


def call_llm(client, system_prompt: str, user_message: str, fallback_action: str) -> tuple[str, str]:
    """
    Call the LLM and return (action, reasoning).
    Falls back to fallback_action on any error.
    """
    if client is None or not ANTHROPIC_API_KEY:
        return fallback_action, "Fallback: no API key configured"

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=128,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            extra_headers={"anthropic-version": "2023-06-01"},
        )
        raw = response.choices[0].message.content.strip()
        # Try to parse JSON
        try:
            data = json.loads(raw)
            action = str(data.get("action", fallback_action)).strip()
            reasoning = str(data.get("reasoning", ""))[:80]
        except json.JSONDecodeError:
            # Attempt to extract action from raw text
            action = fallback_action
            reasoning = raw[:80]
        return action, reasoning
    except Exception as exc:
        return fallback_action, f"Fallback: {str(exc)[:60]}"


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(env, task_id: int, client) -> dict:
    """Run a full episode for a given task and return summary stats."""
    cfg = TASK_CONFIGS[task_id]
    system_prompt = cfg["system_prompt"]
    fallback_action = cfg["fallback_action"]
    patient_id = cfg["patient_id"]

    # Reset
    obs = env.reset(task_id=task_id, patient_id=patient_id)

    initial_risk = obs.get("risk_score", 0.0)
    max_steps = obs.get("max_steps", 3)

    print(
        f'[START] task_id={task_id} patient="{obs["patient_id"]}" '
        f"initial_risk={initial_risk} max_steps={max_steps}",
        flush=True,
    )

    total_reward = 0.0
    step_num = 0
    done = False

    while not done:
        # Build user message from current observation
        user_message = json.dumps(obs, indent=2)

        action, reasoning = call_llm(client, system_prompt, user_message, fallback_action)

        result = env.step(action)
        reward = result["reward"]
        done = result["done"]
        obs = result.get("observation", {})
        total_reward += reward
        step_num += 1

        # Truncate reasoning for display
        reasoning_display = reasoning[:80].replace('"', "'")
        print(
            f'[STEP] step={step_num} action="{action}" '
            f'reasoning="{reasoning_display}" reward={reward:.3f}',
            flush=True,
        )

    avg_reward = total_reward / max(step_num, 1)
    score = round(avg_reward * 100, 1)
    status = "SUCCESS" if score >= 50.0 else "NEEDS_IMPROVEMENT"

    print(
        f"[END] task_id={task_id} total_reward={total_reward:.3f} "
        f"avg_reward={avg_reward:.3f} score={score} status={status}",
        flush=True,
    )

    return {"task_id": task_id, "total_reward": total_reward, "avg_reward": avg_reward, "score": score}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from environment import AlzheimerEnv

    env = AlzheimerEnv()
    client = get_llm_client()

    results = []
    for task_id in [1, 2, 3]:
        try:
            summary = run_episode(env, task_id, client)
            results.append(summary)
        except Exception as exc:
            print(f"[ERROR] task_id={task_id} error={exc}", flush=True)
            results.append({"task_id": task_id, "score": 0.0})

    # Summary
    total_score = sum(r.get("score", 0.0) for r in results)
    avg_score = total_score / len(results)
    status = "PASS" if avg_score >= 70 else "NEEDS_IMPROVEMENT"

    print("", flush=True)
    print("=== INFERENCE COMPLETE ===", flush=True)
    print(f"Tasks run: {len(results)}", flush=True)
    print(f"Total score: {total_score:.1f} / 300", flush=True)
    print(f"Status: {status}", flush=True)


if __name__ == "__main__":
    main()
