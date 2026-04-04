"""
AlzheimerEnv — local submission validator.

Checks all components before submission.
"""

import sys
import os
import traceback

sys.path.insert(0, os.path.dirname(__file__))

PASS = "[PASS]"
FAIL = "[FAIL]"
results = []


def check(name: str, fn):
    try:
        fn()
        print(f"  {PASS} {name}")
        results.append((name, True, None))
    except Exception as exc:
        msg = str(exc)
        print(f"  {FAIL} {name}: {msg}")
        if os.environ.get("VERBOSE"):
            traceback.print_exc()
        results.append((name, False, msg))


# ---------------------------------------------------------------------------
# 1. Import AlzheimerEnv
# ---------------------------------------------------------------------------
print("\n── Imports ──")

env_module = None


def _import_env():
    global env_module
    from environment import AlzheimerEnv
    env_module = AlzheimerEnv


check("Import AlzheimerEnv", _import_env)

# ---------------------------------------------------------------------------
# 2. reset()
# ---------------------------------------------------------------------------
print("\n── reset() ──")

env_instance = None


def _reset_task1():
    global env_instance
    from environment import AlzheimerEnv
    env_instance = AlzheimerEnv()
    state = env_instance.reset(task_id=1)
    assert "patient_id" in state, "Missing patient_id"
    assert "risk_score" in state, "Missing risk_score"
    assert "gene_expression" in state, "Missing gene_expression"
    assert "step" in state, "Missing step"
    assert "max_steps" in state, "Missing max_steps"
    assert state["step"] == 0, f"Expected step=0, got {state['step']}"


check("reset(task_id=1) returns valid state", _reset_task1)


def _reset_task2():
    env_instance.reset(task_id=2)


check("reset(task_id=2) works", _reset_task2)


def _reset_task3():
    env_instance.reset(task_id=3)


check("reset(task_id=3) works", _reset_task3)


def _reset_specific_patient():
    state = env_instance.reset(task_id=1, patient_id="PT-001")
    assert state["patient_id"] == "PT-001", f"Wrong patient: {state['patient_id']}"


check("reset(patient_id='PT-001') selects correct patient", _reset_specific_patient)

# ---------------------------------------------------------------------------
# 3. step()
# ---------------------------------------------------------------------------
print("\n── step() ──")


def _step_task1_valid():
    env_instance.reset(task_id=1, patient_id="PT-001")
    result = env_instance.step("classify:AD")
    assert "observation" in result, "Missing observation"
    assert "reward" in result, "Missing reward"
    assert "done" in result, "Missing done"
    assert "info" in result, "Missing info"
    r = result["reward"]
    assert 0.0 <= r <= 1.0, f"Reward out of range: {r}"


check("step('classify:AD') returns valid result", _step_task1_valid)


def _step_task1_control():
    env_instance.reset(task_id=1, patient_id="PT-002")
    result = env_instance.step("classify:Control")
    r = result["reward"]
    assert 0.0 <= r <= 1.0, f"Reward out of range: {r}"


check("step('classify:Control') reward in [0,1]", _step_task1_control)


def _step_task2_rank():
    env_instance.reset(task_id=2, patient_id="PT-003")
    result = env_instance.step("rank:[APOE,APP,PSEN1]")
    r = result["reward"]
    assert 0.0 <= r <= 1.0, f"Reward out of range: {r}"


check("step('rank:[APOE,APP,PSEN1]') reward in [0,1]", _step_task2_rank)


def _step_task3_downregulate():
    env_instance.reset(task_id=3, patient_id="PT-005")
    result = env_instance.step("downregulate:APOE")
    r = result["reward"]
    assert 0.0 <= r <= 1.0, f"Reward out of range: {r}"
    info = result["info"]
    assert "new_risk" in info, "Missing new_risk in info"
    assert info["new_risk"] < info["old_risk"], "Risk should decrease"


check("step('downregulate:APOE') reduces risk", _step_task3_downregulate)


def _step_task3_upregulate():
    env_instance.reset(task_id=3, patient_id="PT-001")
    result = env_instance.step("upregulate:TREM2")
    r = result["reward"]
    assert 0.0 <= r <= 1.0, f"Reward out of range: {r}"


check("step('upregulate:TREM2') reward in [0,1]", _step_task3_upregulate)


def _step_invalid_action():
    env_instance.reset(task_id=1, patient_id="PT-001")
    result = env_instance.step("totally_invalid")
    r = result["reward"]
    assert 0.0 <= r <= 1.0, f"Reward out of range on invalid action: {r}"


check("Invalid action returns reward in [0,1]", _step_invalid_action)

# ---------------------------------------------------------------------------
# 4. state()
# ---------------------------------------------------------------------------
print("\n── state() ──")


def _state_check():
    env_instance.reset(task_id=1, patient_id="PT-001")
    state = env_instance.state()
    for key in ("task_id", "patient_id", "risk_score", "gene_expression", "step", "max_steps"):
        assert key in state, f"Missing key: {key}"


check("state() returns all required fields", _state_check)

# ---------------------------------------------------------------------------
# 5. list_tasks()
# ---------------------------------------------------------------------------
print("\n── list_tasks() ──")


def _list_tasks_check():
    tasks = env_instance.list_tasks()
    assert len(tasks) >= 3, f"Expected 3+ tasks, got {len(tasks)}"
    for t in tasks:
        for key in ("id", "name", "difficulty", "max_steps"):
            assert key in t, f"Missing key '{key}' in task {t}"


check("list_tasks() returns 3+ tasks with required fields", _list_tasks_check)

# ---------------------------------------------------------------------------
# 6. Reward bounds — full episode per task
# ---------------------------------------------------------------------------
print("\n── Reward bounds (full episodes) ──")


def _rewards_task1():
    from environment import AlzheimerEnv
    e = AlzheimerEnv()
    for pid, action in [("PT-001", "classify:AD"), ("PT-002", "classify:Control"),
                        ("PT-003", "classify:AD"), ("PT-004", "classify:Control"),
                        ("PT-005", "classify:AD")]:
        e.reset(task_id=1, patient_id=pid)
        result = e.step(action)
        r = result["reward"]
        assert 0.0 <= r <= 1.0, f"Reward {r} out of [0,1] for {pid}"


check("All Task 1 rewards in [0.0, 1.0]", _rewards_task1)


def _rewards_task2():
    from environment import AlzheimerEnv
    e = AlzheimerEnv()
    for pid in ["PT-001", "PT-003", "PT-005"]:
        e.reset(task_id=2, patient_id=pid)
        result = e.step("rank:[APOE,APP,PSEN1]")
        r = result["reward"]
        assert 0.0 <= r <= 1.0, f"Reward {r} out of [0,1] for {pid}"


check("All Task 2 rewards in [0.0, 1.0]", _rewards_task2)


def _rewards_task3():
    from environment import AlzheimerEnv
    e = AlzheimerEnv()
    for pid in ["PT-001", "PT-003", "PT-005"]:
        e.reset(task_id=3, patient_id=pid)
        result = e.step("downregulate:APOE")
        r = result["reward"]
        assert 0.0 <= r <= 1.0, f"Reward {r} out of [0,1] for {pid}"


check("All Task 3 rewards in [0.0, 1.0]", _rewards_task3)

# ---------------------------------------------------------------------------
# 7. openenv.yaml validation
# ---------------------------------------------------------------------------
print("\n── openenv.yaml ──")


def _yaml_valid():
    import yaml
    yaml_path = os.path.join(os.path.dirname(__file__), "openenv.yaml")
    assert os.path.exists(yaml_path), "openenv.yaml not found"
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    required_fields = ["name", "version", "description", "tasks", "observation_space", "action_space", "reward_range"]
    for field in required_fields:
        assert field in data, f"Missing field: {field}"
    assert len(data["tasks"]) >= 3, f"Expected 3+ tasks, got {len(data['tasks'])}"
    assert data["reward_range"] == [0.0, 1.0], f"Wrong reward_range: {data['reward_range']}"


check("openenv.yaml is valid YAML with required fields", _yaml_valid)

# ---------------------------------------------------------------------------
# 8. File existence checks
# ---------------------------------------------------------------------------
print("\n── File existence ──")

BASE = os.path.dirname(__file__)

for fname in [
    "inference.py", "Dockerfile", "server.py", "environment.py",
    "tasks/task1_classify.py", "tasks/task2_biomarker.py", "tasks/task3_intervene.py",
    "graders/grader1.py", "graders/grader2.py", "graders/grader3.py",
    "requirements.txt", "openenv.yaml",
]:
    path = fname  # relative
    fpath = os.path.join(BASE, path)

    def _file_check(p=fpath, n=fname):
        assert os.path.exists(p), f"{n} not found at {p}"

    check(f"{fname} exists", _file_check)

# ---------------------------------------------------------------------------
# Final verdict
# ---------------------------------------------------------------------------
print()
passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)
total = len(results)

print(f"Results: {passed}/{total} checks passed")
print()

if failed == 0:
    print("✓ READY TO SUBMIT")
    sys.exit(0)
else:
    print(f"✗ FIX {failed} ISSUE(S) BEFORE SUBMITTING")
    for name, ok, err in results:
        if not ok:
            print(f"  - {name}: {err}")
    sys.exit(1)
