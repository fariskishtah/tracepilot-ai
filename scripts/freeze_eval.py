"""Freeze the evaluation protocol BEFORE any final-test run.
Records SHA-256 of: test questions, corpus manifest, configs, all source files, plus model ids and a timestamp, and verifies that no
`test_*` result file exists yet. After this point the agent code/config/questions must not change; if they do, later runs are POST-HOC."""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def h(paths):
    d = hashlib.sha256()
    for p in sorted(paths):
        d.update(str(p.relative_to(ROOT)).encode())
        d.update(p.read_bytes())
    return d.hexdigest()


def main():
    existing = [p.name for p in (ROOT / "artifacts" / "metrics").glob("test_*.jsonl")]
    if existing:
        sys.exit(f"REFUSING: test result files already exist ({existing}); a freeze must precede every test run")
    qs = (ROOT / "data/evaluation/questions.jsonl").read_text().splitlines()
    test_q = [l for l in qs if json.loads(l)["split"] == "test"]
    rec = {
        "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_questions": len(test_q), "test_questions_sha256": hashlib.sha256("\n".join(test_q).encode()).hexdigest(),
        "all_questions_sha256": hashlib.sha256("\n".join(qs).encode()).hexdigest(),
        "corpus_manifest_sha256": hashlib.sha256((ROOT / "data/manifests/sources.json").read_bytes()).hexdigest(),
        "configs_sha256": h(list((ROOT / "configs").glob("*.yaml"))),
        "source_sha256": h(list((ROOT / "src").rglob("*.py"))),
        "llm": "qwen2.5:3b-instruct", "embedding": "sentence-transformers/all-MiniLM-L6-v2", "registry_version": "1.0.0",
        "policy": "No agent code, config, rubric or test question may change after this record. Runs made after a change are labelled post-hoc.",
    }
    try:
        subprocess.run(["git", "add", "-A"], cwd=ROOT, check=True, capture_output=True)
        subprocess.run(["git", "-c", "user.name=Faris Mohamed Kishtah", "-c", "user.email=fariskishtah@users.noreply.github.com", "commit", "-qm", "Freeze evaluation protocol (before test runs)"], cwd=ROOT, capture_output=True)
        rec["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception as e:
        rec["git_commit"] = f"n/a ({type(e).__name__})"
    (ROOT / "data/evaluation/FROZEN.json").write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
