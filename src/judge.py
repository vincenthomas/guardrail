"""LLM-as-judge for the one field string equality can't score.

The judge sees the email, the gold requested_action, and the candidate's
requested_action, and answers MATCH or NO_MATCH with a one-line reason.
Judged pairwise against gold, not open-ended quality — narrow judges agree
with humans more than "rate 1-10" judges.

Idempotent like run.py. Usage: python -m src.judge A
Writes outputs/judge_<config>.jsonl.

Judge-vs-human agreement: `python -m src.judge --sample 30` emits
outputs/agreement_sample.csv with judge verdicts and a blank human column;
fill it, then `python -m src.judge --agreement` reports the %.
"""
import argparse
import json
import time
from pathlib import Path

import yaml
from anthropic import Anthropic

from .run import load_jsonl
from .checks import parse_output

JUDGE_PROMPT = """A support system extracted the customer's requested action from an email.
Decide if the extraction matches the reference. MATCH means a support agent
acting on the extraction would do what the reference describes, including any
deadline. Ignore wording differences.

Email: {email}
Reference action: {gold}
Extracted action: {candidate}

Reply with exactly one line: MATCH or NO_MATCH, then " — " and a reason under 15 words."""


def judge_config(name: str):
    cfg = yaml.safe_load(Path("configs/configs.yaml").read_text())["judge"]
    cases = {c["id"]: c for c in load_jsonl("golden/cases.jsonl")}
    outputs = load_jsonl(f"outputs/{name}.jsonl")
    out_path = Path("outputs") / f"judge_{name}.jsonl"
    done = {r["id"] for r in load_jsonl(out_path)} if out_path.exists() else set()

    client = Anthropic()
    with out_path.open("a") as f:
        for row in outputs:
            if row["id"] in done:
                continue
            parsed = parse_output(row["raw"])
            candidate = (parsed or {}).get("requested_action")
            if not candidate:
                f.write(json.dumps({"id": row["id"], "match": 0,
                                    "reason": "no requested_action extracted",
                                    "input_tokens": 0, "output_tokens": 0}) + "\n")
                continue
            case = cases[row["id"]]
            t0 = time.perf_counter()
            resp = client.messages.create(
                model=cfg["model"], max_tokens=60, temperature=cfg["temperature"],
                messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                    email=case["email"], gold=case["gold"]["requested_action"],
                    candidate=candidate)}],
            )
            text = resp.content[0].text.strip()
            f.write(json.dumps({
                "id": row["id"],
                "match": int(text.startswith("MATCH")),
                "reason": text,
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
            }) + "\n")
            f.flush()
            print(f"judge {name} {row['id']} {time.perf_counter()-t0:.2f}s {text[:60]}")


def make_agreement_sample(n: int = 30, seed: int = 7):
    """Sample judged cases across configs into a CSV for independent human labeling."""
    import random
    rows = []
    for cfg in ("A", "B"):
        cases = {c["id"]: c for c in load_jsonl("golden/cases.jsonl")}
        outs = {r["id"]: r for r in load_jsonl(f"outputs/{cfg}.jsonl")}
        for j in load_jsonl(f"outputs/judge_{cfg}.jsonl"):
            parsed = parse_output(outs[j["id"]]["raw"]) or {}
            rows.append({
                "config": cfg, "id": j["id"],
                "email": cases[j["id"]]["email"],
                "gold_action": cases[j["id"]]["gold"]["requested_action"],
                "extracted_action": parsed.get("requested_action"),
                "judge_match": j["match"], "human_match": "",
            })
    random.Random(seed).shuffle(rows)
    import csv
    with open("outputs/agreement_sample.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows[:n])
    print(f"wrote outputs/agreement_sample.csv ({min(n, len(rows))} rows) — fill human_match with 1/0")


def report_agreement():
    import csv
    with open("outputs/agreement_sample.csv") as f:
        rows = [r for r in csv.DictReader(f) if r["human_match"] != ""]
    if not rows:
        print("no human labels yet")
        return
    agree = sum(int(r["judge_match"]) == int(r["human_match"]) for r in rows)
    print(f"judge-human agreement: {agree}/{len(rows)} = {agree/len(rows):.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--agreement", action="store_true")
    args = ap.parse_args()
    if args.sample:
        make_agreement_sample(args.sample)
    elif args.agreement:
        report_agreement()
    else:
        judge_config(args.config)
