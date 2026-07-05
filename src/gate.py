"""The experimentation layer: compare two configs like a launch review.

A candidate ships only if the primary metric does not regress AND every
guardrail stays inside its threshold — the multi-metric launch criteria
pattern from growth experimentation, applied to prompt changes.

Usage: python -m src.gate A B
Writes results/verdict.md and prints it.
"""
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from .checks import check_case
from .run import load_jsonl


def config_metrics(name: str, pricing: dict, model: str):
    cases = {c["id"]: c for c in load_jsonl("golden/cases.jsonl")}
    outputs = load_jsonl(f"outputs/{name}.jsonl")
    judge = {r["id"]: r["match"] for r in load_jsonl(f"outputs/judge_{name}.jsonl")}

    checks = [check_case(r["raw"], cases[r["id"]]["gold"]) for r in outputs]
    lat = [r["latency_s"] for r in outputs]
    in_tok = sum(r["input_tokens"] for r in outputs)
    out_tok = sum(r["output_tokens"] for r in outputs)
    price = pricing[model]
    cost = (in_tok * price["input"] + out_tok * price["output"]) / 1e6

    field_scores = []
    for r, c in zip(outputs, checks):
        action = judge.get(r["id"], 0)
        field_scores.append(np.mean([c["intent_match"], c["urgency_match"],
                                     c["product_match"], action]))
    return {
        "n": len(outputs),
        "field_accuracy": float(np.mean(field_scores)),
        "intent_acc": float(np.mean([c["intent_match"] for c in checks])),
        "urgency_acc": float(np.mean([c["urgency_match"] for c in checks])),
        "product_acc": float(np.mean([c["product_match"] for c in checks])),
        "action_match": float(np.mean([judge.get(r["id"], 0) for r in outputs])),
        "format_failure_rate": float(np.mean([not c["format_ok"] for c in checks])),
        "latency_p50_ms": float(np.percentile(lat, 50) * 1000),
        "latency_p95_ms": float(np.percentile(lat, 95) * 1000),
        "cost_per_1k_usd": cost / len(outputs) * 1000,
    }


def main(control: str, candidate: str):
    cfg = yaml.safe_load(Path("configs/configs.yaml").read_text())
    gate = cfg["gate"]
    a = config_metrics(control, cfg["pricing"], cfg["configs"][control]["model"])
    b = config_metrics(candidate, cfg["pricing"], cfg["configs"][candidate]["model"])

    primary = gate["primary"]
    lift = b[primary] - a[primary]
    breaches = []
    for metric, rule in gate["guardrails"].items():
        if "max_abs_increase" in rule and b[metric] - a[metric] > rule["max_abs_increase"]:
            breaches.append(f"{metric}: +{b[metric]-a[metric]:.3f} > {rule['max_abs_increase']}")
        if "max_pct_increase" in rule and a[metric] > 0 and \
                (b[metric] - a[metric]) / a[metric] * 100 > rule["max_pct_increase"]:
            breaches.append(f"{metric}: +{(b[metric]-a[metric])/a[metric]*100:.0f}% > {rule['max_pct_increase']}%")

    ship = lift >= gate["min_primary_lift"] and not breaches
    lines = [f"# Verdict: {candidate} vs {control} — {'SHIP' if ship else 'BLOCK'}", ""]
    lines.append(f"| metric | {control} | {candidate} | delta |")
    lines.append("|---|---|---|---|")
    for m in ["field_accuracy", "intent_acc", "urgency_acc", "product_acc", "action_match",
              "format_failure_rate", "latency_p50_ms", "latency_p95_ms", "cost_per_1k_usd"]:
        lines.append(f"| {m} | {a[m]:.3f} | {b[m]:.3f} | {b[m]-a[m]:+.3f} |")
    lines.append("")
    lines.append(f"Primary ({primary}): {lift:+.3f} (required ≥ {gate['min_primary_lift']})")
    lines.append("Guardrail breaches: " + ("; ".join(breaches) if breaches else "none"))
    lines.append(f"n = {a['n']} golden cases (v1); costs use configs.yaml unit prices")
    report = "\n".join(lines)
    Path("results").mkdir(exist_ok=True)
    Path("results/verdict.md").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
