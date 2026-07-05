"""Run a config over the golden set. Idempotent: already-run case ids are
skipped, so interrupted runs resume instead of double-spending tokens.

Usage: python -m src.run A [--cases golden/cases.jsonl]
Writes outputs/<config>.jsonl: one row per case with raw output, latency, usage.
"""
import argparse
import json
import time
from pathlib import Path

import yaml
from anthropic import Anthropic


def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--cases", default="golden/cases.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path("configs/configs.yaml").read_text())["configs"][args.config]
    prompt_tpl = Path(cfg["prompt"]).read_text()
    cases = load_jsonl(args.cases)

    out_path = Path("outputs") / f"{args.config}.jsonl"
    out_path.parent.mkdir(exist_ok=True)
    done = {r["id"] for r in load_jsonl(out_path)} if out_path.exists() else set()

    client = Anthropic()
    n = 0
    with out_path.open("a") as f:
        for case in cases:
            if case["id"] in done:
                continue
            if args.limit and n >= args.limit:
                break
            t0 = time.perf_counter()
            resp = client.messages.create(
                model=cfg["model"],
                max_tokens=cfg["max_tokens"],
                temperature=cfg["temperature"],
                messages=[{"role": "user",
                           "content": prompt_tpl.replace("{email}", case["email"])}],
            )
            latency = time.perf_counter() - t0
            f.write(json.dumps({
                "id": case["id"],
                "raw": resp.content[0].text,
                "latency_s": round(latency, 3),
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
            }) + "\n")
            f.flush()
            n += 1
            print(f"{args.config} {case['id']} {latency:.2f}s")


if __name__ == "__main__":
    main()
