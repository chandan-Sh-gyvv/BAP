"""
BAP — True CoT Baseline Benchmark
===================================
For each query in the 3-domain test suite:
  1. Runs BAP to get actual step count (bap_steps)
  2. Calls Groq API with a standard CoT prompt and reads
     usage.completion_tokens from the response
  3. Computes true TRR = bap_steps / cot_completion_tokens

Requires GROQ_API_KEY in environment.
Results saved to benchmark_cot_results.json.

Usage:
    python benchmark_cot_baseline.py
"""

import os
import json
import time
import urllib.request
import urllib.error

import bap_engine

# ── Groq settings ──────────────────────────────────────────────────────────────

GROQ_MODEL  = "llama-3.1-8b-instant"
GROQ_URL    = "https://api.groq.com/openai/v1/chat/completions"
COT_PROMPT  = "Think step by step and answer: {query}"
INTER_QUERY_SLEEP = 1.5   # seconds between Groq calls — stay under rate limit

# ── Same queries as test_all_domains.py ───────────────────────────────────────

FIBONACCI_CASES = [
    {"query": "What is F(7)?",
     "expected_answer_contains": "13"},
    {"query": "What is the next number after 8 in the Fibonacci sequence?",
     "expected_answer_contains": "13"},
    {"query": "What is the recurrence relation for Fibonacci numbers?",
     "expected_answer_contains": "f(n-1) + f(n-2)"},
    {"query": "What is Binet's formula?",
     "expected_answer_contains": "sqrt(5)"},
    {"query": "What is the golden ratio phi?",
     "expected_answer_contains": "phi"},
    {"query": "How does the Fibonacci sequence relate to Pascal's triangle?",
     "expected_answer_contains": "pascal"},
    {"query": "What are Lucas numbers?",
     "expected_answer_contains": "lucas"},
    {"query": "What is the Pisano period?",
     "expected_answer_contains": "pisano"},
    {"query": "Are consecutive Fibonacci numbers coprime?",
     "expected_answer_contains": "coprime"},
    {"query": "What is the Zeckendorf representation?",
     "expected_answer_contains": "zeckendorf"},
]

ARITHMETIC_CASES = [
    {"query": "What is 17 mod 5?",
     "expected_answer_contains": "remainder"},
    {"query": "How do you compute the mean of a dataset?",
     "expected_answer_contains": "mean"},
    {"query": "What is the quadratic formula?",
     "expected_answer_contains": "sqrt"},
    {"query": "What is the Pythagorean theorem?",
     "expected_answer_contains": "right"},
    {"query": "How do you add fractions with different denominators?",
     "expected_answer_contains": "denominator"},
    {"query": "What are the rules for exponents?",
     "expected_answer_contains": "exponent"},
    {"query": "What is a prime number?",
     "expected_answer_contains": "prime"},
    {"query": "How do logarithms work?",
     "expected_answer_contains": "log"},
    {"query": "What is the sum of the first n natural numbers?",
     "expected_answer_contains": "n * (n + 1)"},
    {"query": "What is standard deviation?",
     "expected_answer_contains": "variance"},
]

MULTIHOP_CASES = [
    {"query": "What scientific field did Marie Curie work in?",
     "expected_answer_contains": "physics"},
    {"query": "Who invented the telephone?",
     "expected_answer_contains": "bell"},
    {"query": "What is the capital of France?",
     "expected_answer_contains": "paris"},
    {"query": "When did World War II end?",
     "expected_answer_contains": "1945"},
    {"query": "Who discovered penicillin?",
     "expected_answer_contains": "fleming"},
    {"query": "What did Einstein receive the Nobel Prize for?",
     "expected_answer_contains": "photoelectric"},
    {"query": "Who wrote Hamlet?",
     "expected_answer_contains": "shakespeare"},
    {"query": "When was the first FIFA World Cup held?",
     "expected_answer_contains": "1930"},
    {"query": "What is the tallest mountain in the world?",
     "expected_answer_contains": "everest"},
    {"query": "Can two contradictory facts about the same subject both be true?",
     "expected_answer_contains": "cannot"},
]

DOMAINS = [
    ("fibonacci_reasoning", FIBONACCI_CASES),
    ("arithmetic",          ARITHMETIC_CASES),
    ("multihop_facts",      MULTIHOP_CASES),
]

# ── Groq CoT call ──────────────────────────────────────────────────────────────

def _call_groq_cot(query: str, api_key: str) -> dict:
    """
    Call Groq with CoT prompt.  Returns dict:
        content           : full response text
        completion_tokens : actual output tokens (from usage.completion_tokens)
    Raises on HTTP error.
    """
    payload = json.dumps({
        "model":       GROQ_MODEL,
        "messages":    [{"role": "user",
                         "content": COT_PROMPT.format(query=query)}],
        "temperature": 0.3,
        "max_tokens":  512,
    }).encode()

    req = urllib.request.Request(
        GROQ_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent":    "bap-cot-baseline/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())

    return {
        "content":           data["choices"][0]["message"]["content"].strip(),
        "completion_tokens": data["usage"]["completion_tokens"],
    }

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not set. Export it before running:\n"
            "    export GROQ_API_KEY=gsk_..."
        )

    all_rows   = []   # one entry per query
    domain_agg = {}   # domain -> {bap_steps, cot_tokens, trr rows}

    print("=" * 75)
    print("BAP vs TRUE CoT BASELINE")
    print(f"CoT model: {GROQ_MODEL}   |   prompt: '{COT_PROMPT[:40]}...'")
    print("=" * 75)

    for domain, cases in DOMAINS:
        print(f"\n{'='*75}")
        print(f"DOMAIN: {domain}  ({len(cases)} queries)")
        print(f"{'='*75}")

        # Load BAP — fresh space so strength updates don't bleed between domains
        space  = bap_engine.load_thought_space(domain)
        engine = bap_engine.BAPEngine(space)

        domain_rows = []

        for tc in cases:
            query    = tc["query"]
            expected = tc.get("expected_answer_contains", "")

            # ── BAP run ────────────────────────────────────────────────────────
            print(f"\n[BAP] Running: {query[:60]}")
            bap_result = engine.run(query, max_steps=10,
                                    expected_contains=expected)
            bap_steps  = bap_result.steps_taken

            # ── CoT run ────────────────────────────────────────────────────────
            print(f"[COT] Calling Groq...")
            try:
                cot_resp   = _call_groq_cot(query, api_key)
                cot_tokens = cot_resp["completion_tokens"]
                print(f"[COT] completion_tokens={cot_tokens}")
            except Exception as e:
                print(f"[COT] ERROR: {e} — skipping query")
                cot_tokens = None

            # ── True TRR ───────────────────────────────────────────────────────
            if cot_tokens and cot_tokens > 0:
                true_trr = round(bap_steps / cot_tokens, 4)
            else:
                true_trr = None

            row = {
                "domain":     domain,
                "query":      query,
                "bap_steps":  bap_steps,
                "cot_tokens": cot_tokens,
                "true_trr":   true_trr,
            }
            all_rows.append(row)
            domain_rows.append(row)

            # Rate-limit courtesy pause
            time.sleep(INTER_QUERY_SLEEP)

        domain_agg[domain] = domain_rows

    # ── Per-domain results table ───────────────────────────────────────────────

    print("\n\n" + "=" * 75)
    print("RESULTS TABLE")
    print("=" * 75)
    print(f"  {'Query':<45}  {'Dom':>4}  {'BAP':>4}  {'CoT':>5}  {'TRR':>7}")
    print(f"  {'-'*45}  {'-'*4}  {'-'*4}  {'-'*5}  {'-'*7}")

    for row in all_rows:
        dom_abbr  = row["domain"][:4]
        trr_str   = f"{row['true_trr']:.4f}" if row["true_trr"] is not None else "  N/A"
        cot_str   = str(row["cot_tokens"]) if row["cot_tokens"] is not None else "N/A"
        print(
            f"  {row['query'][:45]:<45}  {dom_abbr:>4}  "
            f"{row['bap_steps']:>4}  {cot_str:>5}  {trr_str:>7}"
        )

    # ── Domain averages ────────────────────────────────────────────────────────

    print("\n" + "=" * 75)
    print("DOMAIN AVERAGES  (True TRR = BAP steps / actual CoT completion tokens)")
    print("=" * 75)
    print(f"  {'Domain':<28}  {'Avg BAP':>7}  {'Avg CoT':>8}  {'Avg TRR':>8}  {'vs Estimated'}")
    print(f"  {'-'*28}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*13}")

    ESTIMATED = {
        "fibonacci_reasoning": 0.237,
        "arithmetic":          0.248,
        "multihop_facts":      0.310,
    }

    summary_rows = []
    for domain, rows in domain_agg.items():
        valid = [r for r in rows if r["true_trr"] is not None]
        if not valid:
            continue
        avg_bap = sum(r["bap_steps"]  for r in valid) / len(valid)
        avg_cot = sum(r["cot_tokens"] for r in valid) / len(valid)
        avg_trr = sum(r["true_trr"]   for r in valid) / len(valid)
        est     = ESTIMATED.get(domain, 0.0)
        diff    = avg_trr - est
        diff_str = f"{diff:+.4f}"
        print(
            f"  {domain:<28}  {avg_bap:>7.1f}  {avg_cot:>8.1f}  "
            f"{avg_trr:>8.4f}  {diff_str}"
        )
        summary_rows.append({
            "domain": domain, "avg_bap_steps": round(avg_bap, 2),
            "avg_cot_tokens": round(avg_cot, 2), "avg_true_trr": round(avg_trr, 4),
            "estimated_trr": est, "diff_vs_estimated": round(diff, 4),
        })

    # Global average
    global_valid = [r for r in all_rows if r["true_trr"] is not None]
    if global_valid:
        g_bap = sum(r["bap_steps"]  for r in global_valid) / len(global_valid)
        g_cot = sum(r["cot_tokens"] for r in global_valid) / len(global_valid)
        g_trr = sum(r["true_trr"]   for r in global_valid) / len(global_valid)
        print(f"  {'GLOBAL':<28}  {g_bap:>7.1f}  {g_cot:>8.1f}  {g_trr:>8.4f}")

    print("=" * 75)
    print(f"  Target: BAP TRR < 0.40   (CoT baseline TRR = 1.0 by definition)")
    print(f"  Improvement over CoT: {1/g_trr:.1f}x reduction in reasoning tokens")
    print("=" * 75)

    # ── Save JSON ──────────────────────────────────────────────────────────────

    output = {
        "model":        GROQ_MODEL,
        "prompt":       COT_PROMPT,
        "rows":         all_rows,
        "domain_summary": summary_rows,
        "global": {
            "avg_bap_steps":  round(g_bap, 2),
            "avg_cot_tokens": round(g_cot, 2),
            "avg_true_trr":   round(g_trr, 4),
            "cot_trr":        1.0,
            "token_reduction_factor": round(1 / g_trr, 2),
        } if global_valid else {},
    }

    out_path = "benchmark_cot_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[SAVED] {out_path}")


if __name__ == "__main__":
    main()
