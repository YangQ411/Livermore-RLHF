import os
import sys
import csv
import time
import json
import yaml
import argparse
from tqdm import tqdm
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# ---------- OpenAI client (>=1.0 style) ----------
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


JUDGE_SYSTEM_PROMPT = """You are a meticulous evaluator channeling Jesse Livermore's trading discipline.
Score answers on a 0–10 scale for how well they reflect *Livermore-style* principles:
1) Risk management & capital preservation
2) Discipline & patience (avoid impulsive novelty)
3) Evidence-driven reasoning; no bold claims without basis
4) Clarity & actionability (rules/checklists); avoid noise-chasing
5) Humility: acknowledge uncertainty; avoid confident price predictions
6) Consistency with Livermore ethos (cut losses, let winners run, sit tight when appropriate)
Penalize hallucinations, unjustified forecasts, and inconsistency.

Return STRICT JSON with keys:
- scores: object of {model_key: number 0..10}
- winner: one of the model keys with the highest score (or null if tie)
- rationale: short text (~2-4 sentences) explaining the ranking
- flags: { "hallucination": boolean, "overconfidence": boolean }
Only output JSON. No prose outside JSON.
"""

JUDGE_USER_TEMPLATE = """Task category: {category}
Prompt:
{prompt}

Candidate answers:
{candidates}

Please score each candidate and select a winner using the rubric.
Model keys MUST match exactly the ones shown in "Candidate answers".
"""


# ------------------------ Config ------------------------

def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_cfg(cfg: Dict[str, Any], key: str, default=None):
    parts = key.split(".")
    cur = cfg
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


# --------------------- Data helpers ---------------------

def extract_prompt(sample: Dict[str, Any]) -> str:
    for key in ("prompt", "question", "input", "query"):
        v = sample.get(key)
        if isinstance(v, str) and v.strip():
            return v
    raise ValueError("Missing prompt. Expected one of: 'prompt','question','input','query'.")


def extract_category(sample: Dict[str, Any]) -> str:
    for key in ("category", "label", "big_label", "tag"):
        v = sample.get(key)
        if isinstance(v, str) and v.strip():
            return v
    return "uncategorized"


def detect_answer_mapping(sample: Dict[str, Any]) -> Dict[str, str]:
    """
    Detect A/B/C model-to-answer mapping:
    returns dict like {'a_model': 'answer_a', 'b_model': 'answer_b', 'c_model': 'answer_c'}
    """
    mapping = {}
    pairs = [("a_model", "answer_a"), ("b_model", "answer_b"), ("c_model", "answer_c")]
    for mk, ak in pairs:
        if mk in sample and ak in sample and isinstance(sample[ak], str):
            mapping[mk] = ak
    if not mapping:
        raise ValueError("Could not detect (a_model/b_model/c_model) paired with (answer_a/b/c).")
    return mapping


# -------------------- OpenAI calling --------------------

@dataclass
class Judgment:
    idx: int
    category: str
    scores: Dict[str, float]    # keys: a_model/b_model/c_model
    winner: Optional[str]       # one of the above keys
    rationale: str
    flags: Dict[str, Any]


def call_openai_judge(client, model_name: str, system_prompt: str, user_prompt: str,
                      retries: int = 3, backoff: float = 2.0) -> Dict[str, Any]:
    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = resp.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            if attempt == retries:
                raise
            time.sleep(backoff)
            backoff *= 2.0
    raise RuntimeError("Unreachable")


# ------------------------ Main -------------------------

def main():
    # config path
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config = load_yaml_config(args.config)

    # ---- pull values from YAML ----
    input_path  = config['gpt_eval']['input_file']
    output_dir = config['gpt_eval']['save_dir']
    api_key = config['gpt_eval']['openai_api_key']
    model_name = config['gpt_eval']['judge_model']
    max_eval = config['gpt_eval']['max_eval']    
    sleep_time = float(config['gpt_eval']['sleep'])
    category_flt = config['gpt_eval']['category_filter']
    retries  = config['gpt_eval']['retries']
    backoff  = float(config['gpt_eval']['wait_sec'])

    if not input_path or not os.path.exists(input_path):
        print(f"Input file not found: {input_path}", file=sys.stderr); sys.exit(2)
    if not api_key:
        print("OpenAI API key missing (set in config or environment OPENAI_API_KEY).", file=sys.stderr); sys.exit(2)
    if OpenAI is None:
        print("openai package missing. Run: pip install openai", file=sys.stderr); sys.exit(2)

    os.makedirs(output_dir, exist_ok=True)
    detailed_path = os.path.join(output_dir, "judgments.jsonl")
    overall_csv   = os.path.join(output_dir, "summary_overall.csv")
    bycat_csv     = os.path.join(output_dir, "summary_by_category.csv")

    # ---- load data ----
    with open(input_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    rows = raw if isinstance(raw, list) else raw.get("data", [])
    if not rows:
        print("Input must be a list[dict] or a dict with key 'data' as list.", file=sys.stderr); sys.exit(2)
    if max_eval > 0:
        rows = rows[:max_eval]

    # detect mapping from first row
    mapping = detect_answer_mapping(rows[0])   
    model_keys = list(mapping.keys())           

    client = OpenAI(api_key=api_key)

    # ---- run judging ----
    judgments: List[Judgment] = []
    with open(detailed_path, "w", encoding="utf-8") as out_f:
        kept = 0
        print(f"Starting GPT judging on {len(rows)} items using {model_name}...")
        for idx, sample in enumerate(tqdm(rows, desc="Judging progress", unit="item")):
            try:
                cat = extract_category(sample)
                if category_flt and category_flt != cat:
                    continue
                prompt = extract_prompt(sample)

                cand_lines = []
                for mk, ak in mapping.items():
                    ans = sample.get(ak, "")
                    cand_lines.append(f"- {mk}:\n{ans}\n")
                candidates_block = "\n".join(cand_lines)

                user_prompt = JUDGE_USER_TEMPLATE.format(category=cat, prompt=prompt, candidates=candidates_block)
                result = call_openai_judge(client, model_name, JUDGE_SYSTEM_PROMPT, user_prompt,
                                       retries=retries, backoff=backoff)

                raw_scores = result.get("scores", {})
                scores = {k: float(v) for k, v in raw_scores.items() if k in model_keys}
                winner = result.get("winner", None)
                if winner not in scores:
                    winner = None

                jud = Judgment(
                    idx=idx,
                    category=cat,
                    scores=scores,
                    winner=winner,
                    rationale=result.get("rationale", ""),
                    flags=result.get("flags", {}),
                    )
                judgments.append(jud)

                out_f.write(json.dumps({
                    "idx": idx,
                    "category": cat,
                    "a_maps_to": sample.get("a_model", ""),
                    "b_maps_to": sample.get("b_model", ""),
                    "c_maps_to": sample.get("c_model", ""),
                    "scores": scores,
                    "winner": winner,
                    "rationale": jud.rationale,
                    "flags": jud.flags
                    }, ensure_ascii=False) + "\n")

                kept += 1
                if sleep_time > 0:
                    time.sleep(sleep_time)

            except Exception as e:
                sys.stderr.write(f"[Row {idx}] ERROR: {e}\n")
                continue

    # ---- aggregate to TRUE model names (baseline_model/sft_model/dpo_model) ----
    # reload input for exact A/B/C→true mapping per row
    sum_scores = Counter()
    count_scores = Counter()
    win_counts = Counter()
    bycat_sum = defaultdict(Counter)
    bycat_cnt = defaultdict(Counter)
    bycat_wins = defaultdict(Counter)

    # iterate in lockstep over rows & judgments that were kept (category filter respected)
    j_iter = iter(judgments)
    for idx, sample in enumerate(rows):
        try:
            j = next(j_iter)
        except StopIteration:
            break

        cat = j.category
        amap, bmap, cmap = sample.get("a_model", ""), sample.get("b_model", ""), sample.get("c_model", "")

        # scores
        for mk, sc in j.scores.items():
            if mk == "a_model" and amap:
                sum_scores[amap] += sc; count_scores[amap] += 1
                bycat_sum[cat][amap] += sc; bycat_cnt[cat][amap] += 1
            elif mk == "b_model" and bmap:
                sum_scores[bmap] += sc; count_scores[bmap] += 1
                bycat_sum[cat][bmap] += sc; bycat_cnt[cat][bmap] += 1
            elif mk == "c_model" and cmap:
                sum_scores[cmap] += sc; count_scores[cmap] += 1
                bycat_sum[cat][cmap] += sc; bycat_cnt[cat][cmap] += 1

        # winner
        if j.winner == "a_model" and amap:
            win_counts[amap] += 1; bycat_wins[cat][amap] += 1
        elif j.winner == "b_model" and bmap:
            win_counts[bmap] += 1; bycat_wins[cat][bmap] += 1
        elif j.winner == "c_model" and cmap:
            win_counts[cmap] += 1; bycat_wins[cat][cmap] += 1

    def avg_for(m):
        c = count_scores.get(m, 0)
        return (sum_scores[m] / c) if c else None

    true_models = list(count_scores.keys())
    preferred_order = ["dpo_model", "sft_model", "baseline_model"]
    ordered = [m for m in preferred_order if m in true_models] + [m for m in true_models if m not in preferred_order]

    # ---- print console summary (overall + per-category) ----
    print("\nAverage Livermore-style Scores:\nmodel")
    for m in ordered:
        a = avg_for(m); print(f"{m: <15} {a:.2f}" if a is not None else f"{m: <15} N/A")

    print("\nWin counts per model:")
    for m in ordered:
        print(f"{m: <15} {win_counts.get(m, 0)}")

    print("\nFinal summary:\n")
    header = f"{'model': <15} {'avg_score': >9} {'wins': >6} {'count': >7}"
    print(header)
    for m in ordered:
        a = avg_for(m); w = win_counts.get(m, 0); c = count_scores.get(m, 0)
        print(f"{m: <15} {(f'{a:.2f}' if a is not None else 'N/A'): >9} {w: >6} {c: >7}")

    print("\nPer-category breakdown:")
    print(header)
    for cat in sorted(bycat_sum.keys()):
        print(f"\n[{cat}]")
        for m in ordered:
            c = bycat_cnt[cat].get(m, 0)
            a = (bycat_sum[cat][m] / c) if c else None
            w = bycat_wins[cat].get(m, 0)
            print(f"{m: <15} {(f'{a:.2f}' if a is not None else 'N/A'): >9} {w: >6} {c: >7}")

    # ---- save CSV summaries ----
    with open(overall_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "avg_score", "wins", "count"])
        for m in ordered:
            a = avg_for(m)
            w.writerow([m, f"{a:.4f}" if a is not None else "", win_counts.get(m, 0), count_scores.get(m, 0)])

    with open(bycat_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["category", "model", "avg_score", "wins", "count"])
        for cat in sorted(bycat_sum.keys()):
            for m in ordered:
                c = bycat_cnt[cat].get(m, 0)
                if c:
                    a = bycat_sum[cat][m] / c
                    w.writerow([cat, m, f"{a:.4f}", bycat_wins[cat].get(m, 0), c])

    print(f"\nSaved detailed judgments to: {os.path.abspath(detailed_path)}")
    print(f"Saved overall summary CSV:   {os.path.abspath(overall_csv)}")
    print(f"Saved per-category CSV:      {os.path.abspath(bycat_csv)}")


if __name__ == "__main__":
    main()