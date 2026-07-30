#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RecProbe —— 确定性判分
======================================================================
对接 build_recprobe.py 生成的评测集。无需人工 / 大模型裁判。

判分规则：
    single_choice (P1/P2/P3/P5)  解析 "答案: X" 里的选项字母 → Accuracy
    ranking       (P4)           解析 "答案: C3 > C1 > ..." 完整排序
                                 → NDCG@K (基于 meta.relevance 分级增益)
                                 + Kendall-τ-b (模型序 vs 相关度分级序，含并列)

输入：
    --gold   评测集 jsonl (build_recprobe.py 的输出，含 answer/meta)
    --pred   预测 jsonl，每行至少含 {"task_id": ..., "output"/"prediction": "<模型原文>"}
             (也兼容预测文件本身就带 gold 字段的情形)
用法：
    python score_recprobe.py --gold all_tasks.jsonl --pred preds.jsonl
"""
import argparse
import json
import math
import re
from collections import defaultdict
from typing import Dict, List, Optional


# ---------------------------------------------------------------- 解析
ANS_LINE_RE = re.compile(r"答案[:：]\s*(.+)", re.IGNORECASE)


def get_answer_field(text: str) -> str:
    """从模型原文里抽 "答案:" 那一行；抽不到则用全文。"""
    if not text:
        return ""
    m = ANS_LINE_RE.search(text)
    return (m.group(1) if m else text).strip()


def parse_choice(text: str, valid_letters: List[str]) -> Optional[str]:
    """单选：从答案文本里抓第一个合法选项字母。"""
    seg = get_answer_field(text)
    for ch in seg:
        up = ch.upper()
        if up in valid_letters:
            return up
    # 退而求其次：全文里找
    for ch in (text or ""):
        if ch.upper() in valid_letters:
            return ch.upper()
    return None


def parse_ranking(text: str, valid_labels: List[str]) -> List[str]:
    """排序：按出现顺序抽取候选标签(如 视频A/视频B/...)，去重。
    标签直接来自 gold 的 relevance.keys()，对标签命名无硬编码假设。"""
    # 长标签优先，避免 "视频A" 被 "视频" 之类前缀截断
    pat = re.compile("|".join(re.escape(l) for l in
                              sorted(valid_labels, key=len, reverse=True)))
    seg = get_answer_field(text)
    found, seen = [], set()
    for tok in pat.findall(seg):
        if tok not in seen:
            seen.add(tok)
            found.append(tok)
    if not found:  # 兜底扫全文
        for tok in pat.findall(text or ""):
            if tok not in seen:
                seen.add(tok)
                found.append(tok)
    return found


# ---------------------------------------------------------------- 指标
def dcg(gains: List[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(pred_order: List[str], relevance: Dict[str, int], k: Optional[int] = None) -> float:
    labels = list(relevance.keys())
    k = k or len(labels)
    # 不做兜底补全：只对模型【实际给出】的顺序计分，未排到的标签直接舍弃
    # (解析不到标签时 pred_order 为空 -> gains 为空 -> NDCG=0，不再蹭分)
    gains = [relevance.get(l, 0) for l in pred_order[:k]]
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = dcg(ideal)
    return dcg(gains) / idcg if idcg > 0 else 0.0


def kendall_tau(pred_order: List[str], relevance: Dict[str, int]) -> float:
    """排序一致性 (Kendall-τ / Goodman-Kruskal Γ 形式)。
    只在【金标准相关度不同】的候选对上计分：高相关却被排到低相关之后 = 逆序对。
    同级候选谁前谁后不惩罚 → 完美排序=1.0，完全反序=-1.0。"""
    labels = list(relevance.keys())
    # 不做兜底补全：未被模型排序的标签视为"未给出"，排到所有已给标签之后
    pred_rank = {l: i for i, l in enumerate(pred_order)}
    unranked = len(pred_order)                          # 未给出标签统一的"末尾"名次
    true_score = {l: relevance.get(l, 0) for l in labels}
    n = len(labels)
    concord = discord = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = labels[i], labels[j]
            if true_score[a] == true_score[b]:
                continue                                 # 金标准并列对：不计分
            # 真值更该推的那个，是否被模型排在更前
            hi, lo = (a, b) if true_score[a] > true_score[b] else (b, a)
            if pred_rank.get(hi, unranked) < pred_rank.get(lo, unranked):
                concord += 1
            else:
                discord += 1
    denom = concord + discord
    return (concord - discord) / denom if denom > 0 else 0.0


# ---------------------------------------------------------------- 主流程
def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_pred_text(p: Dict) -> str:
    for key in ("output", "prediction", "pred", "response", "generated", "answer_raw"):
        if key in p and p[key]:
            return str(p[key])
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--k", type=int, default=5, help="NDCG@K")
    ap.add_argument("--report", default=None, help="可选：把明细结果写到这个 json")
    args = ap.parse_args()

    gold = {g["task_id"]: g for g in load_jsonl(args.gold)}
    preds = {p["task_id"]: p for p in load_jsonl(args.pred) if "task_id" in p}

    per_task = defaultdict(lambda: {"n": 0, "acc_hit": 0,
                                    "ndcg_sum": 0.0, "tau_sum": 0.0,
                                    "rank_n": 0, "invalid": 0})
    missing = 0
    for tid, g in gold.items():
        p = preds.get(tid)
        if p is None:
            missing += 1
            continue
        text = get_pred_text(p)
        name = g["task_name"]
        st = per_task[name]
        st["n"] += 1
        if g["qtype"] == "single_choice":
            letters = list(g["options"].keys())
            pred = parse_choice(text, letters)
            if pred == g["answer"]:
                st["acc_hit"] += 1
        elif g["qtype"] == "ranking":
            rel = {k: int(v) for k, v in g["meta"]["relevance"].items()}
            order = parse_ranking(text, list(rel.keys()))
            st["rank_n"] += 1
            # 无兜底：模型必须给出【完整且合法】的排序（覆盖全部候选标签），
            # 否则判为无效 -> NDCG/τ 记 0，不再用原序补全蹭分。
            if len(order) == len(rel) and set(order) == set(rel.keys()):
                st["ndcg_sum"] += ndcg_at_k(order, rel, args.k)
            else:
                st["invalid"] += 1

    # 汇总
    print("=" * 71)
    print(f"{'task':<20}{'n':>7}{'ACC':>10}{'NDCG@'+str(args.k):>12}{'invalid':>9}")
    print("-" * 71)
    summary = {}
    for name in sorted(per_task):
        st = per_task[name]
        acc = st["acc_hit"] / st["n"] if st["n"] and st["rank_n"] == 0 else None
        # NDCG 在【全部排序题】上取均值（无效条以 0 计入分子），更真实反映可用性
        ndcg = st["ndcg_sum"] / st["rank_n"] if st["rank_n"] else None
        invalid = st["invalid"] if st["rank_n"] else None
        summary[name] = {"n": st["n"], "acc": acc, "ndcg": ndcg,
                         "invalid": invalid}
        print(f"{name:<20}{st['n']:>7}"
              f"{('%.4f' % acc) if acc is not None else '-':>10}"
              f"{('%.4f' % ndcg) if ndcg is not None else '-':>12}"
              f"{(str(invalid)) if invalid is not None else '-':>9}")
    print("-" * 71)
    # 单选总体 ACC (P1/P2/P3/P5)
    mc_n = sum(s["n"] for k, s in per_task.items() if s["rank_n"] == 0)
    mc_hit = sum(s["acc_hit"] for k, s in per_task.items() if s["rank_n"] == 0)
    if mc_n:
        print(f"{'[单选总体 ACC]':<20}{mc_n:>7}{mc_hit/mc_n:>10.4f}")
    if missing:
        print(f"[WARN] {missing} 条 gold 在预测里缺失，未计分。")
    print("=" * 71)

    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "missing": missing,
                       "k": args.k}, f, ensure_ascii=False, indent=2)
        print(f"明细已写出: {args.report}")


if __name__ == "__main__":
    main()
