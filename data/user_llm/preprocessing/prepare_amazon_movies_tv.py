#!/usr/bin/env python3
"""从 Amazon 2014 Movies & TV 原始数据构造 USER-LLM 风格数据。

默认读取 ``../amazon2014/`` 下的 reviews / meta，全部输出到
``../amazon_movies_tv/``（即 ``seqllm/data/user_llm/amazon_movies_tv/``）：

- ``id_descriptions.json`` / ``asin_to_id.tsv`` / ``special_tokens.json``
- ``pretrain_train.jsonl`` / ``pretrain_test.jsonl``
- ``review_store.jsonl``（item token 与序列一致，均为 ``<item_{int}>``）
- ``warmup_train.jsonl`` / ``warmup_eval.jsonl`` / ``warmup_stats.json``
- ``favcategory_train.jsonl`` / ``favcategory_test.jsonl``
- ``reviewg_train.jsonl`` / ``reviewg_test.jsonl``
- ``stats.json``

Token 约定（由已训练 vocab 反推，脚本内不依赖任何 added_tokens 文件）：
1. 交互三元组：``<item_i> <category_j> <az_rating_k>``
2. item / category id 均为从 0 起的连续整数；rating 固定 ``k=0..4`` 对应星级 1..5
3. 扩词表顺序：全部 ``<item_*>`` → 全部 ``<category_*>`` → ``<az_rating_0..4>``
4. item id：保留用户交互里出现过的 ASIN，按 ASIN 字典序编号
5. category：优先取 ``Movies & TV`` 路径的最深叶子；允许空类别 ``""``
6. 交互不要求 ASIN 必须出现在 meta 中（缺 meta 时 title/category 置空）

其它：
- ReviewG 按 ``(user_id, timestamp, item_token)`` 精确匹配评论
- ``<az_rating_k>`` 对应星级 ``k+1``，instruction 使用真实星级

示例：
    python3 prepare_amazon_movies_tv.py
    python3 prepare_amazon_movies_tv.py --stages pretrain warmup
    python3 prepare_amazon_movies_tv.py --stages favcategory reviewg
"""

from __future__ import annotations

import argparse
import gzip
import html
import json
import random
import re
from ast import literal_eval
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover

    def tqdm(iterable=None, **kwargs):  # type: ignore[no-redef]
        return iterable if iterable is not None else iter(())


SCRIPT_DIR = Path(__file__).resolve().parent
USERLLM_DIR = SCRIPT_DIR.parent  # .../seqllm/data/user_llm
AMAZON_DIR = USERLLM_DIR / "amazon2014"
# 输出与脚本同级：seqllm/data/user_llm/amazon_movies_tv/
# （不要写到 workspace 根下的 data/user_llm/）
DEFAULT_OUTPUT_DIR = USERLLM_DIR / "amazon_movies_tv"

TOKENS_PER_INTERACTION = 3
TOTAL_REVIEWS = 4_607_047  # Movies_and_TV.json.gz approx
TOTAL_META = 208_321

WARMUP_INSTRUCTION = (
    "Based on the user's historical interaction sequence with movies and TV "
    "shows, predict the user's future interaction sequence with movies and TV "
    "shows."
)
FAVCATEGORY_INSTRUCTIONS = [
    "What is the user's favorite category?",
    "What category does the user prefer most?",
    "Based on the user's history, what is their favorite category?",
    "Identify the user's most preferred category.",
    "What type of items does this user like best?",
]
REVIEWG_INSTRUCTIONS = [
    "Write a review for {item_name} with a rating of {rating}.",
    "Please write a review for {item_name}. Your rating is {rating}.",
    "Based on your experience, write a review for {item_name} with rating {rating}.",
    "Share your thoughts about {item_name}. Rating: {rating}.",
    "Generate a review for {item_name} (Rating: {rating}).",
]

# Amazon overall ratings 1..5 -> <az_rating_0> .. <az_rating_4>
RATING_VALUES = [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def normalize_rating(raw) -> str:
    value = float(raw)
    if value.is_integer():
        return str(int(value))
    return f"{value:g}"


def rating_to_token(rating: str) -> str:
    value = int(float(rating))
    if value not in RATING_VALUES:
        raise ValueError(f"unexpected rating: {rating}")
    return f"<az_rating_{value - 1}>"


def rating_token_to_stars(token: str) -> int:
    """`<az_rating_k>` -> star value k+1."""
    if not (token.startswith("<az_rating_") and token.endswith(">")):
        raise ValueError(f"bad rating token: {token}")
    return int(token[len("<az_rating_") : -1]) + 1


def rating_description(stars: int) -> str:
    return f"The user's rating for this movie or TV show is {stars}"


def count_lines(path: Path) -> int:
    total = 0
    with path.open("rb") as file:
        for _ in file:
            total += 1
    return total


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(obj, file, ensure_ascii=False, indent=2)


def parse_interactions(text: str) -> List[Tuple[str, str, str]]:
    tokens = text.split()
    if len(tokens) % TOKENS_PER_INTERACTION != 0:
        raise ValueError(
            f"token count {len(tokens)} is not divisible by "
            f"{TOKENS_PER_INTERACTION}"
        )
    return [
        (tokens[i], tokens[i + 1], tokens[i + 2])
        for i in range(0, len(tokens), TOKENS_PER_INTERACTION)
    ]


def interactions_to_text(interactions: List[Tuple[str, str, str]]) -> str:
    return " ".join(
        f"{item} {category} {rating}"
        for item, category, rating in interactions
    )


def open_text_maybe_gzip(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


_BRACKET_TAG_RE = re.compile(r"\[([^\]]+)\]")


def normalize_title(title: str) -> str:
    """对齐已有产物的标题清洗：``[VHS]``→``VHS``，并做基础 HTML 反转义。"""
    if not title:
        return ""
    text = html.unescape(str(title)).strip()
    text = _BRACKET_TAG_RE.sub(r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _category_paths(categories) -> List[List[str]]:
    if not isinstance(categories, list):
        return []
    paths: List[List[str]] = []
    for path in categories:
        if isinstance(path, list):
            cleaned = [e.strip() for e in path if isinstance(e, str) and e.strip()]
            if cleaned:
                paths.append(cleaned)
        elif isinstance(path, str) and path.strip():
            paths.append([path.strip()])
    return paths


def leaf_category(categories) -> str:
    """优先取 ``Movies & TV`` 路径的最深叶子；否则取任意路径最深叶子。

    反推依据：参考词表含 Drama / Comedy / All * Titles 等细类，而非仅
    ``Movies`` / ``TV``；空字符串也是合法 category（可映射为某个
    ``<category_*>``）。
    """
    paths = _category_paths(categories)
    if not paths:
        return ""

    movies_tv = [p for p in paths if p[0] == "Movies & TV"]
    candidates = movies_tv or paths
    best = max(candidates, key=len)
    return best[-1]


def item_meta_or_empty(
    meta: Dict[str, Dict[str, str]], asin: str
) -> Dict[str, str]:
    return meta.get(asin) or {"title": "", "category": ""}


# ---------------------------------------------------------------------------
# Load meta / reviews
# ---------------------------------------------------------------------------

def load_meta(meta_path: Path) -> Dict[str, Dict[str, str]]:
    meta: Dict[str, Dict[str, str]] = {}
    with open_text_maybe_gzip(meta_path) as file:
        for line in tqdm(
            file,
            total=TOTAL_META,
            desc="scan meta",
            unit="item",
            mininterval=1.0,
        ):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                try:
                    obj = literal_eval(line)
                except (ValueError, SyntaxError):
                    continue
            asin = obj.get("asin")
            if not asin:
                continue
            title = obj.get("title") or ""
            if not isinstance(title, str):
                title = str(title)
            categories = obj.get("categories", obj.get("category"))
            meta[asin] = {
                "title": normalize_title(title),
                "category": leaf_category(categories),
            }
    return meta


def first_pass_user_counts(reviews_path: Path) -> Dict[str, int]:
    """统计用户交互数；不要求 ASIN 出现在 meta 中。"""
    user_count: Dict[str, int] = defaultdict(int)
    with open_text_maybe_gzip(reviews_path) as file:
        for line in tqdm(
            file,
            total=TOTAL_REVIEWS,
            desc="scan reviews pass1",
            unit="row",
            mininterval=1.0,
        ):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            uid = obj.get("reviewerID")
            asin = obj.get("asin")
            if not uid or not asin:
                continue
            if obj.get("unixReviewTime") is None:
                continue
            user_count[uid] += 1
    return user_count


def second_pass_collect(
    reviews_path: Path,
    kept_users: set,
) -> Dict[str, List[Tuple[int, str, str, str]]]:
    """Return {user: [(timestamp, asin, rating_str, summary), ...]}."""
    rows_by_user: Dict[str, List[Tuple[int, str, str, str]]] = defaultdict(list)
    with open_text_maybe_gzip(reviews_path) as file:
        for line in tqdm(
            file,
            total=TOTAL_REVIEWS,
            desc="scan reviews pass2",
            unit="row",
            mininterval=1.0,
        ):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            uid = obj.get("reviewerID")
            if uid not in kept_users:
                continue
            asin = obj.get("asin")
            if not asin:
                continue
            try:
                ts = int(obj["unixReviewTime"])
            except (KeyError, TypeError, ValueError):
                continue
            rating = normalize_rating(obj.get("overall", 0))
            try:
                if int(float(rating)) not in RATING_VALUES:
                    continue
            except ValueError:
                continue
            summary = obj.get("summary")
            if not isinstance(summary, str):
                summary = ""
            rows_by_user[uid].append((ts, asin, rating, summary.strip()))
    return rows_by_user


def build_special_tokens(
    num_items: int, num_categories: int, num_ratings: int = len(RATING_VALUES)
) -> List[str]:
    """扩词表用 token 列表：item → category → az_rating（与参考 vocab 布局一致）。"""
    tokens = [f"<item_{i}>" for i in range(num_items)]
    tokens.extend(f"<category_{i}>" for i in range(num_categories))
    tokens.extend(f"<az_rating_{i}>" for i in range(num_ratings))
    return tokens


# ---------------------------------------------------------------------------
# Stage 1: pretrain + id_descriptions + review_store
# ---------------------------------------------------------------------------

def build_pretrain(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    meta = load_meta(args.meta)
    print(f"[pretrain] meta items: {len(meta)}")

    user_count = first_pass_user_counts(args.reviews)
    kept_users = {
        uid for uid, cnt in user_count.items() if cnt >= args.min_interactions
    }
    print(
        f"[pretrain] users >= {args.min_interactions}: {len(kept_users)} "
        f"/ {len(user_count)}"
    )
    del user_count

    rows_by_user = second_pass_collect(args.reviews, kept_users)

    # Contiguous item ids over ASINs that actually appear, sorted for stability.
    seen_asins = sorted(
        {asin for rows in rows_by_user.values() for _, asin, _, _ in rows}
    )
    asin_to_id = {asin: idx for idx, asin in enumerate(seen_asins)}

    # Category ids: first-seen order while iterating sorted ASINs.
    category_id_map: Dict[str, int] = {}
    for asin in seen_asins:
        cat = item_meta_or_empty(meta, asin)["category"]
        if cat not in category_id_map:
            category_id_map[cat] = len(category_id_map)

    id_desc = {
        "item": {
            f"<item_{asin_to_id[asin]}>": item_meta_or_empty(meta, asin)["title"]
            for asin in seen_asins
        },
        "category": {
            f"<category_{cid}>": cat for cat, cid in category_id_map.items()
        },
        "rating": {
            f"<az_rating_{idx}>": rating_description(stars)
            for idx, stars in enumerate(RATING_VALUES)
        },
    }
    write_json(args.output_dir / "id_descriptions.json", id_desc)

    special_tokens = build_special_tokens(len(asin_to_id), len(category_id_map))
    write_json(args.output_dir / "special_tokens.json", special_tokens)
    print(
        f"[pretrain] special_tokens: items={len(asin_to_id)} "
        f"categories={len(category_id_map)} ratings={len(RATING_VALUES)} "
        f"total={len(special_tokens)}"
    )

    asin_map_path = args.output_dir / "asin_to_id.tsv"
    with asin_map_path.open("w", encoding="utf-8") as file:
        for asin, idx in asin_to_id.items():
            file.write(f"{asin}\t{idx}\n")
    print(f"[pretrain] wrote {asin_map_path} ({len(asin_to_id)} items)")

    train_path = args.output_dir / "pretrain_train.jsonl"
    test_path = args.output_dir / "pretrain_test.jsonl"
    review_store_path = args.output_dir / "review_store.jsonl"

    kept = 0
    train_windows = 0
    test_windows = 0
    review_records = 0
    meta_hit_items = sum(1 for asin in seen_asins if asin in meta)

    with train_path.open("w", encoding="utf-8") as train_f, test_path.open(
        "w", encoding="utf-8"
    ) as test_f, review_store_path.open("w", encoding="utf-8") as rev_f:
        for user_id in tqdm(
            sorted(rows_by_user.keys()),
            desc="emit windows",
            unit="user",
            mininterval=1.0,
        ):
            rows = rows_by_user[user_id]
            rows.sort(key=lambda item: (item[0], item[1]))
            if len(rows) < args.min_interactions:
                continue
            kept += 1

            # review_store uses the SAME numeric item tokens as the sequence.
            for ts, asin, _rating, summary in rows:
                item_token = f"<item_{asin_to_id[asin]}>"
                rev_f.write(
                    json.dumps(
                        {
                            "user_id": user_id,
                            "item_token": item_token,
                            "asin": asin,
                            "timestamp": ts,
                            "summary": summary,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                review_records += 1

            tokens = []
            timestamps = []
            for ts, asin, rating, _summary in rows:
                item_id = asin_to_id[asin]
                cat = item_meta_or_empty(meta, asin)["category"]
                cid = category_id_map[cat]
                tokens.append(
                    f"<item_{item_id}> <category_{cid}> {rating_to_token(rating)}"
                )
                timestamps.append(ts)

            num_windows = len(tokens) - args.seq_len + 1
            last_idx = num_windows - 1
            for start in range(num_windows):
                record = {
                    "user_id": user_id,
                    "text": " ".join(tokens[start : start + args.seq_len]),
                    "timestamps": timestamps[start : start + args.seq_len],
                }
                line = json.dumps(record, ensure_ascii=False) + "\n"
                if start == last_idx:
                    test_f.write(line)
                    test_windows += 1
                else:
                    train_f.write(line)
                    train_windows += 1

    stats = {
        "dataset": "AmazonReview_MoviesAndTV",
        "seq_len": args.seq_len,
        "min_interactions": args.min_interactions,
        "kept_users": kept,
        "train_windows": train_windows,
        "test_windows": test_windows,
        "num_items": len(asin_to_id),
        "num_categories": len(category_id_map),
        "num_ratings": len(RATING_VALUES),
        "meta_coverage_items": meta_hit_items,
        "special_tokens": len(special_tokens),
        "review_records": review_records,
        "reviews": str(args.reviews),
        "meta": str(args.meta),
        "token_layout": "item -> category -> az_rating",
    }
    write_json(args.output_dir / "pretrain_stats.json", stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


# ---------------------------------------------------------------------------
# Stage 2: warmup
# ---------------------------------------------------------------------------

def split_window_by_ratio(
    tokens: List[str], ratio: float
) -> Tuple[List[str], List[str], int]:
    n_tokens = len(tokens)
    if n_tokens % TOKENS_PER_INTERACTION != 0:
        raise ValueError(
            f"expected token count multiple of {TOKENS_PER_INTERACTION}, "
            f"got {n_tokens}"
        )
    n_interactions = n_tokens // TOKENS_PER_INTERACTION
    n_input = max(1, min(n_interactions - 1, round(ratio * n_interactions)))
    split_idx = n_input * TOKENS_PER_INTERACTION
    return tokens[:split_idx], tokens[split_idx:], n_input


def build_warmup_train(args: argparse.Namespace) -> dict:
    src = args.output_dir / "pretrain_train.jsonl"
    dst = args.output_dir / "warmup_train.jsonl"
    if not src.is_file():
        raise FileNotFoundError(f"missing {src}; run --stages pretrain first")

    total = count_lines(src)
    n_seen = 0
    n_in = 0
    n_out = 0
    hist: Dict[int, int] = {}

    with src.open("r", encoding="utf-8") as fin, dst.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in tqdm(
            fin, total=total, desc="warmup_train", unit="line", mininterval=1.0
        ):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            tokens = rec["text"].split()
            input_tokens, output_tokens, n_input = split_window_by_ratio(
                tokens, args.input_ratio
            )
            n_total = len(tokens) // TOKENS_PER_INTERACTION
            hist[n_total] = hist.get(n_total, 0) + 1

            out_rec = {
                "instruction": WARMUP_INSTRUCTION,
                "input": " ".join(input_tokens),
                "output": " ".join(output_tokens),
                "user_id": rec["user_id"],
            }
            timestamps = rec.get("timestamps")
            if isinstance(timestamps, list) and len(timestamps) == n_total:
                out_rec["input_timestamps"] = timestamps[:n_input]
                out_rec["output_timestamps"] = timestamps[n_input:]

            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            n_seen += 1
            n_in += len(input_tokens)
            n_out += len(output_tokens)

    summary = {
        "source": str(src),
        "dest": str(dst),
        "num_records": n_seen,
        "avg_input_tokens": round(n_in / max(1, n_seen), 2),
        "avg_output_tokens": round(n_out / max(1, n_seen), 2),
        "tokens_per_interaction": TOKENS_PER_INTERACTION,
        "input_ratio": args.input_ratio,
        "interactions_per_window_histogram": dict(sorted(hist.items())),
    }
    write_json(args.output_dir / "warmup_stats.json", summary)
    print(f"[warmup] train -> {dst} ({n_seen} records)")
    return summary


def build_warmup_eval(args: argparse.Namespace) -> dict:
    src = args.output_dir / "pretrain_test.jsonl"
    dst = args.output_dir / "warmup_eval.jsonl"
    if not src.is_file():
        raise FileNotFoundError(f"missing {src}; run --stages pretrain first")

    total = count_lines(src)
    n_seen = 0
    skipped = 0
    with src.open("r", encoding="utf-8") as fin, dst.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in tqdm(
            fin, total=total, desc="warmup_eval", unit="line", mininterval=1.0
        ):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            interactions = parse_interactions(rec["text"])
            if len(interactions) < 2:
                skipped += 1
                continue
            out_rec = {
                "instruction": WARMUP_INSTRUCTION,
                "input": interactions_to_text(interactions[:-1]),
                "output": interactions_to_text([interactions[-1]]),
                "user_id": rec["user_id"],
            }
            timestamps = rec.get("timestamps")
            if isinstance(timestamps, list) and len(timestamps) == len(interactions):
                out_rec["timestamps"] = timestamps
            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            n_seen += 1

    print(f"[warmup] eval -> {dst} ({n_seen} records, skipped={skipped})")
    return {"num_records": n_seen, "skipped": skipped}


# ---------------------------------------------------------------------------
# Stage 3: favcategory
# ---------------------------------------------------------------------------

def most_frequent_category(interactions: List[Tuple[str, str, str]]) -> str:
    return Counter(cat for _, cat, _ in interactions).most_common(1)[0][0]


def build_favcategory(args: argparse.Namespace) -> dict:
    id_desc_path = args.output_dir / "id_descriptions.json"
    train_src = args.output_dir / "pretrain_train.jsonl"
    test_src = args.output_dir / "pretrain_test.jsonl"
    for path in (id_desc_path, train_src, test_src):
        if not path.is_file():
            raise FileNotFoundError(f"missing {path}; run --stages pretrain first")

    with id_desc_path.open("r", encoding="utf-8") as file:
        id_descriptions = json.load(file)
    category_desc = id_descriptions.get("category", {})

    counts = {}
    for split, src in (("train", train_src), ("test", test_src)):
        dst = args.output_dir / f"favcategory_{split}.jsonl"
        rng = random.Random(args.seed + (0 if split == "train" else 1))
        total = count_lines(src)
        written = 0
        with src.open("r", encoding="utf-8") as fin, dst.open(
            "w", encoding="utf-8"
        ) as fout:
            for line in tqdm(
                fin,
                total=total,
                desc=f"favcategory_{split}",
                unit="line",
                mininterval=1.0,
            ):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                interactions = parse_interactions(rec["text"])
                if len(interactions) < 3:
                    continue
                fav = most_frequent_category(interactions)
                sample = {
                    "task": "favcategory",
                    "split": split,
                    "user_id": str(rec["user_id"]),
                    "instruction": rng.choice(FAVCATEGORY_INSTRUCTIONS),
                    "input": interactions_to_text(interactions),
                    "output": fav,
                    "metadata": {
                        "history_length": len(interactions),
                        "category_description": category_desc.get(fav, fav),
                    },
                }
                fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                written += 1
        counts[split] = written
        print(f"[favcategory] {split} -> {dst} ({written})")

    return counts


# ---------------------------------------------------------------------------
# Stage 4: reviewg
# ---------------------------------------------------------------------------

def load_review_index(
    review_store_path: Path,
) -> Dict[Tuple[str, int, str], List[str]]:
    """Index by (user_id, timestamp, item_token) -> list of summaries."""
    index: Dict[Tuple[str, int, str], List[str]] = defaultdict(list)
    with review_store_path.open("r", encoding="utf-8") as file:
        for line in tqdm(file, desc="load review_store", unit="row", mininterval=1.0):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            summary = rec.get("summary") or ""
            if not isinstance(summary, str) or not summary.strip():
                continue
            key = (
                str(rec["user_id"]),
                int(rec["timestamp"]),
                str(rec["item_token"]),
            )
            index[key].append(summary.strip())
    print(f"[reviewg] indexed {len(index)} (user, ts, item) keys")
    return index


def build_reviewg(args: argparse.Namespace) -> dict:
    """One sample per pretrain window: predict review of the LAST interaction."""
    id_desc_path = args.output_dir / "id_descriptions.json"
    review_store_path = args.output_dir / "review_store.jsonl"
    train_src = args.output_dir / "pretrain_train.jsonl"
    test_src = args.output_dir / "pretrain_test.jsonl"
    for path in (id_desc_path, review_store_path, train_src, test_src):
        if not path.is_file():
            raise FileNotFoundError(f"missing {path}; run --stages pretrain first")

    with id_desc_path.open("r", encoding="utf-8") as file:
        id_descriptions = json.load(file)
    item_desc = id_descriptions.get("item", {})
    review_index = load_review_index(review_store_path)

    counts = {}
    for split, src in (("train", train_src), ("test", test_src)):
        dst = args.output_dir / f"reviewg_{split}.jsonl"
        rng = random.Random(args.seed + (10 if split == "train" else 11))
        total = count_lines(src)
        written = 0
        missed = 0
        with src.open("r", encoding="utf-8") as fin, dst.open(
            "w", encoding="utf-8"
        ) as fout:
            for line in tqdm(
                fin,
                total=total,
                desc=f"reviewg_{split}",
                unit="line",
                mininterval=1.0,
            ):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                interactions = parse_interactions(rec["text"])
                timestamps = rec.get("timestamps") or []
                if len(interactions) < 2 or len(timestamps) != len(interactions):
                    missed += 1
                    continue

                item, category, rating = interactions[-1]
                ts = int(timestamps[-1])
                key = (str(rec["user_id"]), ts, item)
                summaries = review_index.get(key)
                if not summaries:
                    missed += 1
                    continue

                history = interactions[:-1]
                stars = rating_token_to_stars(rating)
                item_name = item_desc.get(item, item)
                instruction = rng.choice(REVIEWG_INSTRUCTIONS).format(
                    item_name=item_name, rating=stars
                )
                sample = {
                    "task": "reviewg",
                    "split": split,
                    "user_id": str(rec["user_id"]),
                    "instruction": instruction,
                    "input": interactions_to_text(history),
                    "output": summaries[0],
                    "metadata": {
                        "history_length": len(history),
                        "target_item": item,
                        "target_item_name": item_name,
                        "target_category": category,
                        "target_rating": rating,
                        "timestamp": ts,
                    },
                }
                fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                written += 1

        counts[split] = {"written": written, "missed": missed}
        print(
            f"[reviewg] {split} -> {dst} "
            f"(written={written}, missed={missed})"
        )
    return counts


def update_stats(args: argparse.Namespace, patch: dict) -> None:
    stats_path = args.output_dir / "stats.json"
    stats = {"dataset": "amazon_movies_tv", "tasks": {}, "total": {}}
    if stats_path.exists():
        try:
            with stats_path.open("r", encoding="utf-8") as file:
                stats = json.load(file)
        except json.JSONDecodeError:
            pass
    stats.setdefault("dataset", "amazon_movies_tv")
    stats.setdefault("tasks", {})
    stats["tasks"].update(patch)
    stats["total"] = {
        "train": sum(
            (
                v.get("train", 0)
                if isinstance(v, dict) and "train" in v
                else v.get("written", 0)
                if isinstance(v, dict) and "written" in v
                else 0
            )
            for v in stats["tasks"].values()
            if isinstance(v, dict)
        ),
        "test": sum(
            (
                v.get("test", 0)
                if isinstance(v, dict) and "test" in v
                else 0
            )
            for v in stats["tasks"].values()
            if isinstance(v, dict)
        ),
    }
    # Prefer explicit train/test fields when present.
    train_sum = 0
    test_sum = 0
    for value in stats["tasks"].values():
        if not isinstance(value, dict):
            continue
        if "train" in value:
            train_sum += int(value["train"])
        if "test" in value:
            test_sum += int(value["test"])
    if train_sum or test_sum:
        stats["total"] = {"train": train_sum, "test": test_sum}
    write_json(stats_path, stats)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reviews",
        type=Path,
        default=AMAZON_DIR / "reviews_Movies_and_TV.json.gz",
        help="Amazon reviews jsonl (.gz)，默认 ../amazon2014/reviews_Movies_and_TV.json.gz",
    )
    parser.add_argument(
        "--meta",
        type=Path,
        default=AMAZON_DIR / "meta_Movies_and_TV.json.gz",
        help="Amazon meta jsonl (.gz)，默认 ../amazon2014/meta_Movies_and_TV.json.gz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="统一输出目录（默认 ../amazon_movies_tv）",
    )
    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument("--min-interactions", type=int, default=52)
    parser.add_argument("--input-ratio", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["pretrain", "warmup", "favcategory", "reviewg", "all"],
        default=["all"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stages = set(args.stages)
    if "all" in stages:
        stages = {"pretrain", "warmup", "favcategory", "reviewg"}

    if "pretrain" in stages:
        print("=" * 72)
        print("Stage: pretrain + id_descriptions + review_store")
        print("=" * 72)
        build_pretrain(args)

    if "warmup" in stages:
        print("=" * 72)
        print("Stage: warmup_train / warmup_eval")
        print("=" * 72)
        build_warmup_train(args)
        build_warmup_eval(args)

    if "favcategory" in stages:
        print("=" * 72)
        print("Stage: favcategory")
        print("=" * 72)
        counts = build_favcategory(args)
        update_stats(
            args,
            {
                "favcategory": {
                    "train": counts.get("train", 0),
                    "test": counts.get("test", 0),
                }
            },
        )

    if "reviewg" in stages:
        print("=" * 72)
        print("Stage: reviewg")
        print("=" * 72)
        counts = build_reviewg(args)
        update_stats(
            args,
            {
                "reviewg": {
                    "mode": "last_interaction_exact_item_match",
                    "train": counts.get("train", {}).get("written", 0),
                    "test": counts.get("test", {}).get("written", 0),
                    "train_missed": counts.get("train", {}).get("missed", 0),
                    "test_missed": counts.get("test", {}).get("missed", 0),
                }
            },
        )

    print("Done.")


if __name__ == "__main__":
    main()
