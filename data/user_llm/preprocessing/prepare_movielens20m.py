#!/usr/bin/env python3
"""从 MovieLens-20M 原始 CSV 构造 USER-LLM 风格数据。

默认读取 ``../ml-20m/movies.csv`` / ``ratings.csv``，全部输出到
``../movielens20m/``（即 ``seqllm/data/user_llm/movielens20m/``）：

- ``id_descriptions.json``
- ``pretrain_train.jsonl`` / ``pretrain_test.jsonl``（中间产物）
- ``warmup_train.jsonl`` / ``warmup_eval.jsonl`` / ``warmup_stats.json``
- ``favgenre_train.jsonl`` / ``favgenre_test.jsonl``
- ``stats.json``（favgenre 统计；若已有其它 task 统计会合并保留）

处理逻辑对齐现有产物：
- 用户内按时间排序，保留交互数 >= ``seq_len + 2`` 的用户
- 长度 ``seq_len`` 的滑动窗口：最后一个窗口进 test，其余进 train
- 每个交互编码为 ``<movie_X> <genre_Y> <ml_rating_Z>``
- warmup_train：窗口按交互数 30% / 70% 切分
- warmup_eval：窗口除最后 1 个交互外作为 input，最后 1 个作为 output
- favgenre：窗口内出现次数最多的 genre token 作为标签

示例：
    python3 prepare_movielens20m.py
    python3 prepare_movielens20m.py --stages pretrain warmup
    python3 prepare_movielens20m.py --stages favgenre
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover

    def tqdm(iterable=None, **kwargs):  # type: ignore[no-redef]
        return iterable if iterable is not None else iter(())


SCRIPT_DIR = Path(__file__).resolve().parent
USERLLM_DIR = SCRIPT_DIR.parent  # .../seqllm/data/user_llm
ML20M_DIR = USERLLM_DIR / "ml-20m"
# 输出与脚本同级：seqllm/data/user_llm/movielens20m/
# （不要写到 workspace 根下的 data/user_llm/）
DEFAULT_OUTPUT_DIR = USERLLM_DIR / "movielens20m"

TOKENS_PER_INTERACTION = 3
WARMUP_INSTRUCTION = (
    "Based on the user's viewing history, predict the user's future viewing behavior."
)
FAVGENRE_INSTRUCTIONS = [
    "What is the user's favorite genre?",
    "What genre does the user prefer most?",
    "Based on the user's history, what is their favorite genre?",
    "Identify the user's most preferred genre.",
    "What type of content does this user like best?",
]

# MovieLens-20M half-star ratings -> token id
RATING_VALUES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def normalize_rating(raw_rating: str) -> str:
    value = float(raw_rating)
    if value.is_integer():
        return str(int(value))
    return f"{value:g}"


def rating_to_token(rating: str) -> str:
    value = float(rating)
    idx = int(round(value * 2)) - 1  # 0.5 -> 0, ..., 5.0 -> 9
    if not 0 <= idx < len(RATING_VALUES):
        raise ValueError(f"unexpected rating: {rating}")
    return f"<ml_rating_{idx}>"


def rating_description(rating: str) -> str:
    return f"The user's rating for this movie is {rating}"


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


def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                yield json.loads(line)


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
        f"{item} {genre} {rating}" for item, genre, rating in interactions
    )


# ---------------------------------------------------------------------------
# Raw MovieLens loading
# ---------------------------------------------------------------------------

def load_movies(movies_csv: Path) -> Dict[int, Dict[str, str]]:
    movies: Dict[int, Dict[str, str]] = {}
    with movies_csv.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            movie_id = int(row["movieId"])
            movies[movie_id] = {
                "title": row["title"],
                "genres": row["genres"],
            }
    return movies


def build_genre_id_map(movies: Dict[int, Dict[str, str]]) -> Dict[str, int]:
    genre_id_map: Dict[str, int] = {}
    for movie_id in sorted(movies):
        genre = movies[movie_id]["genres"]
        if genre not in genre_id_map:
            genre_id_map[genre] = len(genre_id_map)
    return genre_id_map


def build_id_descriptions(
    movies: Dict[int, Dict[str, str]],
    genre_id_map: Dict[str, int],
) -> dict:
    rating_section = {}
    for idx, value in enumerate(RATING_VALUES):
        rating = normalize_rating(str(value))
        rating_section[f"<ml_rating_{idx}>"] = rating_description(rating)

    return {
        "movie": {
            f"<movie_{movie_id}>": movies[movie_id]["title"]
            for movie_id in sorted(movies)
        },
        "genre": {
            f"<genre_{gid}>": genre for genre, gid in genre_id_map.items()
        },
        "rating": rating_section,
    }


def render_interaction(
    movie_id: int,
    rating: str,
    movies: Dict[int, Dict[str, str]],
    genre_id_map: Dict[str, int],
) -> str:
    genre_id = genre_id_map[movies[movie_id]["genres"]]
    return (
        f"<movie_{movie_id}> "
        f"<genre_{genre_id}> "
        f"{rating_to_token(rating)}"
    )


def iter_user_rows(
    ratings_csv: Path,
    valid_movies: set,
) -> Iterator[Tuple[int, List[Tuple[int, int, str]]]]:
    """Yield (user_id, [(timestamp, movie_id, rating_str), ...])."""
    current_user: Optional[int] = None
    current_rows: List[Tuple[int, int, str]] = []

    with ratings_csv.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in tqdm(
            reader,
            total=20_000_263,
            desc="scan ratings",
            unit="row",
            mininterval=1.0,
        ):
            user_id = int(row["userId"])
            movie_id = int(row["movieId"])
            if movie_id not in valid_movies:
                continue
            rating = normalize_rating(row["rating"])
            timestamp = int(row["timestamp"])

            if current_user is None:
                current_user = user_id
            if user_id != current_user:
                current_rows.sort(key=lambda item: (item[0], item[1]))
                yield current_user, current_rows
                current_user = user_id
                current_rows = []
            current_rows.append((timestamp, movie_id, rating))

    if current_user is not None and current_rows:
        current_rows.sort(key=lambda item: (item[0], item[1]))
        yield current_user, current_rows


# ---------------------------------------------------------------------------
# Stage 1: pretrain + id_descriptions
# ---------------------------------------------------------------------------

def build_pretrain(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    movies = load_movies(args.movies_csv)
    genre_id_map = build_genre_id_map(movies)
    id_desc = build_id_descriptions(movies, genre_id_map)

    id_desc_path = args.output_dir / "id_descriptions.json"
    write_json(id_desc_path, id_desc)
    print(f"[pretrain] wrote {id_desc_path}")

    train_path = args.output_dir / "pretrain_train.jsonl"
    test_path = args.output_dir / "pretrain_test.jsonl"
    valid_movies = set(movies.keys())

    kept_users = 0
    train_windows = 0
    test_windows = 0

    with train_path.open("w", encoding="utf-8") as train_f, test_path.open(
        "w", encoding="utf-8"
    ) as test_f:
        user_bar = tqdm(
            desc="users",
            total=138_493,
            unit="user",
            mininterval=1.0,
        )
        for user_id, rows in iter_user_rows(args.ratings_csv, valid_movies):
            user_bar.update(1)
            if len(rows) < args.min_interactions:
                continue
            kept_users += 1
            user_bar.set_postfix(
                kept=kept_users,
                train=train_windows,
                refresh=False,
            )

            tokens = [
                render_interaction(
                    movie_id, rating, movies, genre_id_map
                )
                for _, movie_id, rating in rows
            ]
            num_windows = len(tokens) - args.seq_len + 1
            last_window_idx = num_windows - 1
            for start in range(num_windows):
                window = tokens[start : start + args.seq_len]
                record = {
                    "user_id": user_id,
                    "text": " ".join(window),
                }
                line = json.dumps(record, ensure_ascii=False) + "\n"
                if start == last_window_idx:
                    test_f.write(line)
                    test_windows += 1
                else:
                    train_f.write(line)
                    train_windows += 1
        user_bar.close()

    stats = {
        "dataset": "MovieLens20M",
        "seq_len": args.seq_len,
        "min_interactions": args.min_interactions,
        "kept_users": kept_users,
        "train_windows": train_windows,
        "test_windows": test_windows,
        "num_movies": len(movies),
        "num_genres": len(genre_id_map),
        "num_ratings": len(RATING_VALUES),
        "outputs": {
            "pretrain_train": str(train_path),
            "pretrain_test": str(test_path),
            "id_descriptions": str(id_desc_path),
        },
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
        raise FileNotFoundError(
            f"missing {src}; please run --stages pretrain first"
        )

    total_lines = count_lines(src)
    n_seen = 0
    n_in = 0
    n_out = 0
    interactions_distribution: Dict[int, int] = {}

    with src.open("r", encoding="utf-8") as fin, dst.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in tqdm(
            fin,
            total=total_lines,
            desc="warmup_train",
            unit="line",
            mininterval=1.0,
        ):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            tokens = rec["text"].split()
            input_tokens, output_tokens, _ = split_window_by_ratio(
                tokens, args.input_ratio
            )
            n_total = len(tokens) // TOKENS_PER_INTERACTION
            interactions_distribution[n_total] = (
                interactions_distribution.get(n_total, 0) + 1
            )
            out_rec = {
                "instruction": WARMUP_INSTRUCTION,
                "input": " ".join(input_tokens),
                "output": " ".join(output_tokens),
                "user_id": rec["user_id"],
            }
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
        "interactions_per_window_histogram": dict(
            sorted(interactions_distribution.items())
        ),
    }
    write_json(args.output_dir / "warmup_stats.json", summary)
    print(f"[warmup] train -> {dst} ({n_seen} records)")
    return summary


def build_warmup_eval(args: argparse.Namespace) -> dict:
    src = args.output_dir / "pretrain_test.jsonl"
    dst = args.output_dir / "warmup_eval.jsonl"
    if not src.is_file():
        raise FileNotFoundError(
            f"missing {src}; please run --stages pretrain first"
        )

    total_lines = count_lines(src)
    n_seen = 0
    skipped = 0

    with src.open("r", encoding="utf-8") as fin, dst.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in tqdm(
            fin,
            total=total_lines,
            desc="warmup_eval",
            unit="line",
            mininterval=1.0,
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
            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            n_seen += 1

    summary = {
        "source": str(src),
        "dest": str(dst),
        "num_records": n_seen,
        "skipped": skipped,
        "instruction": WARMUP_INSTRUCTION,
    }
    print(f"[warmup] eval  -> {dst} ({n_seen} records, skipped={skipped})")
    return summary


# ---------------------------------------------------------------------------
# Stage 3: favgenre
# ---------------------------------------------------------------------------

def most_frequent_genre(interactions: List[Tuple[str, str, str]]) -> str:
    counter = Counter(genre for _, genre, _ in interactions)
    return counter.most_common(1)[0][0]


def generate_favgenre_file(
    src: Path,
    dst: Path,
    id_descriptions: dict,
    split: str,
    seed: int,
) -> int:
    rng = random.Random(seed + (0 if split == "train" else 1))
    genre_desc = id_descriptions.get("genre", {})
    total = count_lines(src)
    written = 0

    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8") as fin, dst.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in tqdm(
            fin,
            total=total,
            desc=f"favgenre_{split}",
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
            fav = most_frequent_genre(interactions)
            sample = {
                "task": "favgenre",
                "split": split,
                "user_id": str(rec["user_id"]),
                "instruction": rng.choice(FAVGENRE_INSTRUCTIONS),
                "input": interactions_to_text(interactions),
                "output": fav,
                "metadata": {
                    "history_length": len(interactions),
                    "category_description": genre_desc.get(fav, fav),
                },
            }
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
            written += 1

    print(f"[favgenre] {split} -> {dst} ({written} records)")
    return written


def build_favgenre(args: argparse.Namespace) -> dict:
    id_desc_path = args.output_dir / "id_descriptions.json"
    train_src = args.output_dir / "pretrain_train.jsonl"
    test_src = args.output_dir / "pretrain_test.jsonl"
    for path in (id_desc_path, train_src, test_src):
        if not path.is_file():
            raise FileNotFoundError(
                f"missing {path}; please run --stages pretrain first"
            )

    with id_desc_path.open("r", encoding="utf-8") as file:
        id_descriptions = json.load(file)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    n_train = generate_favgenre_file(
        train_src,
        args.output_dir / "favgenre_train.jsonl",
        id_descriptions,
        "train",
        args.seed,
    )
    n_test = generate_favgenre_file(
        test_src,
        args.output_dir / "favgenre_test.jsonl",
        id_descriptions,
        "test",
        args.seed,
    )

    stats_path = args.output_dir / "stats.json"
    stats = {"dataset": "movielens20m", "tasks": {}, "total": {}}
    if stats_path.exists():
        try:
            with stats_path.open("r", encoding="utf-8") as file:
                stats = json.load(file)
        except json.JSONDecodeError:
            pass

    stats.setdefault("dataset", "movielens20m")
    stats.setdefault("tasks", {})
    stats["tasks"]["favgenre"] = {"train": n_train, "test": n_test}
    stats["total"] = {
        "train": sum(v.get("train", 0) for v in stats["tasks"].values()),
        "test": sum(v.get("test", 0) for v in stats["tasks"].values()),
    }
    write_json(stats_path, stats)
    print(f"[favgenre] stats -> {stats_path}")
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--movies-csv",
        type=Path,
        default=ML20M_DIR / "movies.csv",
    )
    parser.add_argument(
        "--ratings-csv",
        type=Path,
        default=ML20M_DIR / "ratings.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="统一输出目录（id_descriptions / pretrain / warmup / favgenre）",
    )
    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument(
        "--min-interactions",
        type=int,
        default=52,
        help="Keep users with at least this many interactions "
        "(default: seq_len + 2).",
    )
    parser.add_argument(
        "--input-ratio",
        type=float,
        default=0.3,
        help="Warmup train: fraction of interactions used as input.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["pretrain", "warmup", "favgenre", "all"],
        default=["all"],
        help="Which stages to run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stages = set(args.stages)
    if "all" in stages:
        stages = {"pretrain", "warmup", "favgenre"}

    if args.min_interactions < args.seq_len + 2:
        print(
            f"[warn] min_interactions={args.min_interactions} < "
            f"seq_len+2={args.seq_len + 2}; some users may lack a "
            "train window."
        )

    if "pretrain" in stages:
        print("=" * 72)
        print("Stage: pretrain + id_descriptions")
        print("=" * 72)
        build_pretrain(args)

    if "warmup" in stages:
        print("=" * 72)
        print("Stage: warmup_train / warmup_eval")
        print("=" * 72)
        build_warmup_train(args)
        build_warmup_eval(args)

    if "favgenre" in stages:
        print("=" * 72)
        print("Stage: favgenre")
        print("=" * 72)
        build_favgenre(args)

    print("Done.")


if __name__ == "__main__":
    main()
