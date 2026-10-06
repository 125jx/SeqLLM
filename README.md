# SeqLLM

[![English](https://img.shields.io/badge/English-blue)](README.md) [![简体中文](https://img.shields.io/badge/%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-blue)](README_zh.md)

A codebase for sequential recommendation and user behavior modeling, built on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).

This repository extends a general LLM fine-tuning framework with two end-to-end recommendation workflows:

1. **User-LLM-style tasks** on MovieLens-20M and Amazon Movies & TV. Interactions are encoded as special tokens (item, category, and rating), with support for embedding warmup, projector alignment, and downstream supervised fine-tuning (SFT).
2. **OpenOneRec / SID tasks** using semantic IDs (SIDs) from [OpenOneRec](https://huggingface.co/datasets/OpenOneRec/OpenOneRec-RecIF). The workflow covers pretraining and SFT for video recommendation, item understanding, and related tasks.

## Contents

- [Requirements](#requirements)
- [Repository layout](#repository-layout)
- [Data preparation](#data-preparation)
- [Models with extended vocabularies](#models-with-extended-vocabularies)
- [Training](#training)
- [Evaluation](#evaluation)
- [Configuration notes](#configuration-notes)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Requirements

| Component | Recommended version or setup |
| --- | --- |
| Python | 3.10 or newer |
| CUDA | Compatible with your local PyTorch and vLLM installations |
| GPUs | Training scripts default to one machine with 8 GPUs; adjust `CUDA_VISIBLE_DEVICES` as needed |
| LLaMA-Factory | Included in this repository; install with `pip install -e .` |
| vLLM | 0.11.2 for inference and video recommendation evaluation |

## Repository layout

```text
seqllm/
├── src/llamafactory/          # Training framework based on LLaMA-Factory, with seq_align/SID extensions
├── data/
│   ├── dataset_info.json      # LLaMA-Factory dataset registrations
│   ├── user_llm/
│   │   ├── ml-20m/            # Download MovieLens-20M source data yourself
│   │   ├── amazon2014/        # Download Amazon Movies & TV source data yourself
│   │   └── preprocessing/     # MovieLens/Amazon preprocessing
│   └── openonerec/
│       ├── RecIF/             # Download OpenOneRec-RecIF yourself
│       ├── general_data/      # Download general SFT/pretraining data yourself
│       └── preprocessing/     # RecIF to pretraining/SFT JSONL
├── scripts/
│   ├── run.sh                 # Single-node, multi-GPU training entry point
│   ├── build_qwen3_8b_token_variants.py
│   ├── eval_video_rec.py
│   └── ...
├── examples/
│   ├── userllm/               # User-LLM training/evaluation YAML files
│   └── onerec/                # OpenOneRec three-stage training YAML files
└── evaluation/                # Language, video recommendation, and item-understanding evaluation
```

Large source datasets, model weights, and training outputs are not included by default. Download the data described below and set your local paths in the YAML files.

## Data preparation

### 1. User-LLM: MovieLens-20M and Amazon Movies & TV

**Download**

| Dataset | Source | Suggested location |
| --- | --- | --- |
| MovieLens-20M | [GroupLens](https://grouplens.org/datasets/movielens/20m/) | `data/user_llm/ml-20m/` |
| Amazon Movies & TV | [Amazon Review Data](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html) | `data/user_llm/amazon2014/` |

**Preprocess**

```bash
cd data/user_llm/preprocessing

bash prepare_userllm.sh --datasets both --stages all

# Or process each dataset separately:
# bash prepare_userllm.sh --datasets movielens --stages all
# bash prepare_userllm.sh --datasets amazon --stages all
```

Main outputs:

| Dataset | Output directory | Key files |
| --- | --- | --- |
| MovieLens | `data/user_llm/movielens20m/` | `id_descriptions.json`, `warmup_*.jsonl`, `favgenre_*.jsonl` |
| Amazon | `data/user_llm/amazon_movies_tv/` | `id_descriptions.json`, `special_tokens.json`, `warmup_*.jsonl`, `favcategory_*.jsonl`, `reviewg_*.jsonl` |

By default, each interaction is encoded as a token triplet. For example, an Amazon interaction looks like:

```text
<item_i> <category_j> <az_rating_k>
```

### 2. OpenOneRec

**The large source datasets are not included in this repository.** Download `data/openonerec/RecIF` and `data/openonerec/general_data` from Hugging Face and place them in the corresponding directories. The repository provides preprocessing scripts and paths for the processed data.

**Download source data (required)**

| Local directory | Source | Notes |
| --- | --- | --- |
| `data/openonerec/RecIF/` | [OpenOneRec-RecIF](https://huggingface.co/datasets/OpenOneRec/OpenOneRec-RecIF) | Source data for recommendation sequences, SIDs, and RecIF-Bench; preprocessing scripts read from this directory by default |
| `data/openonerec/general_data/` | [OpenOneRec-General-SFT](https://huggingface.co/datasets/OpenOneRec/OpenOneRec-General-SFT); optional: [OpenOneRec-General-Pretrain](https://huggingface.co/datasets/OpenOneRec/OpenOneRec-General-Pretrain) | Download and prepare general-domain data as training-ready JSONL, then register it in `dataset_info.json` |

Example with the Hugging Face CLI:

```bash
# RecIF to data/openonerec/RecIF
huggingface-cli download OpenOneRec/OpenOneRec-RecIF \
  --repo-type dataset \
  --local-dir data/openonerec/RecIF

# General data to data/openonerec/general_data
huggingface-cli download OpenOneRec/OpenOneRec-General-SFT \
  --repo-type dataset \
  --local-dir data/openonerec/general_data
```

**Language SFT subset (from `general_data`)**

1. Download General-SFT to `data/openonerec/general_data/` as shown above.
2. Sample as needed (for example, 10k or 100k records) and write a JSONL file such as `data/openonerec/data/OpenOneRec_SFT_100k.jsonl`.
3. Register the file in `data/dataset_info.json` (the example key in this repository is `general_data`).

**Sequence and semantic data (from `RecIF`)**

```bash
# Download to data/openonerec/RecIF before preprocessing.
cd data/openonerec/preprocessing

# 1. Extract pretraining Parquet files and convert them to JSONL.
bash prepare_pretrain.sh
# Output: data/openonerec/data/pretrain_jsonl/
#   pretrain_video_rec.jsonl
#   pretrain_item_understand.jsonl

# 2. Build SFT JSONL files.
bash prepare_sft.sh
# Output: data/openonerec/data/sft_jsonl/
#   sft_sid_caption.jsonl
#   sft_video_rec.jsonl
#   sft_video_rec_expand.jsonl
```

Then check that `file_name` entries in `data/dataset_info.json` point to these JSONL files and to the file prepared for `general_data`.

## Models with extended vocabularies

Recommendation tasks add many special tokens to the base model (Qwen3-8B by default). Build model variants with:

```bash
python scripts/build_qwen3_8b_token_variants.py \
  --base-model /path/to/Qwen3-8B \
  --output-root /path/to/output_models \
  --targets sid movielens amazon \
  --semantic-device cuda
```

| `--targets` value | Tokens | Initialization |
| --- | --- | --- |
| `sid` | OpenOneRec SID tokens | Random initialization |
| `movielens` | MovieLens movie, genre, and rating tokens | Mean semantic embeddings from `id_descriptions.json` |
| `amazon` | Amazon item, category, and rating tokens | Same method as above |

Common options:

```bash
--movielens-json data/user_llm/movielens20m/id_descriptions.json
--amazon-json    data/user_llm/amazon_movies_tv/id_descriptions.json
--sid-tokens     /path/to/sid_added_tokens.json
--overwrite      # Allow overwriting existing output directories
```

Set `model_name_or_path` in subsequent training YAML files to the generated model directory.

## Training

Shared entry point (8 GPUs by default):

```bash
bash scripts/run.sh
```

### User-LLM

Run the stages in order, after updating the model, data, and output paths in the YAML files:

| Stage | Configuration | Purpose |
| --- | --- | --- |
| Embedding warmup | `examples/userllm/prefix_guided_sft.yaml` | Warm up the new behavior tokens |
| Projector + LLM alignment | `examples/userllm/user_align.yaml` | Align user representations with the LLM |
| Downstream SFT | `examples/userllm/downstream.yaml` | Fine-tune for downstream tasks |

```bash
YAML=examples/userllm/prefix_guided_sft.yaml \
LOG_FILE=log/userllm_warmup.log bash scripts/run.sh

YAML=examples/userllm/user_align.yaml \
LOG_FILE=log/userllm_align.log bash scripts/run.sh

YAML=examples/userllm/downstream.yaml \
LOG_FILE=log/userllm_downstream.log bash scripts/run.sh
```

### OpenOneRec

Run these three stages in order:

```bash
# Stage 1: Update only the embeddings/lm_head for newly added SID tokens.
YAML=examples/onerec/warmup.yaml \
LOG_FILE=log/onerec_warmup.log bash scripts/run.sh

# Stage 2: Jointly train the projector and LLM backbone.
YAML=examples/onerec/projector.yaml \
LOG_FILE=log/onerec_projector.log bash scripts/run.sh

# Stage 3: Prefix-guided SFT.
YAML=examples/onerec/prefix_guided_sft.yaml \
LOG_FILE=log/onerec_sft.log bash scripts/run.sh
```

Each stage normally passes its `output_dir` to the next stage as `model_name_or_path`. If resuming from an intermediate checkpoint, update the later YAML files accordingly.

## Evaluation

### Language ability

```bash
bash evaluation/eval_lang/eval_mmlu.sh
bash evaluation/eval_lang/eval_ceval.sh
bash evaluation/eval_lang/eval_agieval.sh
```

### User-LLM downstream tasks

```bash
# Update the model path in the evaluation YAML first.
YAML=examples/userllm/downstream_eval.yaml \
LOG_FILE=log/userllm_eval.log bash scripts/run.sh
```

### OpenOneRec: video recommendation

```bash
# Reads evaluation/eval_recif_video_rec/eval_video_rec.yaml by default.
bash evaluation/eval_recif_video_rec/run_eval_video_rec.sh

# Override the model path.
MODEL_PATH=/path/to/ckpt \
bash evaluation/eval_recif_video_rec/run_eval_video_rec.sh
```

### OpenOneRec: item understanding

```bash
cd evaluation/eval_item_understanding_task
bash run_eval.sh /path/to/ckpt 8
# An optional third argument, think, enables thinking mode.
# bash run_eval.sh /path/to/ckpt 8 think
```

## Configuration notes

Before publishing or running the repository, review and update machine-specific paths in the YAML and shell files, including:

| Field or variable | Meaning |
| --- | --- |
| `model_name_or_path` | Base model or checkpoint from the previous stage |
| `dataset_dir` | Data root containing `dataset_info.json` |
| `output_dir` | Directory for training outputs |
| `HF_DATASETS_CACHE` | Writable Hugging Face datasets cache |
| `original_vocab_size` | Vocabulary size before adding tokens; used by embedding-freezing logic |

Example entry in `data/dataset_info.json`:

```json
{
  "sft_video_rec": {
    "file_name": "openonerec/data/sft_jsonl/sft_video_rec.jsonl",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  }
}
```

## Acknowledgments

- Training framework: [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- Base model: [Qwen3](https://github.com/QwenLM/Qwen3)
- Recommendation data and benchmarks: [OpenOneRec](https://huggingface.co/OpenOneRec), [MovieLens](https://grouplens.org/datasets/movielens/), and [Amazon Review Data (McAuley)](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html)
- Related work: User-LLM ([arXiv:2402.13598](https://arxiv.org/abs/2402.13598))

## License

The main repository follows the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0), consistent with LLaMA-Factory. Third-party datasets and model weights are subject to their respective licenses and terms of use.
