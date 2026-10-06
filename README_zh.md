# SeqLLM

[![English](https://img.shields.io/badge/English-blue)](README.md) [![简体中文](https://img.shields.io/badge/%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-blue)](README_zh.md)

基于 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 的序列推荐 / 用户行为建模训练与评测代码库。

本仓库在通用 LLM 微调框架之上，补充了两类推荐场景的完整流水线：

1. **User-LLM 风格任务**：MovieLens-20M、Amazon Movies & TV
  将交互编码为特殊 token（item / category / rating），支持 embedding warmup、projector 对齐与下游 SFT。
2. **OpenOneRec / SID 任务**：基于 [OpenOneRec](https://huggingface.co/datasets/OpenOneRec/OpenOneRec-RecIF) 的语义 ID（SID）预训练与 SFT，覆盖 video-rec、item understanding 等。

---

## 目录

- [环境要求](#环境要求)
- [仓库结构](#仓库结构)
- [数据准备](#数据准备)
- [扩展词表模型](#扩展词表模型)
- [训练](#训练)
- [评测](#评测)
- [配置说明](#配置说明)
- [致谢](#致谢)
- [License](#license)

---



## 环境要求


| 项目            | 建议版本                                    |
| ------------- | --------------------------------------- |
| Python        | ≥ 3.10                                  |
| CUDA          | 与本机 PyTorch / vLLM 匹配                   |
| GPU           | 训练默认按单机 8 卡编写，可改 `CUDA_VISIBLE_DEVICES` |
| LLaMA-Factory | 本仓库内置（`pip install -e .`）               |
| vLLM          | 0.11.2（推理 / video-rec 评测）               |


---

---



## 仓库结构

```text
seqllm/
├── src/llamafactory/          # 训练框架（基于 LLaMA-Factory，含 seq_align / SID 等扩展）
├── data/
│   ├── dataset_info.json      # LLaMA-Factory 数据集注册
│   ├── user_llm/
│   │   ├── ml-20m/            # 【需自行下载】MovieLens-20M 原始数据
│   │   ├── amazon2014/        # 【需自行下载】Amazon Movies & TV 原始数据
│   │   └── preprocessing/     # MovieLens / Amazon 预处理
│   └── openonerec/
│       ├── RecIF/             # 【需自行下载】OpenOneRec-RecIF
│       ├── general_data/      # 【需自行下载】通用 SFT / 预训练语料
│       └── preprocessing/     # RecIF → pretrain / SFT JSONL
├── scripts/
│   ├── run.sh                 # 单机多卡训练入口
│   ├── build_qwen3_8b_token_variants.py
│   ├── eval_video_rec.py
│   └── ...
├── examples/
│   ├── userllm/               # User-LLM 训练 / 评测 YAML
│   └── onerec/                # OpenOneRec 三阶段训练 YAML
└── evaluation/                # 语言能力 / video-rec / item-understanding 等评测脚本
```

> 原始大规模数据、模型权重、训练产物默认不入库。请按下文自行下载，并在 YAML 中填写本机路径。

---



## 数据准备



### 1. User-LLM：MovieLens-20M & Amazon Movies & TV

**下载**


| 数据集                | 来源                                                                                                                           | 建议放置位置                      |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------- | --------------------------- |
| MovieLens-20M      | [https://grouplens.org/datasets/movielens/20m/](https://grouplens.org/datasets/movielens/20m/)                               | `data/user_llm/ml-20m/`     |
| Amazon Movies & TV | [https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html) | `data/user_llm/amazon2014/` |


**预处理**

```bash
cd data/user_llm/preprocessing

bash prepare_userllm.sh --datasets both --stages all

# 也可分开跑：
# bash prepare_userllm.sh --datasets movielens --stages all
# bash prepare_userllm.sh --datasets amazon --stages all
```

主要输出：


| 数据集       | 输出目录                              | 关键文件                                                                                                  |
| --------- | --------------------------------- | ----------------------------------------------------------------------------------------------------- |
| MovieLens | `data/user_llm/movielens20m/`     | `id_descriptions.json`、`warmup_*.jsonl`、`favgenre_*.jsonl`                                            |
| Amazon    | `data/user_llm/amazon_movies_tv/` | `id_descriptions.json`、`special_tokens.json`、`warmup_*.jsonl`、`favcategory_*.jsonl`、`reviewg_*.jsonl` |


交互默认编码为三元组 token，例如 Amazon：

```text
<item_i> <category_j> <az_rating_k>
```



### 2. OpenOneRec

> **本仓库不包含原始大规模数据。**
> `data/openonerec/RecIF` 与 `data/openonerec/general_data` 均需自行从 Hugging Face 下载后放入对应目录；仅提供 `preprocessing/` 脚本与处理后的路径约定。

**原始数据下载（必做）**

| 本地目录 | 来源 | 说明 |
| -------- | ---- | ---- |
| `data/openonerec/RecIF/` | [OpenOneRec-RecIF](https://huggingface.co/datasets/OpenOneRec/OpenOneRec-RecIF) | 推荐序列 / SID / RecIF-Bench 原始数据；预处理脚本默认从此目录读取 |
| `data/openonerec/general_data/` | [OpenOneRec-General-SFT](https://huggingface.co/datasets/OpenOneRec/OpenOneRec-General-SFT)（可选：[OpenOneRec-General-Pretrain](https://huggingface.co/datasets/OpenOneRec/OpenOneRec-General-Pretrain)） | 通用域语料；需自行下载并整理为训练可用的 JSONL / 注册到 `dataset_info.json` |

示例（Hugging Face CLI）：

```bash
# RecIF → data/openonerec/RecIF
huggingface-cli download OpenOneRec/OpenOneRec-RecIF \
  --repo-type dataset \
  --local-dir data/openonerec/RecIF

# 通用数据 → data/openonerec/general_data
huggingface-cli download OpenOneRec/OpenOneRec-General-SFT \
  --repo-type dataset \
  --local-dir data/openonerec/general_data
```

**语言 SFT 子集（基于 `general_data`）**

1. 将 General-SFT 下载到 `data/openonerec/general_data/`（见上表）
2. 按需采样（例如 10k / 100k）写成 JSONL（例如 `data/openonerec/data/OpenOneRec_SFT_100k.jsonl`）
3. 在 `data/dataset_info.json` 中注册路径（仓库内示例键名：`general_data`）

**序列 / 语义数据（基于 `RecIF`）**

```bash
# 确认已下载到 data/openonerec/RecIF 后再跑预处理

cd data/openonerec/preprocessing

# 1) 提取预训练 parquet 并转为 JSONL
bash prepare_pretrain.sh
# 输出：data/openonerec/data/pretrain_jsonl/
#   pretrain_video_rec.jsonl
#   pretrain_item_understand.jsonl

# 2) 构造 SFT JSONL
bash prepare_sft.sh
# 输出：data/openonerec/data/sft_jsonl/
#   sft_sid_caption.jsonl
#   sft_video_rec.jsonl
#   sft_video_rec_expand.jsonl
```

然后在 `data/dataset_info.json` 中确认 `file_name` 指向上述 JSONL（以及 `general_data` 对应文件）。

---



## 扩展词表模型

推荐场景会向基座模型（默认 Qwen3-8B）注入大量特殊 token。使用：

```bash
python scripts/build_qwen3_8b_token_variants.py \
  --base-model /path/to/Qwen3-8B \
  --output-root /path/to/output_models \
  --targets sid movielens amazon \
  --semantic-device cuda
```


| `--targets` | 含义                                     | 初始化方式                            |
| ----------- | -------------------------------------- | -------------------------------- |
| `sid`       | OpenOneRec SID token                   | 随机初始化                            |
| `movielens` | MovieLens movie / genre / rating token | 由 `id_descriptions.json` 语义均值初始化 |
| `amazon`    | Amazon item / category / rating token  | 同上                               |


常用参数：

```bash
--movielens-json data/user_llm/movielens20m/id_descriptions.json
--amazon-json    data/user_llm/amazon_movies_tv/id_descriptions.json
--sid-tokens     /path/to/sid_added_tokens.json
--overwrite      # 允许覆盖已有输出目录
```

生成后的模型目录请填入后续训练 YAML 的 `model_name_or_path`。

---



## 训练

统一入口（默认 8 卡）：

```bash
bash scripts/run.sh
```



### User-LLM

建议按阶段串联（先改 YAML 里的模型 / 数据 / 输出路径）：


| 阶段                 | 配置                                        | 说明           |
| ------------------ | ----------------------------------------- | ------------ |
| Embedding warmup   | `examples/userllm/prefix_guided_sft.yaml` | 暖启新增行为 token |
| Projector + LLM 对齐 | `examples/userllm/user_align.yaml`        | 用户表征与 LLM 对齐 |
| Downstream SFT     | `examples/userllm/downstream.yaml`        | 下游任务微调       |


```bash
YAML=examples/userllm/prefix_guided_sft.yaml \
LOG_FILE=log/userllm_warmup.log bash scripts/run.sh

YAML=examples/userllm/user_align.yaml \
LOG_FILE=log/userllm_align.log bash scripts/run.sh

YAML=examples/userllm/downstream.yaml \
LOG_FILE=log/userllm_downstream.log bash scripts/run.sh
```



### OpenOneRec

固定三阶段顺序：

```bash
# Stage 1：仅更新新增 SID token 的 embedding / lm_head
YAML=examples/onerec/warmup.yaml \
LOG_FILE=log/onerec_warmup.log bash scripts/run.sh

# Stage 2：联合训练 projector 与 LLM backbone
YAML=examples/onerec/projector.yaml \
LOG_FILE=log/onerec_projector.log bash scripts/run.sh

# Stage 3：prefix-guided SFT
YAML=examples/onerec/prefix_guided_sft.yaml \
LOG_FILE=log/onerec_sft.log bash scripts/run.sh
```

各阶段默认通过 `output_dir` → 下一阶段 `model_name_or_path` 串联。若从中间 checkpoint 续训，请同步修改后续 YAML。

---



## 评测



### 语言能力

```bash
bash evaluation/eval_lang/eval_mmlu.sh
bash evaluation/eval_lang/eval_ceval.sh
bash evaluation/eval_lang/eval_agieval.sh
```



### User-LLM 下游

```bash
# 按评测 YAML 启动（先改模型路径）
YAML=examples/userllm/downstream_eval.yaml \
LOG_FILE=log/userllm_eval.log bash scripts/run.sh
```



### OpenOneRec：Video-Rec

```bash
# 默认读取 evaluation/eval_recif_video_rec/eval_video_rec.yaml
bash evaluation/eval_recif_video_rec/run_eval_video_rec.sh

# 覆盖模型路径
MODEL_PATH=/path/to/ckpt \
bash evaluation/eval_recif_video_rec/run_eval_video_rec.sh
```



### OpenOneRec：Item Understanding

```bash
cd evaluation/eval_item_understanding_task
bash run_eval.sh /path/to/ckpt 8
# 第三个参数 think 可开启思考模式
# bash run_eval.sh /path/to/ckpt 8 think
```

---



## 配置说明

开源发布前，请务必检查并改写 YAML / shell 中的本机路径，至少包括：


| 字段 / 变量               | 含义                             |
| --------------------- | ------------------------------ |
| `model_name_or_path`  | 基座或上一阶段 checkpoint             |
| `dataset_dir`         | 指向含 `dataset_info.json` 的数据根目录 |
| `output_dir`          | 训练产物目录                         |
| `HF_DATASETS_CACHE`   | HuggingFace datasets 可写缓存      |
| `original_vocab_size` | 扩词前词表大小；与冻结 embedding 逻辑相关     |


`data/dataset_info.json` 示例：

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

---



## 致谢

- 训练框架：[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- 基座模型：[Qwen3](https://github.com/QwenLM/Qwen3)
- 推荐数据与基准：[OpenOneRec](https://huggingface.co/OpenOneRec)、[MovieLens](https://grouplens.org/datasets/movielens/)、[Amazon Review Data (McAuley)](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html)
- 相关工作：User-LLM（[arXiv:2402.13598](https://arxiv.org/abs/2402.13598)）

---



## License

本仓库主体遵循 [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)（与 LLaMA-Factory 一致）。
第三方数据、模型权重请遵守其各自的许可证与使用条款。
