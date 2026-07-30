# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/language-modeling/run_clm.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional


@dataclass
class DataArguments:
    r"""Arguments pertaining to what data we are going to input our model for training and evaluation."""

    template: Optional[str] = field(
        default=None,
        metadata={"help": "Which template to use for constructing prompts in training and inference."},
    )
    dataset: Optional[str] = field(
        default=None,
        metadata={"help": "The name of dataset(s) to use for training. Use commas to separate multiple datasets."},
    )
    eval_dataset: Optional[str] = field(
        default=None,
        metadata={"help": "The name of dataset(s) to use for evaluation. Use commas to separate multiple datasets."},
    )
    dataset_dir: str = field(
        default="data",
        metadata={"help": "Path to the folder containing the datasets."},
    )
    media_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the folder containing the images, videos or audios. Defaults to `dataset_dir`."},
    )
    cutoff_len: int = field(
        default=2048,
        metadata={"help": "The cutoff length of the tokenized inputs in the dataset."},
    )
    train_on_prompt: bool = field(
        default=False,
        metadata={"help": "Whether or not to disable the mask on the prompt."},
    )
    loss_skip_first_n_lines: int = field(
        default=0,
        metadata={"help": "Skip first N lines in prompt for loss computation. Lines after will compute loss. 0 means no loss on prompt."},
    )
    mask_history: bool = field(
        default=False,
        metadata={"help": "Whether or not to mask the history and train on the last turn only."},
    )
    streaming: bool = field(
        default=False,
        metadata={"help": "Enable dataset streaming."},
    )
    buffer_size: int = field(
        default=16384,
        metadata={"help": "Size of the buffer to randomly sample examples from in dataset streaming."},
    )
    mix_strategy: Literal["concat", "interleave_under", "interleave_over"] = field(
        default="concat",
        metadata={"help": "Strategy to use in dataset mixing (concat/interleave) (undersampling/oversampling)."},
    )
    interleave_probs: Optional[str] = field(
        default=None,
        metadata={"help": "Probabilities to sample data from datasets. Use commas to separate multiple datasets."},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets."},
    )
    preprocessing_batch_size: int = field(
        default=1000,
        metadata={"help": "The number of examples in one group in pre-processing."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the pre-processing."},
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes, truncate the number of examples for each dataset."},
    )
    eval_num_beams: Optional[int] = field(
        default=None,
        metadata={"help": "Number of beams to use for evaluation. This argument will be passed to `model.generate`"},
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={"help": "Whether or not to ignore the tokens corresponding to the pad label in loss computation."},
    )
    val_size: float = field(
        default=0.0,
        metadata={"help": "Size of the validation set, should be an integer or a float in range `[0,1)`."},
    )
    eval_on_each_dataset: bool = field(
        default=False,
        metadata={"help": "Whether or not to evaluate on each dataset separately."},
    )
    packing: Optional[bool] = field(
        default=None,
        metadata={"help": "Enable sequences packing in training. Will automatically enable in pre-training."},
    )
    neat_packing: bool = field(
        default=False,
        metadata={"help": "Enable sequence packing without cross-attention."},
    )
    pt_end_token: Literal["eos", "pad"] = field(
        default="eos",
        metadata={
            "help": (
                "[PT & SFT stages] Which token to append to each raw sample as the end-of-sample "
                "marker, and whether to include it in loss. "
                "'eos' (default, original LF behavior): for PT append `tokenizer.eos_token` (for "
                "Qwen3 with template=qwen this is `<|im_end|>`) and INCLUDE it in loss; for SFT "
                "do nothing extra (the chat template's trailing `<|im_end|>\\n` is already in loss "
                "for the assistant turn). "
                "'pad': append `tokenizer.pad_token` (for Qwen3 this is `<|endoftext|>`) and "
                "EXCLUDE the trailing position from loss (`labels[-1]=IGNORE_INDEX`). For SFT it "
                "appends pad to the per-sample encoded ids in "
                "`SupervisedDatasetProcessor._encode_data_example` (best-effort: skipped when the "
                "body already fills `cutoff_len`, matching PT's truncation behaviour). When "
                "packing=True every per-sample trailing pad position is masked (NOT just the "
                "global trailing one)."
            )
        },
    )
    tool_format: Optional[str] = field(
        default=None,
        metadata={"help": "Tool format to use for constructing function calling examples."},
    )
    default_system: Optional[str] = field(
        default=None,
        metadata={"help": "Override the default system message in the template."},
    )
    enable_thinking: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether or not to enable thinking mode for reasoning models."},
    )
    tokenized_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Path to save or load the tokenized datasets. "
                "If tokenized_path not exists, it will save the tokenized datasets. "
                "If tokenized_path exists, it will load the tokenized datasets."
            )
        },
    )
    data_shared_file_system: bool = field(
        default=False,
        metadata={"help": "Whether or not to use a shared file system for the datasets."},
    )
    iceberg_beh_text_train_config: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Path to Iceberg behavior-text config YAML for streaming pretraining (train). "
                "If set, LLaMA-Factory will export it to ICEBERG_BEH_TEXT_TRAIN_CONFIG automatically."
            )
        },
    )
    iceberg_beh_text_eval_config: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Path to Iceberg behavior-text config YAML for streaming pretraining (eval). "
                "If set, LLaMA-Factory will export it to ICEBERG_BEH_TEXT_EVAL_CONFIG automatically."
            )
        },
    )
    
    # ========================================
    # SFT Input Mode: 控制输入中保留哪些内容（文本层面真正剔除）
    # ========================================
    sft_input_mode: Literal["all", "language_only", "sequence_only"] = field(
        default="all",
        metadata={
            "help": (
                "SFT input mode to control which content to keep in the input/prompt. "
                "Content is truly removed (not masked) at text level, reducing sequence length. "
                "'all': keep all content (default). "
                "'language_only': remove sequence tokens (<key:value> format like <时间:周三_18点>, <支付场景:付款码>) "
                "from input, keep language text (商户信息, 摘要, 投诉等). "
                "'sequence_only': only keep sequence tokens and section markers (<风险序列>, <近期序列>), "
                "remove other language text from input."
            )
        },
    )
    
    # ========================================
    # Per-user balanced loss (按用户平均 loss，影响梯度)
    # ========================================
    user_balanced_loss: bool = field(
        default=False,
        metadata={
            "help": (
                "[SFT stage, non-packing] If True, re-weight the per-token loss so that, for the "
                "datasets listed in `user_balanced_loss_datasets`, every unique user (identified by "
                "`user_id_key`) contributes equally to the gradient instead of every token. "
                "Each sample of user u gets a per-token weight w = scale / n_u, where n_u is the "
                "number of samples that user u has in that dataset; the batch loss becomes "
                "sum(w * ce) / sum(w). Datasets NOT in the list keep weight 1.0 (standard "
                "token-level loss). The trainer auto-detects the per-sample `loss_weight` column "
                "produced by the data pipeline and switches `model_accepts_loss_kwargs=False` so "
                "HF normalizes by gradient_accumulation_steps correctly."
            )
        },
    )
    user_balanced_loss_datasets: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Comma-separated dataset aliases (as in dataset_info.json, e.g. 'video_rec') that "
                "should use per-user balanced loss. Required when `user_balanced_loss=True`. "
                "Datasets not listed keep standard token-level loss (weight 1.0)."
            )
        },
    )
    user_id_key: str = field(
        default="uid",
        metadata={
            "help": (
                "Raw column name holding the user id in the listed datasets, used to count samples "
                "per user (n_u) when `user_balanced_loss=True`."
            )
        },
    )
    user_balanced_loss_normalize: Literal["mean1", "raw"] = field(
        default="mean1",
        metadata={
            "help": (
                "Scaling of the per-sample weight w = scale / n_u for user-balanced loss. "
                "'mean1' (default): scale = mean samples per user of that dataset (N/U), so the "
                "token-length-weighted mean weight is ~1.0 -> the dataset's overall gradient "
                "magnitude relative to other (weight-1) datasets is approximately preserved, only "
                "redistributed equally across users. 'raw': scale = 1.0 (w = 1/n_u), which shrinks "
                "the dataset's relative magnitude by ~mean(1/n_u) in mixed batches."
            )
        },
    )

    def __post_init__(self):
        def split_arg(arg):
            if isinstance(arg, str):
                return [item.strip() for item in arg.split(",")]
            return arg

        self.dataset = split_arg(self.dataset)
        self.eval_dataset = split_arg(self.eval_dataset)
        self.user_balanced_loss_datasets = split_arg(self.user_balanced_loss_datasets)

        if self.user_balanced_loss and not self.user_balanced_loss_datasets:
            raise ValueError(
                "`user_balanced_loss=True` requires `user_balanced_loss_datasets` to list at least "
                "one dataset alias (e.g. `user_balanced_loss_datasets: video_rec`)."
            )

        if self.user_balanced_loss and self.streaming:
            raise ValueError(
                "`user_balanced_loss=True` is incompatible with `streaming=True` (per-user sample "
                "counts n_u require a full non-streaming pass over the dataset)."
            )

        if self.media_dir is None:
            self.media_dir = self.dataset_dir

        if self.dataset is None and self.val_size > 1e-6:
            raise ValueError("Cannot specify `val_size` if `dataset` is None.")

        if self.eval_dataset is not None and self.val_size > 1e-6:
            raise ValueError("Cannot specify `val_size` if `eval_dataset` is not None.")

        if self.interleave_probs is not None:
            if self.mix_strategy == "concat":
                raise ValueError("`interleave_probs` is only valid for interleaved mixing.")

            self.interleave_probs = list(map(float, split_arg(self.interleave_probs)))
            if self.dataset is not None and len(self.dataset) != len(self.interleave_probs):
                raise ValueError("The length of dataset and interleave probs should be identical.")

            # 允许：训练多数据集按 interleave_probs 混合，但评估只用单一数据集（最常见诉求）。
            # 仅当 eval_dataset 本身配置为多个数据集时，才要求其长度与 interleave_probs 一致。
            if (
                self.eval_dataset is not None
                and len(self.eval_dataset) > 1
                and len(self.eval_dataset) != len(self.interleave_probs)
            ):
                raise ValueError("The length of eval dataset and interleave probs should be identical.")

        if self.streaming and self.val_size > 1e-6 and self.val_size < 1:
            raise ValueError("Streaming mode should have an integer val size.")

        if self.streaming and self.max_samples is not None:
            raise ValueError("`max_samples` is incompatible with `streaming`.")

        if self.mask_history and self.train_on_prompt:
            raise ValueError("`mask_history` is incompatible with `train_on_prompt`.")

        if self.loss_skip_first_n_lines > 0 and self.train_on_prompt:
            raise ValueError("`loss_skip_first_n_lines` is incompatible with `train_on_prompt`.")

        if self.neat_packing:
            self.packing = True

        if self.user_balanced_loss and self.packing:
            raise ValueError(
                "`user_balanced_loss=True` is incompatible with `packing`/`neat_packing` (per-user "
                "weighting needs one sample per row to attach its `loss_weight`; packing merges "
                "multiple samples into one row). Set `packing: false`."
            )

        if self.packing:
            self.cutoff_len -= 1  # avoid pad_to_multiple_of, needs improve

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
