# Copyright 2025 the LlamaFactory team.
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
class FreezeArguments:
    r"""Arguments pertaining to the freeze (partial-parameter) training."""

    freeze_trainable_layers: int = field(
        default=2,
        metadata={
            "help": (
                "The number of trainable layers for freeze (partial-parameter) fine-tuning. "
                "Positive numbers mean the last n layers are set as trainable, "
                "negative numbers mean the first n layers are set as trainable."
            )
        },
    )
    freeze_trainable_modules: str = field(
        default="all",
        metadata={
            "help": (
                "Name(s) of trainable modules for freeze (partial-parameter) fine-tuning. "
                "Use commas to separate multiple modules. "
                "Use `all` to specify all the available modules."
            )
        },
    )
    freeze_extra_modules: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Name(s) of modules apart from hidden layers to be set as trainable "
                "for freeze (partial-parameter) fine-tuning. "
                "Use commas to separate multiple modules."
            )
        },
    )
    original_vocab_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The original vocabulary size before adding new tokens. "
                "If set, only the newly added token embeddings (indices >= original_vocab_size) will be trained, "
                "while the original token embeddings remain frozen. "
                "This is useful for training new special tokens without affecting the model's original language ability."
            )
        },
    )


@dataclass
class LoraArguments:
    r"""Arguments pertaining to the LoRA training."""

    additional_target: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Name(s) of modules apart from LoRA layers to be set as trainable "
                "and saved in the final checkpoint. "
                "Use commas to separate multiple modules."
            )
        },
    )
    lora_alpha: Optional[int] = field(
        default=None,
        metadata={"help": "The scale factor for LoRA fine-tuning (default: lora_rank * 2)."},
    )
    lora_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout rate for the LoRA fine-tuning."},
    )
    lora_rank: int = field(
        default=8,
        metadata={"help": "The intrinsic dimension for LoRA fine-tuning."},
    )
    lora_target: str = field(
        default="all",
        metadata={
            "help": (
                "Name(s) of target modules to apply LoRA. "
                "Use commas to separate multiple modules. "
                "Use `all` to specify all the linear modules."
            )
        },
    )
    loraplus_lr_ratio: Optional[float] = field(
        default=None,
        metadata={"help": "LoRA plus learning rate ratio (lr_B / lr_A)."},
    )
    loraplus_lr_embedding: float = field(
        default=1e-6,
        metadata={"help": "LoRA plus learning rate for lora embedding layers."},
    )
    use_rslora: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the rank stabilization scaling factor for LoRA layer."},
    )
    use_dora: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the weight-decomposed lora method (DoRA)."},
    )
    pissa_init: bool = field(
        default=False,
        metadata={"help": "Whether or not to initialize a PiSSA adapter."},
    )
    pissa_iter: int = field(
        default=16,
        metadata={"help": "The number of iteration steps performed by FSVD in PiSSA. Use -1 to disable it."},
    )
    pissa_convert: bool = field(
        default=False,
        metadata={"help": "Whether or not to convert the PiSSA adapter to a normal LoRA adapter."},
    )
    create_new_adapter: bool = field(
        default=False,
        metadata={"help": "Whether or not to create a new adapter with randomly initialized weight."},
    )


@dataclass
class OFTArguments:
    r"""Arguments pertaining to the OFT training."""

    additional_target: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Name(s) of modules apart from LoRA layers to be set as trainable "
                "and saved in the final checkpoint. "
                "Use commas to separate multiple modules."
            )
        },
    )
    module_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout rate for the OFT fine-tuning."},
    )
    oft_rank: int = field(
        default=0,
        metadata={"help": "The intrinsic dimension for OFT fine-tuning."},
    )
    oft_block_size: int = field(
        default=32,
        metadata={"help": "The intrinsic dimension for OFT fine-tuning."},
    )
    oft_target: str = field(
        default="all",
        metadata={
            "help": (
                "Name(s) of target modules to apply OFT. "
                "Use commas to separate multiple modules. "
                "Use `all` to specify all the linear modules."
            )
        },
    )
    create_new_adapter: bool = field(
        default=False,
        metadata={"help": "Whether or not to create a new adapter with randomly initialized weight."},
    )


@dataclass
class RLHFArguments:
    r"""Arguments pertaining to the PPO, DPO and KTO training."""

    pref_beta: float = field(
        default=0.1,
        metadata={"help": "The beta parameter in the preference loss."},
    )
    pref_ftx: float = field(
        default=0.0,
        metadata={"help": "The supervised fine-tuning loss coefficient in DPO training."},
    )
    pref_bco_weight: float = field(
        default=0.0,
        metadata={"help": "The Binary Classifier Optimization coefficient in DPO training."},
    )
    pref_loss: Literal["sigmoid", "hinge", "ipo", "kto_pair", "orpo", "simpo"] = field(
        default="sigmoid",
        metadata={"help": "The type of DPO loss to use."},
    )
    dpo_label_smoothing: float = field(
        default=0.0,
        metadata={"help": "The robust DPO label smoothing parameter in cDPO that should be between 0 and 0.5."},
    )
    kto_chosen_weight: float = field(
        default=1.0,
        metadata={"help": "The weight factor of the desirable losses in KTO training."},
    )
    kto_rejected_weight: float = field(
        default=1.0,
        metadata={"help": "The weight factor of the undesirable losses in KTO training."},
    )
    simpo_gamma: float = field(
        default=0.5,
        metadata={"help": "The target reward margin term in SimPO loss."},
    )
    ppo_buffer_size: int = field(
        default=1,
        metadata={"help": "The number of mini-batches to make experience buffer in a PPO optimization step."},
    )
    ppo_epochs: int = field(
        default=4,
        metadata={"help": "The number of epochs to perform in a PPO optimization step."},
    )
    ppo_score_norm: bool = field(
        default=False,
        metadata={"help": "Use score normalization in PPO training."},
    )
    ppo_target: float = field(
        default=6.0,
        metadata={"help": "Target KL value for adaptive KL control in PPO training."},
    )
    ppo_whiten_rewards: bool = field(
        default=False,
        metadata={"help": "Whiten the rewards before compute advantages in PPO training."},
    )
    ref_model: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the reference model used for the PPO or DPO training."},
    )
    ref_model_adapters: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the adapters of the reference model."},
    )
    ref_model_quantization_bit: Optional[int] = field(
        default=None,
        metadata={"help": "The number of bits to quantize the reference model."},
    )
    reward_model: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the reward model used for the PPO training."},
    )
    reward_model_adapters: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the adapters of the reward model."},
    )
    reward_model_quantization_bit: Optional[int] = field(
        default=None,
        metadata={"help": "The number of bits to quantize the reward model."},
    )
    reward_model_type: Literal["lora", "full", "api"] = field(
        default="lora",
        metadata={"help": "The type of the reward model in PPO training. Lora model only supports lora training."},
    )
    ld_alpha: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Alpha parameter from the LD-DPO paper, which controls the weighting of"
                " the verbose token log-probabilities in responses."
            )
        },
    )


@dataclass
class GaloreArguments:
    r"""Arguments pertaining to the GaLore algorithm."""

    use_galore: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the gradient low-Rank projection (GaLore)."},
    )
    galore_target: str = field(
        default="all",
        metadata={
            "help": (
                "Name(s) of modules to apply GaLore. Use commas to separate multiple modules. "
                "Use `all` to specify all the linear modules."
            )
        },
    )
    galore_rank: int = field(
        default=16,
        metadata={"help": "The rank of GaLore gradients."},
    )
    galore_update_interval: int = field(
        default=200,
        metadata={"help": "Number of steps to update the GaLore projection."},
    )
    galore_scale: float = field(
        default=2.0,
        metadata={"help": "GaLore scaling coefficient."},
    )
    galore_proj_type: Literal["std", "reverse_std", "right", "left", "full"] = field(
        default="std",
        metadata={"help": "Type of GaLore projection."},
    )
    galore_layerwise: bool = field(
        default=False,
        metadata={"help": "Whether or not to enable layer-wise update to further save memory."},
    )


@dataclass
class ApolloArguments:
    r"""Arguments pertaining to the APOLLO algorithm."""

    use_apollo: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the APOLLO optimizer."},
    )
    apollo_target: str = field(
        default="all",
        metadata={
            "help": (
                "Name(s) of modules to apply APOLLO. Use commas to separate multiple modules. "
                "Use `all` to specify all the linear modules."
            )
        },
    )
    apollo_rank: int = field(
        default=16,
        metadata={"help": "The rank of APOLLO gradients."},
    )
    apollo_update_interval: int = field(
        default=200,
        metadata={"help": "Number of steps to update the APOLLO projection."},
    )
    apollo_scale: float = field(
        default=32.0,
        metadata={"help": "APOLLO scaling coefficient."},
    )
    apollo_proj: Literal["svd", "random"] = field(
        default="random",
        metadata={"help": "Type of APOLLO low-rank projection algorithm (svd or random)."},
    )
    apollo_proj_type: Literal["std", "right", "left"] = field(
        default="std",
        metadata={"help": "Type of APOLLO projection."},
    )
    apollo_scale_type: Literal["channel", "tensor"] = field(
        default="channel",
        metadata={"help": "Type of APOLLO scaling (channel or tensor)."},
    )
    apollo_layerwise: bool = field(
        default=False,
        metadata={"help": "Whether or not to enable layer-wise update to further save memory."},
    )
    apollo_scale_front: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the norm-growth limiter in front of gradient scaling."},
    )


@dataclass
class BAdamArgument:
    r"""Arguments pertaining to the BAdam optimizer."""

    use_badam: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the BAdam optimizer."},
    )
    badam_mode: Literal["layer", "ratio"] = field(
        default="layer",
        metadata={"help": "Whether to use layer-wise or ratio-wise BAdam optimizer."},
    )
    badam_start_block: Optional[int] = field(
        default=None,
        metadata={"help": "The starting block index for layer-wise BAdam."},
    )
    badam_switch_mode: Optional[Literal["ascending", "descending", "random", "fixed"]] = field(
        default="ascending",
        metadata={"help": "the strategy of picking block to update for layer-wise BAdam."},
    )
    badam_switch_interval: Optional[int] = field(
        default=50,
        metadata={
            "help": "Number of steps to update the block for layer-wise BAdam. Use -1 to disable the block update."
        },
    )
    badam_update_ratio: float = field(
        default=0.05,
        metadata={"help": "The ratio of the update for ratio-wise BAdam."},
    )
    badam_mask_mode: Literal["adjacent", "scatter"] = field(
        default="adjacent",
        metadata={
            "help": (
                "The mode of the mask for BAdam optimizer. "
                "`adjacent` means that the trainable parameters are adjacent to each other, "
                "`scatter` means that trainable parameters are randomly choosed from the weight."
            )
        },
    )
    badam_verbose: int = field(
        default=0,
        metadata={
            "help": (
                "The verbosity level of BAdam optimizer. "
                "0 for no print, 1 for print the block prefix, 2 for print trainable parameters."
            )
        },
    )


@dataclass
class SwanLabArguments:
    use_swanlab: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the SwanLab (an experiment tracking and visualization tool)."},
    )
    swanlab_project: Optional[str] = field(
        default="llamafactory",
        metadata={"help": "The project name in SwanLab."},
    )
    swanlab_workspace: Optional[str] = field(
        default=None,
        metadata={"help": "The workspace name in SwanLab."},
    )
    swanlab_run_name: Optional[str] = field(
        default=None,
        metadata={"help": "The experiment name in SwanLab."},
    )
    swanlab_mode: Literal["cloud", "local"] = field(
        default="cloud",
        metadata={"help": "The mode of SwanLab."},
    )
    swanlab_api_key: Optional[str] = field(
        default=None,
        metadata={"help": "The API key for SwanLab."},
    )
    swanlab_logdir: Optional[str] = field(
        default=None,
        metadata={"help": "The log directory for SwanLab."},
    )
    swanlab_lark_webhook_url: Optional[str] = field(
        default=None,
        metadata={"help": "The Lark(飞书) webhook URL for SwanLab."},
    )
    swanlab_lark_secret: Optional[str] = field(
        default=None,
        metadata={"help": "The Lark(飞书) secret for SwanLab."},
    )


@dataclass
class FinetuningArguments(
    SwanLabArguments,
    BAdamArgument,
    ApolloArguments,
    GaloreArguments,
    RLHFArguments,
    LoraArguments,
    OFTArguments,
    FreezeArguments,
):
    r"""Arguments pertaining to which techniques we are going to fine-tuning with."""

    pure_bf16: bool = field(
        default=False,
        metadata={"help": "Whether or not to train model in purely bf16 precision (without AMP)."},
    )
    stage: Literal["pt", "sft", "rm", "ppo", "dpo", "kto", "seq_align"] = field(
        default="sft",
        metadata={"help": "Which stage will be performed in training."},
    )
    finetuning_type: Literal["lora", "oft", "freeze", "full"] = field(
        default="lora",
        metadata={"help": "Which fine-tuning method to use."},
    )
    use_llama_pro: bool = field(
        default=False,
        metadata={"help": "Whether or not to make only the parameters in the expanded blocks trainable."},
    )
    use_adam_mini: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the Adam-mini optimizer."},
    )
    use_mca: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to use MCA (Megatron Core Adapter) training. "
                "Controlled by USE_MCA environment variable."
            )
        },
    )
    use_muon: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the Muon optimizer."},
    )
    use_dft_loss: bool = field(
        default=False,
        metadata={"help": "Whether to use the DFT loss."},
    )
    log_predictions_steps: int = field(
        default=0,
        metadata={"help": "Log model predictions every N steps during training. Set to 0 to disable."},
    )
    freeze_vision_tower: bool = field(
        default=True,
        metadata={"help": "Whether ot not to freeze the vision tower in MLLM training."},
    )
    freeze_multi_modal_projector: bool = field(
        default=True,
        metadata={"help": "Whether or not to freeze the multi modal projector in MLLM training."},
    )
    freeze_language_model: bool = field(
        default=False,
        metadata={"help": "Whether or not to freeze the language model in MLLM training."},
    )
    compute_accuracy: bool = field(
        default=False,
        metadata={"help": "Whether or not to compute the token-level accuracy at evaluation."},
    )
    compute_payment_accuracy: bool = field(
        default=False,
        metadata={"help": "Whether or not to compute the payment behavior prediction accuracy at evaluation."},
    )
    compute_behavior_hr: bool = field(
        default=False,
        metadata={"help": "Whether or not to compute HR@K for SFT behavior prediction using embedding similarity."},
    )
    behavior_hr_k_list: str = field(
        default="1,5,10,50",
        metadata={"help": "Comma-separated list of K values for behavior HR@K computation."},
    )
    compute_hit_rate: bool = field(
        default=False,
        metadata={"help": "Whether or not to compute HR@K for behavior prediction at evaluation."},
    )
    hit_rate_k_list: str = field(
        default="1,5,10,50",
        metadata={"help": "Comma-separated list of K values for HR@K computation."},
    )
    hr_last_position_only: bool = field(
        default=False,
        metadata={"help": "If True, only compute HR@K at the last position (like SASRec). If False, compute at all positions (LLM-style)."},
    )
    beh_tokens_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to beh_tokens.json for HR@K computation (only softmax on beh tokens)."},
    )
    # -------------------------
    # Recall@K for SFT (next-item prediction)
    # -------------------------
    recall_k_list: str = field(
        default="1,5,10",
        metadata={"help": "Comma-separated list of K values for SFT Recall@K computation."},
    )
    eval_candidate_start_id: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional explicit start token id (inclusive) for SFT Recall@K candidates. "
                "If set (optionally with eval_candidate_end_id), this range is used before original_vocab_size."
            )
        },
    )
    eval_candidate_end_id: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional explicit end token id (exclusive) for SFT Recall@K candidates. "
                "If omitted while eval_candidate_start_id is set, defaults to model vocab size."
            )
        },
    )
    eval_fav_start_id: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional explicit start token id (inclusive) for favorite-category/genre eval Recall@K candidates. "
                "When eval dataset name contains fav_category/fav_genre, this range has higher priority than eval_candidate_* and original_vocab_size."
            )
        },
    )
    eval_fav_end_id: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional explicit end token id (exclusive) for favorite-category/genre eval Recall@K candidates. "
                "If omitted while eval_fav_start_id is set, defaults to model vocab size."
            )
        },
    )

    # -------------------------
    # Optional: Grouped LR via gradient scaling (primarily for Stage2 full PT)
    # -------------------------
    use_group_lr: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, enable grouped learning-rate behavior via gradient scaling hooks: "
                "beh token embedding rows (>= original_vocab_size) get a higher effective LR, "
                "lm_head can use a smaller effective LR, backbone uses the base training_args.learning_rate. "
                "Ratios decay naturally with the scheduler."
            )
        },
    )
    group_lr_beh_ratio: float = field(
        default=5.0,
        metadata={"help": "Effective LR multiplier for beh embedding rows (token_id >= original_vocab_size). Typical 3~10."},
    )
    group_lr_lm_head_ratio: float = field(
        default=1.0,
        metadata={"help": "Effective LR multiplier for lm_head parameters. Usually 1.0 (same as backbone) or smaller."},
    )
    # -------------------------
    # Seq-Align (BehaviorProjector) stage
    # -------------------------
    seq_align_projector_layers: int = field(
        default=2,
        metadata={"help": "Number of MLP layers in the BehaviorProjector."},
    )
    seq_align_projector_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout in the BehaviorProjector MLP."},
    )
    seq_align_projector_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a pre-trained projector directory to resume from."},
    )
    seq_align_behavior_token_start_id: int = field(
        default=151675,
        metadata={"help": "Token IDs >= this value are treated as behavior tokens and passed through the projector."},
    )
    seq_align_freeze_llm: bool = field(
        default=True,
        metadata={"help": "Phase 1: True (only projector trainable). Phase 2: False (projector + LLM trainable)."},
    )
    # -------------------------
    # Stage1 extra losses for <beh_*> embedding warm-up
    # -------------------------
    stage1_beh_text_csv_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "CSV path that maps behavior ids to canonical descriptions used by Stage1 extra loss "
                "(beh->text generation)."
            )
        },
    )
    stage1_beh_text_id_col: str = field(
        default="feature_lookup_id",
        metadata={"help": "CSV id column for behavior inner id, used to form token <beh_{id}>."},
    )
    stage1_beh_text_col: str = field(
        default="summary_tags",
        metadata={"help": "CSV text column for canonical behavior description (e.g., summary_tags)."},
    )
    stage1_gen_loss_weight: float = field(
        default=0.0,
        metadata={"help": "Weight for beh->text generation CE loss. Set >0 to enable."},
    )
    stage1_gen_batch_size: int = field(
        default=32,
        metadata={"help": "Number of (beh, desc) pairs used per step for beh->text generation loss."},
    )
    stage1_gen_prompt_template: str = field(
        default="行为含义: {beh_token} -> ",
        metadata={"help": "Prompt template for generation loss, must contain {beh_token} placeholder."},
    )
    stage1_log_steps: int = field(
        default=500,
        metadata={"help": "Log Stage1 diagnostics and (optionally) generation examples every N optimizer steps. Set 0 to disable."},
    )
    stage1_log_num_examples: int = field(
        default=3,
        metadata={"help": "Number of beh->text reconstruction examples to print when stage1_log_steps triggers."},
    )
    stage1_gen_max_new_tokens: int = field(
        default=64,
        metadata={"help": "Max new tokens for periodic beh->text reconstruction preview generation."},
    )
    stage1_gen_num_beams: int = field(
        default=1,
        metadata={"help": "num_beams for periodic reconstruction preview generation (default 1=greedy)."},
    )
    stage1_gen_do_sample: bool = field(
        default=False,
        metadata={"help": "Whether to use sampling for periodic reconstruction preview generation (default False)."},
    )
    disable_shuffling: bool = field(
        default=False,
        metadata={"help": "Whether or not to disable the shuffling of the training set."},
    )
    # -------------------------
    # Behavior-ID-Only Loss (for next-item prediction downstream task)
    # -------------------------
    beh_only_loss: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, only compute loss on behavior token positions (token_id in [beh_only_loss_start_id, beh_only_loss_end_id)). "
                "This is useful for next-item prediction downstream tasks where we only care about predicting item IDs."
            )
        },
    )
    beh_only_loss_start_id: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Start token id (inclusive) for behavior-only loss and softmax range. "
                "Labels outside [start_id, end_id) will be masked (set to -100). "
                "Softmax will also be restricted to this range for more efficient training."
            )
        },
    )
    beh_only_loss_end_id: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "End token id (exclusive) for behavior-only loss and softmax range. "
                "If omitted while beh_only_loss_start_id is set, defaults to model vocab size."
            )
        },
    )
    # -------------------------
    # SID legal-candidate constraint (masked CE over per-prefix legal SID sets)
    # -------------------------
    sid_constraint: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, apply masked cross-entropy over the legal SID candidate set at each "
                "SID slot (s1 -> valid_s1, s2 -> a2b[s1], s3 -> ab2c[(s1,s2)] or valid_s3). The "
                "model forward is unchanged; only the loss renormalizes the softmax to the "
                "current sample's prefix-legal candidates. Requires `sid_constraint_path`."
            )
        },
    )
    sid_constraint_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Path to the SID constraint .npz produced by scripts/build_sid_constraints.py. "
                "Required when `sid_constraint=True`."
            )
        },
    )
    sid_constraint_s3_mode: str = field(
        default="valid_s3",
        metadata={
            "help": (
                "Third-position (s3) constraint mode: 'csr' uses the tight per-prefix ab2c table "
                "(CSR); 'valid_s3' only requires s3 to fall inside the global set of third-position "
                "codes that ever appeared. The first two SID positions are always tightly "
                "constrained; only s3 is switched here. Default 'valid_s3'."
            )
        },
    )
    early_stopping_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Number of steps to stop training if the `metric_for_best_model` does not improve."},
    )
    plot_loss: bool = field(
        default=False,
        metadata={"help": "Whether or not to save the training loss curves."},
    )
    include_effective_tokens_per_second: bool = field(
        default=False,
        metadata={"help": "Whether or not to compute effective tokens per second."},
    )
    def __post_init__(self):
        def split_arg(arg):
            if isinstance(arg, str):
                return [item.strip() for item in arg.split(",")]
            return arg

        self.freeze_trainable_modules: list[str] = split_arg(self.freeze_trainable_modules)
        self.freeze_extra_modules: Optional[list[str]] = split_arg(self.freeze_extra_modules)
        self.lora_alpha: int = self.lora_alpha or self.lora_rank * 2
        self.lora_target: list[str] = split_arg(self.lora_target)
        self.oft_target: list[str] = split_arg(self.oft_target)
        self.additional_target: Optional[list[str]] = split_arg(self.additional_target)
        self.galore_target: list[str] = split_arg(self.galore_target)
        self.apollo_target: list[str] = split_arg(self.apollo_target)
        self.hit_rate_k_list: list[int] = [int(k) for k in self.hit_rate_k_list.split(",")]
        self.recall_k_list: list[int] = [int(k) for k in self.recall_k_list.split(",")] if isinstance(self.recall_k_list, str) else self.recall_k_list
        self.use_ref_model = self.stage == "dpo" and self.pref_loss not in ["orpo", "simpo"]

        if self.eval_candidate_start_id is not None and self.eval_candidate_end_id is not None:
            if int(self.eval_candidate_end_id) <= int(self.eval_candidate_start_id):
                raise ValueError("`eval_candidate_end_id` must be greater than `eval_candidate_start_id`.")

        if self.eval_fav_start_id is not None and self.eval_fav_end_id is not None:
            if int(self.eval_fav_end_id) <= int(self.eval_fav_start_id):
                raise ValueError("`eval_fav_end_id` must be greater than `eval_fav_start_id`.")

        assert self.stage in ["pt", "sft", "rm", "ppo", "dpo", "kto", "seq_align"], (
            f"Unknown stage: {self.stage}. "
            "Supported: pt, sft, rm, ppo, dpo, kto, seq_align."
        )
        assert self.finetuning_type in ["lora", "oft", "freeze", "full"], "Invalid fine-tuning method."
        assert self.ref_model_quantization_bit in [None, 8, 4], "We only accept 4-bit or 8-bit quantization."
        assert self.reward_model_quantization_bit in [None, 8, 4], "We only accept 4-bit or 8-bit quantization."

        if self.stage == "ppo" and self.reward_model is None:
            raise ValueError("`reward_model` is necessary for PPO training.")

        if self.stage == "ppo" and self.reward_model_type == "lora" and self.finetuning_type != "lora":
            raise ValueError("`reward_model_type` cannot be lora for Freeze/Full PPO training.")

        if self.stage == "ppo" and self.reward_model_type == "oft" and self.finetuning_type != "oft":
            raise ValueError("`reward_model_type` cannot be oft for Freeze/Full PPO training.")

        if self.stage == "dpo" and self.pref_loss != "sigmoid" and self.dpo_label_smoothing > 1e-6:
            raise ValueError("`dpo_label_smoothing` is only valid for sigmoid loss function.")

        if self.use_llama_pro and self.finetuning_type == "full":
            raise ValueError("`use_llama_pro` is only valid for Freeze or LoRA training.")

        if self.finetuning_type == "lora" and (self.use_galore or self.use_apollo or self.use_badam):
            raise ValueError("Cannot use LoRA with GaLore, APOLLO or BAdam together.")

        if int(self.use_galore) + int(self.use_apollo) + (self.use_badam) > 1:
            raise ValueError("Cannot use GaLore, APOLLO or BAdam together.")

        if self.pissa_init and (self.stage in ["ppo", "kto"] or self.use_ref_model):
            raise ValueError("Cannot use PiSSA for current training stage.")

        if self.finetuning_type != "lora":
            if self.loraplus_lr_ratio is not None:
                raise ValueError("`loraplus_lr_ratio` is only valid for LoRA training.")

            if self.use_rslora:
                raise ValueError("`use_rslora` is only valid for LoRA training.")

            if self.use_dora:
                raise ValueError("`use_dora` is only valid for LoRA training.")

            if self.pissa_init:
                raise ValueError("`pissa_init` is only valid for LoRA training.")

    def to_dict(self) -> dict[str, Any]:
        args = asdict(self)
        args = {k: f"<{k.upper()}>" if k.endswith("api_key") else v for k, v in args.items()}
        return args
