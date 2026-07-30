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

import json
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Optional

import torch
import transformers
from peft import PeftModel
from transformers import PreTrainedModel, ProcessorMixin, TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, has_length
from transformers.utils import SAFE_WEIGHTS_NAME, WEIGHTS_NAME
from typing_extensions import override

from ..extras import logging
from ..extras.constants import TRAINER_LOG, V_HEAD_SAFE_WEIGHTS_NAME, V_HEAD_WEIGHTS_NAME
from ..extras.misc import get_peak_memory, is_env_enabled, use_ray
from ..extras.packages import is_safetensors_available


if is_safetensors_available():
    from safetensors import safe_open
    from safetensors.torch import save_file


if TYPE_CHECKING:
    from transformers import TrainerControl, TrainerState, TrainingArguments
    from trl import AutoModelForCausalLMWithValueHead

    from ..hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


logger = logging.get_logger(__name__)


def fix_valuehead_checkpoint(
    model: "AutoModelForCausalLMWithValueHead", output_dir: str, safe_serialization: bool
) -> None:
    r"""Fix the valuehead checkpoint files.

    The model is already unwrapped.

    There are three cases:
    1. full tuning without ds_zero3: state_dict = {"model.layers.*": ..., "v_head.summary.*": ...}
    2. lora tuning without ds_zero3: state_dict = {"v_head.summary.*": ...}
    3. under deepspeed zero3: state_dict = {"pretrained_model.model.layers.*": ..., "v_head.summary.*": ...}

    We assume `stage3_gather_16bit_weights_on_model_save=true`.
    """
    if not isinstance(model.pretrained_model, (PreTrainedModel, PeftModel)):
        return

    if safe_serialization:
        path_to_checkpoint = os.path.join(output_dir, SAFE_WEIGHTS_NAME)
        with safe_open(path_to_checkpoint, framework="pt", device="cpu") as f:
            state_dict: dict[str, torch.Tensor] = {key: f.get_tensor(key).clone() for key in f.keys()}
    else:
        path_to_checkpoint = os.path.join(output_dir, WEIGHTS_NAME)
        state_dict: dict[str, torch.Tensor] = torch.load(path_to_checkpoint, map_location="cpu", weights_only=True)

    os.remove(path_to_checkpoint)
    decoder_state_dict, v_head_state_dict = {}, {}
    for name, param in state_dict.items():
        if name.startswith("v_head."):
            v_head_state_dict[name] = param
        else:
            decoder_state_dict[name.replace("pretrained_model.", "", 1)] = param

    model.pretrained_model.save_pretrained(
        output_dir, state_dict=decoder_state_dict or None, safe_serialization=safe_serialization
    )

    if safe_serialization:
        save_file(v_head_state_dict, os.path.join(output_dir, V_HEAD_SAFE_WEIGHTS_NAME), metadata={"format": "pt"})
    else:
        torch.save(v_head_state_dict, os.path.join(output_dir, V_HEAD_WEIGHTS_NAME))

    logger.info_rank0(f"Value head model saved at: {output_dir}")


class FixValueHeadModelCallback(TrainerCallback):
    r"""A callback for fixing the checkpoint for valuehead models."""

    @override
    def on_save(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            output_dir = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
            fix_valuehead_checkpoint(
                model=kwargs.pop("model"), output_dir=output_dir, safe_serialization=args.save_safetensors
            )


class SaveProcessorCallback(TrainerCallback):
    r"""A callback for saving the processor."""

    def __init__(self, processor: "ProcessorMixin") -> None:
        self.processor = processor

    @override
    def on_save(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            output_dir = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
            self.processor.save_pretrained(output_dir)

    @override
    def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            self.processor.save_pretrained(args.output_dir)


class PissaConvertCallback(TrainerCallback):
    r"""A callback for converting the PiSSA adapter to a normal one."""

    @override
    def on_train_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            model = kwargs.pop("model")
            pissa_init_dir = os.path.join(args.output_dir, "pissa_init")
            logger.info_rank0(f"Initial PiSSA adapter will be saved at: {pissa_init_dir}.")
            if isinstance(model, PeftModel):
                init_lora_weights = getattr(model.peft_config["default"], "init_lora_weights")
                setattr(model.peft_config["default"], "init_lora_weights", True)
                model.save_pretrained(pissa_init_dir, safe_serialization=args.save_safetensors)
                setattr(model.peft_config["default"], "init_lora_weights", init_lora_weights)

    @override
    def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            model = kwargs.pop("model")
            pissa_init_dir = os.path.join(args.output_dir, "pissa_init")
            pissa_backup_dir = os.path.join(args.output_dir, "pissa_backup")
            pissa_convert_dir = os.path.join(args.output_dir, "pissa_converted")
            logger.info_rank0(f"Converted PiSSA adapter will be saved at: {pissa_convert_dir}.")
            # 1. save a pissa backup with init_lora_weights: True
            # 2. save a converted lora with init_lora_weights: pissa
            # 3. load the pissa backup with init_lora_weights: True
            # 4. delete the initial adapter and change init_lora_weights to pissa
            if isinstance(model, PeftModel):
                init_lora_weights = getattr(model.peft_config["default"], "init_lora_weights")
                setattr(model.peft_config["default"], "init_lora_weights", True)
                model.save_pretrained(pissa_backup_dir, safe_serialization=args.save_safetensors)
                setattr(model.peft_config["default"], "init_lora_weights", init_lora_weights)
                model.save_pretrained(
                    pissa_convert_dir,
                    safe_serialization=args.save_safetensors,
                    path_initial_model_for_weight_conversion=pissa_init_dir,
                )
                model.load_adapter(pissa_backup_dir, "default", is_trainable=True)
                model.set_adapter("default")
                setattr(model.peft_config["default"], "init_lora_weights", init_lora_weights)


class LogCallback(TrainerCallback):
    r"""A callback for logging training and evaluation status."""

    def __init__(self) -> None:
        # Progress
        self.start_time = 0
        self.cur_steps = 0
        self.max_steps = 0
        self.elapsed_time = ""
        self.remaining_time = ""
        self.thread_pool: Optional[ThreadPoolExecutor] = None
        # Status
        self.aborted = False
        self.do_train = False
        # Web UI
        self.webui_mode = is_env_enabled("LLAMABOARD_ENABLED")
        if self.webui_mode and not use_ray():
            signal.signal(signal.SIGABRT, self._set_abort)
            self.logger_handler = logging.LoggerHandler(os.getenv("LLAMABOARD_WORKDIR"))
            logging.add_handler(self.logger_handler)
            transformers.logging.add_handler(self.logger_handler)

    def _set_abort(self, signum, frame) -> None:
        self.aborted = True

    def _reset(self, max_steps: int = 0) -> None:
        self.start_time = time.time()
        self.cur_steps = 0
        self.max_steps = max_steps
        self.elapsed_time = ""
        self.remaining_time = ""

    def _timing(self, cur_steps: int) -> None:
        cur_time = time.time()
        elapsed_time = cur_time - self.start_time
        avg_time_per_step = elapsed_time / cur_steps if cur_steps != 0 else 0
        remaining_time = (self.max_steps - cur_steps) * avg_time_per_step
        self.cur_steps = cur_steps
        self.elapsed_time = str(timedelta(seconds=int(elapsed_time)))
        self.remaining_time = str(timedelta(seconds=int(remaining_time)))

    def _write_log(self, output_dir: str, logs: dict[str, Any]) -> None:
        with open(os.path.join(output_dir, TRAINER_LOG), "a", encoding="utf-8") as f:
            f.write(json.dumps(logs) + "\n")

    def _create_thread_pool(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        self.thread_pool = ThreadPoolExecutor(max_workers=1)

    def _close_thread_pool(self) -> None:
        if self.thread_pool is not None:
            self.thread_pool.shutdown(wait=True)
            self.thread_pool = None

    @override
    def on_init_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if (
            args.should_save
            and os.path.exists(os.path.join(args.output_dir, TRAINER_LOG))
            and args.overwrite_output_dir
        ):
            logger.warning_rank0_once("Previous trainer log in this folder will be deleted.")
            os.remove(os.path.join(args.output_dir, TRAINER_LOG))

    @override
    def on_train_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            self.do_train = True
            self._reset(max_steps=state.max_steps)
            self._create_thread_pool(output_dir=args.output_dir)

    @override
    def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        self._close_thread_pool()

    @override
    def on_substep_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if self.aborted:
            control.should_epoch_stop = True
            control.should_training_stop = True

    @override
    def on_step_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if self.aborted:
            control.should_epoch_stop = True
            control.should_training_stop = True

    @override
    def on_evaluate(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not self.do_train:
            self._close_thread_pool()

    @override
    def on_predict(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not self.do_train:
            self._close_thread_pool()

    @override
    def on_log(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not args.should_save:
            return

        self._timing(cur_steps=state.global_step)
        logs = dict(
            current_steps=self.cur_steps,
            total_steps=self.max_steps,
            loss=state.log_history[-1].get("loss"),
            eval_loss=state.log_history[-1].get("eval_loss"),
            predict_loss=state.log_history[-1].get("predict_loss"),
            reward=state.log_history[-1].get("reward"),
            accuracy=state.log_history[-1].get("rewards/accuracies"),
            lr=state.log_history[-1].get("learning_rate"),
            epoch=state.log_history[-1].get("epoch"),
            percentage=round(self.cur_steps / self.max_steps * 100, 2) if self.max_steps != 0 else 100,
            elapsed_time=self.elapsed_time,
            remaining_time=self.remaining_time,
        )
        if state.num_input_tokens_seen:
            logs["throughput"] = round(state.num_input_tokens_seen / (time.time() - self.start_time), 2)
            logs["total_tokens"] = state.num_input_tokens_seen

        if is_env_enabled("RECORD_VRAM"):
            vram_allocated, vram_reserved = get_peak_memory()
            logs["vram_allocated"] = round(vram_allocated / (1024**3), 2)
            logs["vram_reserved"] = round(vram_reserved / (1024**3), 2)

        logs = {k: v for k, v in logs.items() if v is not None}
        if self.webui_mode and all(key in logs for key in ("loss", "lr", "epoch")):
            log_str = f"'loss': {logs['loss']:.4f}, 'learning_rate': {logs['lr']:2.4e}, 'epoch': {logs['epoch']:.2f}"
            for extra_key in ("reward", "accuracy", "throughput"):
                if logs.get(extra_key):
                    log_str += f", '{extra_key}': {logs[extra_key]:.2f}"

            logger.info_rank0("{" + log_str + "}")

        if self.thread_pool is not None:
            self.thread_pool.submit(self._write_log, args.output_dir, logs)

    @override
    def on_prediction_step(
        self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs
    ):
        if self.do_train:
            return

        if self.aborted:
            sys.exit(0)

        if not args.should_save:
            return

        eval_dataloader = kwargs.pop("eval_dataloader", None)
        if has_length(eval_dataloader):
            if self.max_steps == 0:
                self._reset(max_steps=len(eval_dataloader))
                self._create_thread_pool(output_dir=args.output_dir)

            self._timing(cur_steps=self.cur_steps + 1)
            if self.cur_steps % 5 == 0 and self.thread_pool is not None:
                logs = dict(
                    current_steps=self.cur_steps,
                    total_steps=self.max_steps,
                    percentage=round(self.cur_steps / self.max_steps * 100, 2) if self.max_steps != 0 else 100,
                    elapsed_time=self.elapsed_time,
                    remaining_time=self.remaining_time,
                )
                self.thread_pool.submit(self._write_log, args.output_dir, logs)


class ReporterCallback(TrainerCallback):
    r"""A callback for reporting training status to external logger."""

    def __init__(
        self,
        model_args: "ModelArguments",
        data_args: "DataArguments",
        finetuning_args: "FinetuningArguments",
        generating_args: "GeneratingArguments",
    ) -> None:
        self.model_args = model_args
        self.data_args = data_args
        self.finetuning_args = finetuning_args
        self.generating_args = generating_args
        os.environ["WANDB_PROJECT"] = os.getenv("WANDB_PROJECT", "llamafactory")

    @override
    def on_train_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not state.is_world_process_zero:
            return

        if "wandb" in args.report_to:
            import wandb

            wandb.config.update(
                {
                    "model_args": self.model_args.to_dict(),
                    "data_args": self.data_args.to_dict(),
                    "finetuning_args": self.finetuning_args.to_dict(),
                    "generating_args": self.generating_args.to_dict(),
                }
            )

        if self.finetuning_args.use_swanlab:
            import swanlab  # type: ignore

            swanlab.config.update(
                {
                    "model_args": self.model_args.to_dict(),
                    "data_args": self.data_args.to_dict(),
                    "finetuning_args": self.finetuning_args.to_dict(),
                    "generating_args": self.generating_args.to_dict(),
                }
            )


class PredictionLogCallback(TrainerCallback):
    """在训练过程中定期打印模型的预测输出。"""
    
    def __init__(self, tokenizer, sample_input_ids: list[int], sample_label: str, log_steps: int = 500):
        """
        Args:
            tokenizer: 分词器
            sample_input_ids: 用于预测的样本 input_ids
            sample_label: 真实标签（用于对比）
            log_steps: 每隔多少步打印一次预测
        """
        self.tokenizer = tokenizer
        self.sample_input_ids = sample_input_ids
        self.sample_label = sample_label
        self.log_steps = log_steps
    
    @override
    def on_step_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if state.global_step % self.log_steps != 0:
            return
        
        model = kwargs.get("model")
        if model is None:
            return
        
        # 设置为评估模式
        model.eval()
        
        try:
            with torch.no_grad():
                input_ids = torch.tensor([self.sample_input_ids], device=model.device)
                
                # 生成预测
                outputs = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=100,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                
                # 只取新生成的部分
                generated_ids = outputs[0, len(self.sample_input_ids):]
                prediction = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                
                logger.info_rank0(f"\n{'='*60}")
                logger.info_rank0(f"[Step {state.global_step}] 模型预测示例:")
                logger.info_rank0(f"【模型预测】: {prediction[:200]}")
                logger.info_rank0(f"【真实标签】: {self.sample_label[:200]}")
                logger.info_rank0(f"{'='*60}\n")
        except Exception as e:
            logger.warning_rank0(f"预测日志失败: {e}")
        finally:
            # 恢复训练模式
            model.train()


class PerTokenLossCallback(TrainerCallback):
    r"""把求和形式的训练 loss 换算为 per-token loss 并打印 / 写回日志。

    背景：transformers >= 4.46 修复 gradient_accumulation 行为后，部分场景
    （packing 路径 / num_items_in_batch 未被透传 / 新增词表 CPT-like 训练等）
    下，模型 forward 返回的 loss 实际上是「label token 上的交叉熵求和」，
    HF Trainer 直接把它当成 step loss 打到日志里，所以会看到几十万、几百万
    这种夸张数字（例如 1.4M）。本 callback 不改任何训练逻辑，只是按
    «tokens_per_step» 把日志里的 sum-loss 还原为 «per-token loss»，
    使曲线变得人类可读。

    Args:
        tokens_per_step: 每个 optimizer step 在「单卡」上参与 loss 的 label
            token 数。若未提供，则在 on_train_begin 时按以下公式估算：
                per_device_train_batch_size
              * gradient_accumulation_steps
              * cutoff_len (packing 路径下每条样本基本都是满长)
            其中 cutoff_len 通过 kwargs 里的 data_args / 训练参数推断，
            推断失败则跳过换算。
        also_overwrite_loss: 是否把日志里原始的 "loss" 字段一并替换为
            per-token loss（默认 False，仅新增 "loss_per_token" 字段，
            保留原始数值便于排查）。
    """

    def __init__(
        self,
        tokens_per_step: Optional[int] = None,
        cutoff_len: Optional[int] = None,
        also_overwrite_loss: bool = False,
    ) -> None:
        self.tokens_per_step = tokens_per_step
        self.cutoff_len = cutoff_len
        self.also_overwrite_loss = also_overwrite_loss
        self._announced = False
        # === LF-DIAG: ring buffer for anomaly detection ====================
        # Records (step, loss_sum, loss_per_token, grad_norm) of the last
        # logging events. Used to detect (a) precise loss==0, (b) grad_norm
        # constant across many steps (placeholder fallback signal).
        self._diag_ring: list[tuple[int, float, float, float]] = []
        self._diag_ring_size: int = 20
        self._diag_warned_grad_norm_const: bool = False

    def _infer_tokens_per_step(self, args: "TrainingArguments") -> Optional[int]:
        bs = int(getattr(args, "per_device_train_batch_size", 0) or 0)
        accum = int(getattr(args, "gradient_accumulation_steps", 1) or 1)
        cutoff_len = int(self.cutoff_len or 0)
        if not cutoff_len:
            cutoff_len = int(getattr(args, "max_seq_length", 0) or 0)
        if bs <= 0 or accum <= 0 or cutoff_len <= 0:
            return None
        return bs * accum * cutoff_len

    @override
    def on_train_begin(
        self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs
    ):
        if self.tokens_per_step is None:
            self.tokens_per_step = self._infer_tokens_per_step(args)

        if self.tokens_per_step and not self._announced:
            logger.info_rank0(
                f"[PerTokenLossCallback] enabled with tokens_per_step={self.tokens_per_step} "
                f"(per_device_bs={args.per_device_train_batch_size} * "
                f"grad_accum={args.gradient_accumulation_steps} * "
                f"cutoff_len≈{self.tokens_per_step // max(1, args.per_device_train_batch_size * args.gradient_accumulation_steps)}). "
                f"logged_loss_per_token ≈ raw_loss / tokens_per_step."
            )
            self._announced = True
        elif not self.tokens_per_step:
            logger.warning_rank0(
                "[PerTokenLossCallback] cannot infer tokens_per_step "
                "(missing per_device_train_batch_size / gradient_accumulation_steps / cutoff_len); "
                "per-token loss logging is disabled."
            )

    @override
    def on_log(
        self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs
    ):
        if not self.tokens_per_step:
            return
        if not state.log_history:
            return
        latest = state.log_history[-1]
        loss = latest.get("loss")
        if loss is None:
            return

        per_token = float(loss) / float(self.tokens_per_step)
        latest["loss_per_token"] = per_token
        latest["loss_sum"] = float(loss)
        if self.also_overwrite_loss:
            latest["loss"] = per_token

        if state.is_world_process_zero:
            step = int(latest.get("step", state.global_step))
            lr = float(latest.get("learning_rate", 0.0) or 0.0)
            epoch = float(latest.get("epoch", 0.0) or 0.0)
            logger.info_rank0(
                f"[per-token loss] step={step} "
                f"loss_sum={float(loss):.2f} "
                f"loss_per_token={per_token:.4f} "
                f"lr={lr:.3e} epoch={epoch:.3f}"
            )

            # === LF-DIAG: anomaly detection on logged metrics ==============
            # Detects (a) loss precisely zero (CE sum over all-IGNORE batch),
            # (b) loss outrageously large (sum-CE not normalized),
            # (c) grad_norm constant across a window (placeholder fallback,
            #     e.g. recurring √2 ≈ 1.4142135 we observed). Always-on; no
            #     env switch because these conditions are unambiguously bad.
            try:
                _gn_raw = latest.get("grad_norm", None)
                grad_norm_val = float(_gn_raw) if _gn_raw is not None else float("nan")
                self._diag_ring.append((step, float(loss), per_token, grad_norm_val))
                if len(self._diag_ring) > self._diag_ring_size:
                    self._diag_ring = self._diag_ring[-self._diag_ring_size :]

                anomalies: list[str] = []

                if float(loss) == 0.0:
                    anomalies.append("loss==0.0 exactly")
                if float(loss) > 1e7:
                    anomalies.append(f"loss>1e7 ({float(loss):.2e})")

                # grad_norm placeholder detection: ≥5 recent steps with range
                # < 1e-6 (covers √2 and any other static fallback value).
                if len(self._diag_ring) >= 5:
                    recent_gn = [
                        gn for _, _, _, gn in self._diag_ring[-5:]
                        if gn == gn  # filter NaN
                    ]
                    if len(recent_gn) >= 5:
                        gn_range = max(recent_gn) - min(recent_gn)
                        if gn_range < 1e-6 and not self._diag_warned_grad_norm_const:
                            anomalies.append(
                                f"grad_norm constant={recent_gn[-1]:.10f} "
                                f"over last 5 logs (range={gn_range:.2e}); "
                                "likely a Trainer/DeepSpeed placeholder fallback, "
                                "not the true global grad norm"
                            )
                            self._diag_warned_grad_norm_const = True

                if anomalies:
                    tail = self._diag_ring[-min(len(self._diag_ring), 10) :]
                    history = " | ".join(
                        f"step={s} loss={l:.2f} pt={pt:.4f} gn={gn:.6f}"
                        for s, l, pt, gn in tail
                    )
                    logger.warning_rank0(
                        f"[LF-DIAG][per-token loss] !! ANOMALY @ step={step} !! "
                        f"flags={anomalies}. recent={{ {history} }}"
                    )
            except Exception as _e:  # noqa: BLE001
                logger.warning_rank0(f"[LF-DIAG][per-token loss] diag failed: {_e}")
