#!/usr/bin/env python3
"""
评估脚本：支持两种评估方法
1. 本地路径加载模型（使用 vLLM 直接加载）
2. 本地 API 调用（通过 OpenAI 兼容接口）

输出格式: <think>...</think>\n结论：建议管控/暂不管控
输出字段:
  - mchid: 商户ID
  - conclusion: 结论（建议管控/暂不管控）
  - prob_1: 答案首 token 位置 P("建")（始终是"建议管控"方向的原始概率）
  - prob_1_norm: P("建") / (P("建") + P("暂"))（二分归一化）
  - top20_hit: 对立候选是否在 top20 中找到

截断规则：按 <风险序列>/<近期序列> 内的事件进行智能压缩，保证输入不超过指定 tokens
"""

import os
import re
import json
import math
import argparse
import time
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple, List, Dict, Any
from tqdm import tqdm
import requests

LABEL_CONTROL = "建议管控"
LABEL_NO_CONTROL = "暂不管控"

# ========== 序列压缩相关常量 ==========
_RISK_SEQ_START = "<风险序列>"
_RISK_SEQ_END = "</风险序列>"
_RECENT_SEQ_START = "<近期序列>"
_RECENT_SEQ_END = "</近期序列>"
_EVENT_START_RE = re.compile(r"<时间:")
_SEQ_ALIGN_RECENT_MIN_EVENTS = 100
# 风险序列最少保留条数：避免把 <风险序列></风险序列> 掏空，造成 OOD 输入
_SEQ_ALIGN_RISK_MIN_EVENTS = 50


# ========== 序列压缩工具函数 ==========
def _extract_tagged_event_block(
    content: str, start_tag: str, end_tag: str, event_start_re: re.Pattern[str]
) -> Tuple[Optional[str], List[str], Optional[str]]:
    """提取标签内的事件块"""
    start_idx = content.find(start_tag)
    if start_idx < 0:
        return None, [], None

    end_idx = content.find(end_tag, start_idx)
    if end_idx < 0:
        return None, [], None

    block_end = end_idx + len(end_tag)
    prefix = content[:start_idx]
    suffix = content[block_end:]
    block_inner = content[start_idx + len(start_tag):end_idx].strip()
    if not block_inner:
        return prefix, [], suffix

    event_starts = [m.start() for m in event_start_re.finditer(block_inner)]
    if not event_starts:
        return prefix, [block_inner], suffix

    events = []
    for idx, event_start in enumerate(event_starts):
        next_start = event_starts[idx + 1] if idx + 1 < len(event_starts) else len(block_inner)
        event_text = block_inner[event_start:next_start].strip()
        if event_text:
            events.append(event_text)

    return prefix, events, suffix


def _extract_risk_block(content: str) -> Tuple[Optional[str], List[str], Optional[str]]:
    return _extract_tagged_event_block(content, _RISK_SEQ_START, _RISK_SEQ_END, _EVENT_START_RE)


def _extract_recent_block(content: str) -> Tuple[Optional[str], List[str], Optional[str]]:
    return _extract_tagged_event_block(content, _RECENT_SEQ_START, _RECENT_SEQ_END, _EVENT_START_RE)


def _replace_tagged_event_block(content: str, start_tag: str, end_tag: str, events: List[str]) -> str:
    """替换标签内的事件块"""
    prefix, _, suffix = _extract_tagged_event_block(content, start_tag, end_tag, _EVENT_START_RE)
    if prefix is None or suffix is None:
        return content

    replacement = ""
    if events:
        replacement = start_tag + "\n" + " ".join(events) + "\n" + end_tag

    merged = prefix + replacement + suffix
    merged = re.sub(r"\n{3,}", "\n\n", merged)
    return merged


def _replace_risk_block(content: str, risk_events: List[str]) -> str:
    return _replace_tagged_event_block(content, _RISK_SEQ_START, _RISK_SEQ_END, risk_events)


def _replace_recent_block(content: str, recent_events: List[str]) -> str:
    return _replace_tagged_event_block(content, _RECENT_SEQ_START, _RECENT_SEQ_END, recent_events)


class TextTruncator:
    """智能文本截断器，按事件序列进行压缩。

    token 预算采用 **精确计数**：对"apply_chat_template 之后的完整 prompt"做 encode，
    和推理时真正送进模型的 token 序列完全一致，避免由 system+user 拼裂字符串估算带来的偏差。

    truncate() 返回三元组：(压缩后 user_content, truncated, reason)
      - truncated: True 表示对原文做过任意形式的压缩或硬截断
      - reason: "none" | "compress_recent" | "compress_risk" | "hard_cut"
    """

    def __init__(self, tokenizer, max_tokens: int):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

    # ---------------- 精确 token 计数 ---------------- #
    def count_prompt_tokens(self, user_content: str) -> int:
        """对 chat template 之后的完整 prompt 做精确 token 计数。"""
        messages = build_messages(user_content)
        templated = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # chat template 的输出本身已经包含 special tokens（如 <|im_start|>），
        # 这里用 add_special_tokens=False 不再追加，保证与真实送入模型的 token 数一致。
        return len(self.tokenizer.encode(templated, add_special_tokens=False))

    def _compress_recent_sequence(
        self, content: str, budget: int, min_keep: int = _SEQ_ALIGN_RECENT_MIN_EVENTS
    ) -> Tuple[str, bool]:
        """压缩近期序列，二分查找最大保留事件数。返回 (压缩后内容, 是否压到预算内)。

        min_keep: 保留下限（默认 100）。若仍超预算，可由上层显式传入 0 放开下限。
        """
        if _RECENT_SEQ_START not in content or _RECENT_SEQ_END not in content:
            return content, self.count_prompt_tokens(content) <= budget

        prefix, recent_events, suffix = _extract_recent_block(content)
        if prefix is None or suffix is None or len(recent_events) <= min_keep:
            return content, self.count_prompt_tokens(content) <= budget

        lo, hi = min_keep, len(recent_events)
        best_content = content
        best_fits = False

        while lo <= hi:
            mid = (lo + hi) // 2
            candidate_content = _replace_recent_block(content, recent_events[-mid:])
            candidate_tokens = self.count_prompt_tokens(candidate_content)

            if candidate_tokens <= budget:
                best_content = candidate_content
                best_fits = True
                lo = mid + 1
            else:
                hi = mid - 1

        if not best_fits:
            candidate_content = _replace_recent_block(content, recent_events[-min_keep:] if min_keep > 0 else [])
            best_content = candidate_content
            best_fits = self.count_prompt_tokens(candidate_content) <= budget

        return best_content, best_fits

    def _compress_risk_sequence(
        self, content: str, budget: int
    ) -> Tuple[str, bool]:
        """压缩风险序列，二分查找最大保留事件数。返回 (压缩后内容, 是否压到预算内)。

        保留下限 _SEQ_ALIGN_RISK_MIN_EVENTS，避免 <风险序列> 被掏空造成 OOD。
        即便压到下限仍超预算，也返回"压到下限后的内容"（fits=False），由上层继续兜底。
        """
        if _RISK_SEQ_START not in content or _RISK_SEQ_END not in content:
            return content, self.count_prompt_tokens(content) <= budget

        prefix, risk_events, suffix = _extract_risk_block(content)
        if prefix is None or suffix is None or not risk_events:
            return content, self.count_prompt_tokens(content) <= budget

        min_keep = min(_SEQ_ALIGN_RISK_MIN_EVENTS, len(risk_events))
        lo, hi = min_keep, len(risk_events)
        best_content = content
        best_fits = False

        while lo <= hi:
            mid = (lo + hi) // 2
            candidate_content = _replace_risk_block(content, risk_events[:mid])
            candidate_tokens = self.count_prompt_tokens(candidate_content)

            if candidate_tokens <= budget:
                best_content = candidate_content
                best_fits = True
                lo = mid + 1
            else:
                hi = mid - 1

        if not best_fits:
            # 连 min_keep 都放不下：返回压到 min_keep 的内容（标签结构完整），由上层 hard_cut 兜底
            best_content = _replace_risk_block(content, risk_events[:min_keep])
            best_fits = self.count_prompt_tokens(best_content) <= budget

        return best_content, best_fits

    def _hard_cut(self, user_content: str) -> str:
        """兜底硬截断：按 user_content 的 token 逐步减半/二分到预算内。

        直接按"user_content 裸 token 数"去 decode 会丢 chat-template 开销，
        所以这里仍用精确计数 count_prompt_tokens 做停机条件。
        """
        user_tokens = self.tokenizer.encode(user_content, add_special_tokens=False)
        lo, hi = 0, len(user_tokens)
        best_text = ""
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate_text = self.tokenizer.decode(user_tokens[:mid], skip_special_tokens=True)
            if self.count_prompt_tokens(candidate_text) <= self.max_tokens:
                best_text = candidate_text
                lo = mid + 1
            else:
                hi = mid - 1
        return best_text

    def truncate(self, user_content: str) -> Tuple[str, bool, str]:
        """智能截断。返回 (压缩后内容, 是否截断, 原因)。

        策略（与训练侧 compress_risk_sequence_first=True 行为对齐）：
          1. 原文即符合预算 -> 不截断
          2. 压缩风险序列（保底 50 条，避免掏空造成 OOD）
          3. 压缩近期序列（保底 100 条）
          4. 兜底硬截断（token 二分）
        """
        if self.count_prompt_tokens(user_content) <= self.max_tokens:
            return user_content, False, "none"

        # 第一步：压缩风险序列（保底 50 条）
        content, fits = self._compress_risk_sequence(user_content, self.max_tokens)
        if fits:
            return content, True, "compress_risk"

        # 第二步：压缩近期序列（保底 100 条）
        content, fits = self._compress_recent_sequence(content, self.max_tokens)
        if fits:
            return content, True, "compress_recent"

        # 第三步：兜底硬截断
        content = self._hard_cut(content)
        return content, True, "hard_cut"


# ========== 数据加载 ==========
def load_data(input_file: str) -> List[Dict]:
    records = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def build_messages(text: str) -> List[Dict[str, str]]:
    # 训练数据未使用 system 消息，推理仅传 user，避免 chat template 注入不一致的 system
    return [
        {"role": "user", "content": text},
    ]


# ========== 结果解析 ==========
def parse_conclusion(output_text: str) -> Optional[str]:
    """从模型输出中解析结论：建议管控 / 暂不管控"""
    match = re.search(r"结论[：:]\s*(建议管控|暂不管控)", output_text)
    if match:
        return match.group(1)
    return None


def compute_prob(logprobs_content: List[Dict], conclusion: str) -> Dict[str, Any]:
    """
    从 logprobs 中计算"建议管控"方向的概率（无论采样结论是什么，存的都是"建议管控"的概率）。

    prob_1:      P("建")                             —— tok1 位置"建"的原始概率
    prob_1_norm: P("建") / (P("建") + P("暂"))       —— tok1 的二分归一化
    top20_hit:   对立候选是否在 top20 中出现
    """
    result = {
        "prob_1": None,
        "prob_1_norm": None,
        "top20_hit": False,
    }
    
    if not logprobs_content:
        return result

    # 重建 token 串
    tokens = [t["token"] for t in logprobs_content]
    full_text = "".join(tokens)

    # 优先在 </think> 之后查找"结论"
    think_end = full_text.find("</think>")
    search_start_char = (think_end + len("</think>")) if think_end != -1 else 0

    idx = full_text.find("结论", search_start_char)
    if idx == -1:
        return result

    # 找到字符位置对应的 token 下标
    char_count = 0
    conclusion_tok_idx = -1
    for i, tok in enumerate(tokens):
        if char_count >= idx:
            conclusion_tok_idx = i
            break
        char_count += len(tok)

    if conclusion_tok_idx == -1:
        return result

    # 从"结论"位置往后找答案首 token
    target_tok_idx = -1
    for i in range(conclusion_tok_idx, min(conclusion_tok_idx + 20, len(tokens))):
        t = tokens[i]
        if t == "建" or "建议" in t or t.startswith("暂"):
            target_tok_idx = i
            break

    if target_tok_idx == -1:
        return result

    # ---- tok1：答案首 token ----
    entry1 = logprobs_content[target_tok_idx]
    sampled_lp1 = entry1["logprob"]
    top_lps1 = entry1.get("top_logprobs", [])
    sampled_tok1 = entry1["token"]

    if sampled_tok1 == "建" or "建议" in sampled_tok1:
        sampled_class = LABEL_CONTROL
    elif sampled_tok1.startswith("暂"):
        sampled_class = LABEL_NO_CONTROL
    else:
        return result

    # ---- tok1：取"建"(=建议管控) 和"暂"(=暂不管控) 的 logprob ----
    # 命名说明：yes = "建议管控" 方向（P(建)），no = "暂不管控" 方向（P(暂)）
    yes_lp1 = sampled_lp1 if sampled_class == LABEL_CONTROL else None
    no_lp1 = sampled_lp1 if sampled_class == LABEL_NO_CONTROL else None
    min_top_lp = None

    for tp in top_lps1:
        tok = tp["token"]
        lp = tp["logprob"]
        if min_top_lp is None or lp < min_top_lp:
            min_top_lp = lp
        if tok == "建" or "建议" in tok:
            if yes_lp1 is None or lp > yes_lp1:
                yes_lp1 = lp
        elif tok.startswith("暂"):
            if no_lp1 is None or lp > no_lp1:
                no_lp1 = lp

    # top20_hit: 归一化所需的对立候选是否在 top20 中出现
    if sampled_class == LABEL_CONTROL:
        result["top20_hit"] = no_lp1 is not None and no_lp1 != sampled_lp1
    else:
        result["top20_hit"] = yes_lp1 is not None

    if yes_lp1 is None:
        if min_top_lp is None:
            return result
        yes_lp1 = min_top_lp
    if no_lp1 is None:
        if min_top_lp is None:
            return result
        no_lp1 = min_top_lp

    # prob_1 / prob_1_norm —— 始终是"建议管控"方向
    prob_1 = math.exp(yes_lp1)
    prob_1_no = math.exp(no_lp1)
    result["prob_1"] = prob_1
    total_1 = prob_1 + prob_1_no
    if total_1 > 0:
        result["prob_1_norm"] = prob_1 / total_1

    return result


# ========== 评估器基类 ==========
class BaseEvaluator:
    """评估器基类"""

    def __init__(
        self,
        max_new_tokens: int,
        top_logprobs: int,
        max_input_tokens: int,
        mark_truncated: bool = True,
    ):
        self.max_new_tokens = max_new_tokens
        self.top_logprobs = top_logprobs
        self.max_input_tokens = max_input_tokens
        self.mark_truncated = mark_truncated
        self.truncator = None

    def truncate_input(self, text: str) -> Tuple[str, bool, str]:
        """截断输入。返回 (处理后文本, 是否截断, 原因)。"""
        if self.truncator:
            return self.truncator.truncate(text)
        return text, False, "none"

    def evaluate_single(self, record: Dict) -> Dict:
        """评估单条记录，子类实现"""
        raise NotImplementedError


class LocalModelEvaluator(BaseEvaluator):
    """本地模型加载评估器"""
    
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int,
        max_new_tokens: int,
        top_logprobs: int,
        max_input_tokens: int,
        mark_truncated: bool = True,
    ):
        super().__init__(max_new_tokens, top_logprobs, max_input_tokens, mark_truncated)
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self._init_model()
    
    def _init_model(self):
        """初始化 vLLM 模型"""
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
        
        print(f"[LocalModel] 加载模型: {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.llm = LLM(
            model=self.model_path,
            tensor_parallel_size=self.tensor_parallel_size,
            trust_remote_code=True,
            max_model_len=self.max_input_tokens + self.max_new_tokens,
        )
        self.sampling_params = SamplingParams(
            max_tokens=self.max_new_tokens,
            temperature=0.0,
            logprobs=self.top_logprobs,
            repetition_penalty=1.05,  # 抑制重复循环（RL模型偶发）
        )
        self.truncator = TextTruncator(self.tokenizer, self.max_input_tokens)
        print("[LocalModel] 模型加载完成")
    
    def _apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """应用 chat template"""
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    
    def evaluate_batch(self, records: List[Dict]) -> List[Dict]:
        """批量评估"""
        prompts = []
        trunc_infos = []  # 每条 (truncated, reason)
        for rec in records:
            text, truncated, reason = self.truncate_input(rec["text"])
            trunc_infos.append((truncated, reason))
            messages = build_messages(text)
            prompt = self._apply_chat_template(messages)
            prompts.append(prompt)

        outputs = self.llm.generate(prompts, self.sampling_params)

        results = []
        for rec, output, (truncated, reason) in zip(records, outputs, trunc_infos):
            output_text = output.outputs[0].text
            logprobs_data = output.outputs[0].logprobs

            logprobs_content = []
            if logprobs_data:
                for token_logprobs in logprobs_data:
                    for token_id, logprob_obj in token_logprobs.items():
                        entry = {
                            "token": logprob_obj.decoded_token,
                            "logprob": logprob_obj.logprob,
                            "top_logprobs": []
                        }
                        for tid, lp in token_logprobs.items():
                            entry["top_logprobs"].append({
                                "token": lp.decoded_token,
                                "logprob": lp.logprob
                            })
                        logprobs_content.append(entry)
                        break

            conclusion = parse_conclusion(output_text)
            prob_result = compute_prob(logprobs_content, conclusion)

            result = {
                "mchid": rec["mchid"],
                "conclusion": conclusion,
                "prob_1": prob_result["prob_1"],
                "prob_1_norm": prob_result["prob_1_norm"],
                "top20_hit": prob_result["top20_hit"],
            }
            if self.mark_truncated:
                result["truncated"] = truncated
                result["truncated_reason"] = reason
            if conclusion == LABEL_CONTROL:
                result["model_output"] = output_text
            if conclusion is None:
                result["output_tail"] = output_text[-200:]
            results.append(result)

        return results


class APIEvaluator(BaseEvaluator):
    """API 调用评估器"""
    
    def __init__(
        self,
        base_url: str,
        model: str,
        max_new_tokens: int,
        top_logprobs: int,
        max_input_tokens: int,
        max_retries: int = 3,
        mark_truncated: bool = True,
    ):
        super().__init__(max_new_tokens, top_logprobs, max_input_tokens, mark_truncated)
        self.base_url = base_url
        self.model = model
        self.max_retries = max_retries
        self._init_tokenizer()
    
    def _init_tokenizer(self):
        """初始化 tokenizer（用于截断）"""
        from transformers import AutoTokenizer
        
        try:
            r = requests.get(f"{self.base_url}/models", timeout=10)
            models = [m["id"] for m in r.json().get("data", [])]
            print(f"[API] 服务可达，可用模型: {models}")
            if models:
                self.model = models[0]
                print(f"[API] 使用模型: {self.model}")
        except Exception as e:
            print(f"[API] 无法访问 {self.base_url}/models: {e}")
        
        print(f"[API] 加载 tokenizer: {self.model}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)
        self.truncator = TextTruncator(self.tokenizer, self.max_input_tokens)
    
    def evaluate_single(self, record: Dict) -> Dict:
        """评估单条记录"""
        mchid = record["mchid"]
        text, truncated, reason = self.truncate_input(record["text"])
        messages = build_messages(text)

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_new_tokens,
            "temperature": 0.0,
            "top_logprobs": self.top_logprobs,
            "logprobs": True,
            "repetition_penalty": 1.05,  # 抑制重复循环（RL模型偶发）
        }

        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    timeout=300,
                )
                if resp.status_code == 400 and attempt < self.max_retries - 1:
                    # 服务端 400 一般是长度越界：用 tokenizer 精确收紧预算后重算，避免按 char 估算。
                    current_text = messages[-1]["content"]
                    current_tokens = self.truncator.count_prompt_tokens(current_text)
                    # 每次把预算收到当前 prompt token 数的 3/4，并不超过全局 max_input_tokens
                    new_budget = min(self.max_input_tokens, max(256, current_tokens * 3 // 4))
                    tighter = TextTruncator(self.tokenizer, new_budget)
                    new_text, _, sub_reason = tighter.truncate(current_text)
                    messages[-1]["content"] = new_text
                    payload["messages"] = messages
                    truncated = True
                    if reason == "none":
                        reason = f"server_400_retry:{sub_reason}"
                    continue
                resp.raise_for_status()
                data = resp.json()

                choice = data["choices"][0]
                output_text = choice["message"]["content"]
                logprobs_content = (
                    choice.get("logprobs", {}) or {}
                ).get("content", [])

                conclusion = parse_conclusion(output_text)
                prob_result = compute_prob(logprobs_content, conclusion)

                result = {
                    "mchid": mchid,
                    "conclusion": conclusion,
                    "prob_1": prob_result["prob_1"],
                    "prob_1_norm": prob_result["prob_1_norm"],
                    "top20_hit": prob_result["top20_hit"],
                }
                if self.mark_truncated:
                    result["truncated"] = truncated
                    result["truncated_reason"] = reason
                if conclusion == LABEL_CONTROL:
                    result["model_output"] = output_text
                if conclusion is None:
                    result["output_tail"] = output_text[-200:]
                return result

            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    err_result = {
                        "mchid": mchid,
                        "conclusion": None,
                        "prob_1": None,
                        "prob_1_norm": None,
                        "top20_hit": False,
                        "error": str(e),
                    }
                    if self.mark_truncated:
                        err_result["truncated"] = truncated
                        err_result["truncated_reason"] = reason
                    return err_result


def main():
    parser = argparse.ArgumentParser(description="评估脚本：支持本地模型加载和 API 调用两种方式")
    
    # 评估方式
    parser.add_argument(
        "--mode",
        choices=["local", "api"],
        required=True,
        help="评估方式: local=本地加载模型, api=调用 API"
    )
    
    # 必需参数
    parser.add_argument("--model", required=True, help="模型路径")
    parser.add_argument("--input", required=True, help="输入文件")
    parser.add_argument("--output", required=True, help="输出文件")
    
    # 通用可选参数
    parser.add_argument("--max_new_tokens", type=int, default=6000, help="最大生成 token 数 (默认: 6000)")
    parser.add_argument("--max_input_tokens", type=int, default=8192, help="输入截断长度 (默认: 8192)")
    parser.add_argument("--top_logprobs", type=int, default=20, help="top logprobs 数量 (默认: 20)")
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 条")
    parser.add_argument(
        "--start_index",
        type=int,
        default=None,
        help="数据分片起始下标（含），用于把同一文件分到多卡并行跑；默认从 0 开始",
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default=None,
        help="数据分片结束下标（不含），默认到末尾；与 --start_index 配合切片",
    )
    parser.add_argument("--resume", action="store_true", help="断点续跑")
    parser.add_argument(
        "--mark_truncated",
        type=lambda x: str(x).lower() in ("1", "true", "yes", "y"),
        default=True,
        help="是否在输出中添加 truncated / truncated_reason 字段 (默认: True)",
    )
    
    # API 模式参数
    parser.add_argument("--base_url", default="http://localhost:8000/v1", help="API 服务地址")
    parser.add_argument("--workers", type=int, default=128, help="并发请求数 (默认: 128)")
    
    # 本地模式参数
    parser.add_argument("--tp", type=int, default=1, help="tensor parallel size (默认: 1)")
    parser.add_argument("--batch_size", type=int, default=32, help="本地推理批大小 (默认: 32)")
    
    args = parser.parse_args()

    # 断点续跑
    done_mchids = set()
    if args.resume and os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done_mchids.add(str(json.loads(line)["mchid"]))
                except Exception:
                    pass
        print(f"[Resume] 已处理 {len(done_mchids)} 条，跳过")

    # 加载数据
    records = load_data(args.input)
    print(f"总数据量: {len(records)}")

    # 数据分片（多卡并行跑同一份文件时使用）
    if args.start_index is not None or args.end_index is not None:
        total = len(records)
        start = args.start_index if args.start_index is not None else 0
        end = args.end_index if args.end_index is not None else total
        # 容错：负数视作从尾部偏移；越界 clamp
        if start < 0:
            start = max(0, total + start)
        if end < 0:
            end = max(0, total + end)
        start = max(0, min(start, total))
        end = max(start, min(end, total))
        records = records[start:end]
        print(f"[Shard] start_index={start}, end_index={end}, 分片后数据量: {len(records)}")

    if args.resume:
        records = [r for r in records if str(r["mchid"]) not in done_mchids]
        print(f"待处理数据量: {len(records)}")

    if not records:
        print("没有待处理数据，退出。")
        return

    if args.limit is not None:
        records = records[:args.limit]
        print(f"[Limit] 只处理前 {len(records)} 条")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    write_mode = "a" if args.resume else "w"

    if args.mode == "local":
        evaluator = LocalModelEvaluator(
            model_path=args.model,
            tensor_parallel_size=args.tp,
            max_new_tokens=args.max_new_tokens,
            top_logprobs=args.top_logprobs,
            max_input_tokens=args.max_input_tokens,
            mark_truncated=args.mark_truncated,
        )
        
        with open(args.output, write_mode, encoding="utf-8") as out_f:
            for i in tqdm(range(0, len(records), args.batch_size), desc="评估进度"):
                batch = records[i:i + args.batch_size]
                results = evaluator.evaluate_batch(batch)
                for result in results:
                    out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()
    
    else:
        evaluator = APIEvaluator(
            base_url=args.base_url,
            model=args.model,
            max_new_tokens=args.max_new_tokens,
            top_logprobs=args.top_logprobs,
            max_input_tokens=args.max_input_tokens,
            mark_truncated=args.mark_truncated,
        )
        
        with open(args.output, write_mode, encoding="utf-8") as out_f:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(evaluator.evaluate_single, rec): rec
                    for rec in records
                }
                for future in tqdm(as_completed(futures), total=len(futures), desc="评估进度"):
                    result = future.result()
                    out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    out_f.flush()

    print(f"评估完成，结果保存至: {args.output}")


if __name__ == "__main__":
    main()
