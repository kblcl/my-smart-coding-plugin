"""
train_lora.py — LoRA 微调训练脚本（模块6）

功能：
  1. 加载 prepare_dataset.py 生成的 instruction / chatml 格式数据
  2. 支持多种基座模型（CodeLlama, DeepSeek Coder, Qwen2.5-Coder, StarCoder2）
  3. 使用 PEFT LoRA 进行参数高效微调
  4. 训练过程记录 loss，支持断点续训
  5. 保存 LoRA adapter 权重

接口规范：
  - 构造函数接收 config 字典（符合接口规范文档）
  - train() 返回包含 adapter_path, train_loss, val_loss, train_runtime_sec, total_steps, error 的字典
  - evaluate_adapter(adapter_path) 返回评估结果字典

用法：
  # Python API 方式调用（符合接口规范）
  from finetune.train_lora import LoRATrainer
  config = {
      "base_model": "Qwen2.5-Coder-7B-Instruct",
      "train_dataset_path": "./data/train.jsonl",
      "val_dataset_path": "./data/val.jsonl",
      "output_dir": "./output/lora",
      "num_epochs": 3,
      "batch_size": 4,
  }
  trainer = LoRATrainer(config)
  result = trainer.train()

依赖安装（如未安装）：
  pip install torch transformers peft datasets accelerate tensorboard
"""

import argparse
import json
import math
import os

# ============================================================================
# 强制离线模式：必须在 import transformers / huggingface_hub 之前设置！
# 与 model_server.py 同样的修复——基座权重已完整缓存，训练时无需联网。
# 不加的话，新版 transformers 加载模型/tokenizer 会联网校验，遇代理/镜像
# 抖动直接抛 httpx 错误导致训练启动即崩。
# ============================================================================
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union


# ======================================================================
# 配置
# ======================================================================

BASE_MODEL_REGISTRY = {
    # model_id 用于 transformers.AutoModelForCausalLM.from_pretrained
    # target_modules 是 LoRA 要注入的线性层
    # template 是 prompt 模板（instruction 格式用）
    "codellama-7b": {
        "model_id": "codellama/CodeLlama-7b-hf",
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "template": (
            "[INST] Write code to solve the following problem:\n\n"
            "{instruction}\n{input}[/INST]\n{output}"
        ),
        "template_no_input": (
            "[INST] Write code to solve the following problem:\n\n"
            "{instruction}[/INST]\n{output}"
        ),
    },
    "codellama-13b": {
        "model_id": "codellama/CodeLlama-13b-hf",
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "template": (
            "[INST] Write code to solve the following problem:\n\n"
            "{instruction}\n{input}[/INST]\n{output}"
        ),
        "template_no_input": (
            "[INST] Write code to solve the following problem:\n\n"
            "{instruction}[/INST]\n{output}"
        ),
    },
    "deepseek-coder-1.3b": {
        "model_id": "deepseek-ai/deepseek-coder-1.3b-instruct",
        "target_modules": ["q_proj", "v_proj"],
        "template": (
            "### Instruction:\n{instruction}\n{input}\n### Response:\n{output}"
        ),
        "template_no_input": (
            "### Instruction:\n{instruction}\n### Response:\n{output}"
        ),
    },
    "deepseek-coder-6.7b": {
        "model_id": "deepseek-ai/deepseek-coder-6.7b-instruct",
        "target_modules": ["q_proj", "v_proj"],
        "template": (
            "### Instruction:\n{instruction}\n{input}\n### Response:\n{output}"
        ),
        "template_no_input": (
            "### Instruction:\n{instruction}\n### Response:\n{output}"
        ),
    },
    "qwen2.5-coder-1.5b": {
        "model_id": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "template": (
            "<|im_start|>user\n{instruction}\n{input}<|im_end|>\n"
            "<|im_start|>assistant\n{output}<|im_end|>"
        ),
        "template_no_input": (
            "<|im_start|>user\n{instruction}<|im_end|>\n"
            "<|im_start|>assistant\n{output}<|im_end|>"
        ),
    },
    "qwen2.5-coder-7b": {
        "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "template": (
            "<|im_start|>user\n{instruction}\n{input}<|im_end|>\n"
            "<|im_start|>assistant\n{output}<|im_end|>"
        ),
        "template_no_input": (
            "<|im_start|>user\n{instruction}<|im_end|>\n"
            "<|im_start|>assistant\n{output}<|im_end|>"
        ),
    },
    "qwen2.5-coder-7b-instruct": {  # 新增：规范中的默认模型名
        "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "template": (
            "<|im_start|>user\n{instruction}\n{input}<|im_end|>\n"
            "<|im_start|>assistant\n{output}<|im_end|>"
        ),
        "template_no_input": (
            "<|im_start|>user\n{instruction}<|im_end|>\n"
            "<|im_start|>assistant\n{output}<|im_end|>"
        ),
    },
    "qwen2.5-0.5b-instruct": {  # 新增：弱代码能力基线，LoRA提升空间大
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "template": (
            "<|im_start|>user\n{instruction}\n{input}<|im_end|>\n"
            "<|im_start|>assistant\n{output}<|im_end|>"
        ),
        "template_no_input": (
            "<|im_start|>user\n{instruction}<|im_end|>\n"
            "<|im_start|>assistant\n{output}<|im_end|>"
        ),
    },
    "starcoder2-3b": {
        "model_id": "bigcode/starcoder2-3b",
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "template": (
            "<minimax:tool_call>user\n{instruction}\n{input}\n</minimax:tool_call>\n"
            "<minimax:tool_call>assistant\n{output}\n</minimax:tool_call>"
        ),
        "template_no_input": (
            "<minimax:tool_call>user\n{instruction}\n</minimax:tool_call>\n"
            "<minimax:tool_call>assistant\n{output}\n</minimax:tool_call>"
        ),
    },
    "starcoder2-7b": {
        "model_id": "bigcode/starcoder2-7b",
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "template": (
            "<minimax:tool_call>user\n{instruction}\n{input}\n<minimax:tool_call>assistant\n{output}\n</minimax:tool_call>"
        ),
        "template_no_input": (
            "<minimax:tool_call>user\n{instruction}\n<minimax:tool_call>assistant\n{output}\n</minimax:tool_call>"
        ),
    },
}

CHATML_MODELS = {"qwen2.5-coder-1.5b", "qwen2.5-coder-7b", "qwen2.5-coder-7b-instruct", "qwen2.5-0.5b-instruct", "codellama-7b", "codellama-13b"}


@dataclass
class TrainingConfig:
    """训练超参数 & 路径配置（与接口规范对齐）"""

    # ── Model ─────────────────────────────────────────────
    base_model: str = "Qwen2.5-Coder-7B-Instruct"  # 规范默认值
    model_path: Optional[str] = None  # 规范：本地模型路径

    # ── LoRA ──────────────────────────────────────────────
    lora_rank: int = 8  # 规范：LoRA rank（别名 lora_r）
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    load_adapter: str = ""  # 新增：从已有 adapter 继续训练（分层微调用）

    # ── Data ──────────────────────────────────────────────
    train_dataset_path: str = ""  # 规范必填
    val_dataset_path: str = ""    # 规范必填
    data_format: str = "instruction"  # "instruction" | "chatml"

    # ── Training ──────────────────────────────────────────
    output_dir: str = "./output/lora_checkpoints"  # 规范必填
    num_epochs: int = 3
    batch_size: int = 4  # 规范
    gradient_accumulation_steps: int = 8  # 规范默认值
    learning_rate: float = 0.0002  # 规范默认值
    warmup_ratio: float = 0.03
    max_seq_length: int = 2048
    fp16: bool = True  # 规范
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 3

    # ── Misc ──────────────────────────────────────────────
    cache_dir: Optional[str] = None
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    seed: int = 42
    use_fast_tokenizer: bool = True
    num_workers: int = 0

    def __post_init__(self):
        # 同步 lora_r 和 lora_rank
        self.lora_r = self.lora_rank

        # 自动切换为 chatml（Qwen 系列）
        if self.base_model in CHATML_MODELS and self.data_format == "instruction":
            self.data_format = "chatml"


# ======================================================================
# 数据加载与格式化
# ======================================================================


def _load_jsonl(path: str) -> List[Dict]:
    """加载 JSONL 文件，返回字典列表"""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def _format_instruction(
    sample: Dict,
    template: str,
    template_no_input: str,
) -> str:
    """将 Alpaca instruction 格式的单条样本格式化为文本。"""
    instruction = sample.get("instruction", "").strip()
    inp = sample.get("input", "").strip()
    output = sample.get("output", "").strip()

    if inp:
        text = template.format(instruction=instruction, input=inp, output=output)
    else:
        text = template_no_input.format(instruction=instruction, output=output)
    return text


def _format_chatml(sample: Dict, tokenizer) -> str:
    """将 ChatML 格式样本转为文本（使用 tokenizer.apply_chat_template）。"""
    messages = sample.get("messages", [])
    if not messages:
        return ""
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return text


def _tokenize_instruction(
    sample: Dict,
    tokenizer,
    template: str,
    template_no_input: str,
    max_length: int,
) -> Dict:
    """
    对 instruction 格式的样本进行 tokenize。
    返回的 labels 中，prompt 部分用 -100 遮盖（不参与 loss 计算）。
    """
    instruction = sample.get("instruction", "").strip()
    inp = sample.get("input", "").strip()
    output = sample.get("output", "").strip()

    if inp:
        prompt_text = template.split("{output}")[0].format(
            instruction=instruction, input=inp
        )
        full_text = template.format(instruction=instruction, input=inp, output=output)
    else:
        prompt_text = template_no_input.split("{output}")[0].format(
            instruction=instruction
        )
        full_text = template_no_input.format(instruction=instruction, output=output)

    # Tokenize 全文本
    full_tokens = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )

    # Tokenize prompt 部分
    prompt_tokens = tokenizer(
        prompt_text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )

    input_ids = full_tokens["input_ids"]
    prompt_len = len(prompt_tokens["input_ids"])
    labels = [-100] * prompt_len + input_ids[prompt_len:]

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def _tokenize_chatml(
    sample: Dict,
    tokenizer,
    max_length: int,
) -> Dict:
    """
    对 ChatML 格式样本进行 tokenize。
    labels 中所有非 assistant 回复的 token 用 -100 遮盖。
    """
    messages = sample.get("messages", [])
    if not messages:
        return {"input_ids": [], "attention_mask": [], "labels": []}

    assistant_indices = [
        i for i, m in enumerate(messages) if m.get("role") == "assistant"
    ]
    if not assistant_indices:
        return {"input_ids": [], "attention_mask": [], "labels": []}

    last_assistant_idx = assistant_indices[-1]
    prompt_messages = messages[:last_assistant_idx]
    full_messages = messages

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False
    )

    prompt_ids = tokenizer(
        prompt_text, truncation=True, max_length=max_length, padding=False
    )["input_ids"]
    full_ids = tokenizer(
        full_text, truncation=True, max_length=max_length, padding=False
    )["input_ids"]

    prompt_len = len(prompt_ids)
    labels = ([-100] * prompt_len) + full_ids[prompt_len:]

    if len(full_ids) > max_length:
        full_ids = full_ids[:max_length]
        labels = labels[:max_length]

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


# ======================================================================
# LoRATrainer 主类（符合接口规范）
# ======================================================================


class LoRATrainer:
    """
    LoRA 微调训练器（符合接口规范文档 V2.0）

    构造函数接收 config 字典，train() 返回规范格式的字典。
    支持 evaluate_adapter() 方法评估 adapter 效果。

    用法（符合接口规范）:
        config = {
            "base_model": "Qwen2.5-Coder-7B-Instruct",
            "train_dataset_path": "./data/train.jsonl",
            "val_dataset_path": "./data/val.jsonl",
            "output_dir": "./output/lora",
            "num_epochs": 3,
            "batch_size": 4,
        }
        trainer = LoRATrainer(config)
        result = trainer.train()
    """

    def __init__(self, config: Union[Dict, TrainingConfig, None] = None):
        """
        初始化训练器（符合接口规范）。

        Args:
            config: 训练配置字典，包含以下字段（均为可选，有默认值）：
                - base_model: 基座模型，默认 "Qwen2.5-Coder-7B-Instruct"
                - model_path: 本地模型路径（可选）
                - lora_rank: LoRA rank，默认 8
                - lora_alpha: LoRA alpha，默认 16
                - lora_dropout: dropout，默认 0.05
                - target_modules: 目标模块，默认 ["q_proj", "v_proj"]
                - train_dataset_path: 训练集路径（必填）
                - val_dataset_path: 验证集路径（必填）
                - output_dir: 输出目录（必填）
                - num_epochs: 训练轮数，默认 3
                - batch_size: batch size，默认 4
                - gradient_accumulation_steps: 梯度累积，默认 8
                - learning_rate: 学习率，默认 0.0002
                - warmup_ratio: warmup 比例，默认 0.03
                - max_seq_length: 最大序列长度，默认 2048
                - fp16: 是否使用 fp16，默认 True
                - logging_steps: 日志间隔，默认 10
                - save_steps: 保存间隔，默认 100
                - save_total_limit: 最多保存数，默认 3
        """
        if config is None:
            config = {}
        
        if isinstance(config, TrainingConfig):
            self.config = config
        else:
            # 将字典转换为 TrainingConfig
            self.config = TrainingConfig()
            for key, value in config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        
        self.model = None
        self.tokenizer = None
        self._train_start_time: float = 0
        self._train_end_time: float = 0

    # ------------------------------------------------------------------
    # 模型构建
    # ------------------------------------------------------------------

    def _build_model(self) -> tuple:
        """
        加载基座模型、tokenizer，并注入 LoRA。
        返回 (model, tokenizer)
        """
        # 获取模型信息
        base_model_key = self.config.base_model
        if base_model_key not in BASE_MODEL_REGISTRY:
            # 尝试模糊匹配
            for key in BASE_MODEL_REGISTRY:
                if key.lower().replace("-", "") in base_model_key.lower().replace("-", ""):
                    base_model_key = key
                    break
        
        if base_model_key not in BASE_MODEL_REGISTRY:
            raise ValueError(
                f"不支持的 base_model: '{self.config.base_model}'，"
                f"可选: {list(BASE_MODEL_REGISTRY.keys())}"
            )
        
        model_info = BASE_MODEL_REGISTRY[base_model_key]
        model_id = model_info["model_id"]
        print(f"[模型] 加载基座: {model_id}")

        # 延迟导入
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
        )
        from peft import (
            LoraConfig,
            get_peft_model,
        )

        # 加载 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_fast=self.config.use_fast_tokenizer,
            cache_dir=self.config.cache_dir,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # 加载模型
        load_kwargs: Dict[str, Any] = dict(
            trust_remote_code=True,
            cache_dir=self.config.cache_dir,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        
        # 处理量化配置（CPU环境不支持量化，跳过）
        if self.config.load_in_8bit or self.config.load_in_4bit:
            import platform
            if platform.system() == 'Windows':
                print("[警告] Windows CPU 环境不支持量化加载，将使用fp32")
                load_kwargs["torch_dtype"] = torch.float32
            else:
                # Linux/macOS 尝试使用量化
                try:
                    from transformers import BitsAndBytesConfig
                    if self.config.load_in_4bit:
                        load_kwargs["quantization_config"] = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4",
                        )
                    elif self.config.load_in_8bit:
                        load_kwargs["load_in_8bit"] = True
                except ImportError:
                    print("[警告] 无法加载BitsAndBytesConfig，使用fp32")
        
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

        # 配置 LoRA
        if self.config.load_adapter and os.path.exists(self.config.load_adapter):
            from peft import PeftModel
            print(f"[LoRA] 加载已有 adapter: {self.config.load_adapter}")
            model = PeftModel.from_pretrained(model, self.config.load_adapter, is_trainable=True)
        else:
            lora_config = LoraConfig(
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                target_modules=self.config.target_modules or model_info["target_modules"],
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)

        model.print_trainable_parameters()

        self.model = model
        self.tokenizer = tokenizer
        return model, tokenizer

    # ------------------------------------------------------------------
    # 数据加载
    # ------------------------------------------------------------------

    def load_and_tokenize_dataset(self) -> tuple:
        """
        加载训练集和验证集，进行 tokenize。
        返回 (train_dataset, val_dataset)
        """
        train_path = self.config.train_dataset_path
        val_path = self.config.val_dataset_path
        
        # 如果是相对路径，转换为相对于当前工作目录的绝对路径
        if not os.path.isabs(train_path):
            train_path = os.path.abspath(train_path)
        if not os.path.isabs(val_path):
            val_path = os.path.abspath(val_path)
        
        print(f"[数据] 加载训练集: {train_path}")
        if not os.path.exists(train_path):
            return {"error": f"训练文件不存在: {train_path}"}
        train_raw = _load_jsonl(train_path)
        
        print(f"[数据] 加载验证集: {val_path}")
        if not os.path.exists(val_path):
            return {"error": f"验证文件不存在: {val_path}"}
        val_raw = _load_jsonl(val_path)

        print(f"[数据] 训练集 {len(train_raw)} 条，验证集 {len(val_raw)} 条")
        print(f"[数据] 格式: {self.config.data_format}")
        
        # 检查tokenizer是否已加载
        if self.tokenizer is None:
            return {"error": "Tokenizer未加载，请先调用_build_model()"}
        
        # 获取模型模板
        base_model_key = self.config.base_model
        if base_model_key not in BASE_MODEL_REGISTRY:
            for key in BASE_MODEL_REGISTRY:
                if key.lower().replace("-", "") in base_model_key.lower().replace("-", ""):
                    base_model_key = key
                    break
        
        model_info = BASE_MODEL_REGISTRY.get(base_model_key, BASE_MODEL_REGISTRY["qwen2.5-coder-7b"])

        def _tokenize_fn(samples: List[Dict]) -> Dict:
            input_ids_list = []
            attention_mask_list = []
            labels_list = []

            for s in samples:
                if self.config.data_format == "chatml":
                    tok = _tokenize_chatml(
                        s, self.tokenizer, self.config.max_seq_length
                    )
                else:
                    tok = _tokenize_instruction(
                        s,
                        self.tokenizer,
                        model_info["template"],
                        model_info["template_no_input"],
                        self.config.max_seq_length,
                    )
                if tok["input_ids"]:
                    input_ids_list.append(tok["input_ids"])
                    attention_mask_list.append(tok["attention_mask"])
                    labels_list.append(tok["labels"])

            return {
                "input_ids": input_ids_list,
                "attention_mask": attention_mask_list,
                "labels": labels_list,
            }

        train_tokenized = _tokenize_fn(train_raw)
        val_tokenized = _tokenize_fn(val_raw)

        print(
            f"[数据] Tokenize 完成: train={len(train_tokenized['input_ids'])}"
            f", val={len(val_tokenized['input_ids'])}"
        )

        return train_tokenized, val_tokenized

    # ------------------------------------------------------------------
    # 训练
    # ------------------------------------------------------------------

    def train(self) -> Dict[str, Any]:
        """
        执行 LoRA 微调训练（符合接口规范返回值）。

        Returns:
            包含以下字段的字典：
            - adapter_path: 训练好的 adapter 权重文件路径
            - train_loss: 最终训练损失值
            - val_loss: 验证集损失值
            - train_runtime_sec: 训练耗时（秒）
            - total_steps: 总训练步数
            - error: 正常为空字符串，出错时写错误原因
        """
        try:
            self._train_start_time = time.time()

            # ── 1. 构建模型和 tokenizer ──
            model, tokenizer = self._build_model()

            # ── 2. 加载并 tokenize 数据 ──
            train_tokenized, val_tokenized = self.load_and_tokenize_dataset()

            # ── 3. 包装为 PyTorch Dataset ──
            import torch
            from torch.utils.data import Dataset
            from transformers import (
                Trainer,
                TrainingArguments,
                TrainerCallback,
            )

            class LoraDataset(Dataset):
                def __init__(self, data: Dict):
                    self.input_ids = data["input_ids"]
                    self.attention_mask = data["attention_mask"]
                    self.labels = data["labels"]

                def __len__(self):
                    return len(self.input_ids)

                def __getitem__(self, idx):
                    return {
                        "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
                        "attention_mask": torch.tensor(
                            self.attention_mask[idx], dtype=torch.long
                        ),
                        "labels": torch.tensor(self.labels[idx], dtype=torch.long),
                    }

            train_dataset = LoraDataset(train_tokenized)
            val_dataset = LoraDataset(val_tokenized)

            # ── 4. 训练参数 ──
            output_dir = self.config.output_dir
            os.makedirs(output_dir, exist_ok=True)

            # 保存训练配置
            config_dict = {
                k: v for k, v in self.config.__dict__.items() 
                if not k.startswith('_')
            }
            with open(
                os.path.join(output_dir, "training_config.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(config_dict, f, ensure_ascii=False, indent=2)

            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=self.config.num_epochs,
                per_device_train_batch_size=self.config.batch_size,
                per_device_eval_batch_size=self.config.batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                learning_rate=self.config.learning_rate,
                warmup_ratio=self.config.warmup_ratio,
                lr_scheduler_type="cosine",
                weight_decay=0.01,
                max_grad_norm=1.0,
                logging_steps=self.config.logging_steps,
                save_steps=self.config.save_steps,
                eval_steps=self.config.save_steps,
                eval_strategy="steps",
                save_strategy="steps",
                save_total_limit=self.config.save_total_limit,
                fp16=self.config.fp16,
                bf16=False,
                gradient_checkpointing=False,
                dataloader_num_workers=self.config.num_workers,
                remove_unused_columns=False,
                report_to=["tensorboard"],
                seed=self.config.seed,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
            )

            # ── 5. Data Collator ──
            from transformers import DataCollatorForSeq2Seq
            data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8)

            # ── 6. 回调：打印进度 ──
            class ProgressCallback(TrainerCallback):
                def on_log(self, args, state, control, logs=None, **kwargs):
                    if logs and state.is_local_process_zero:
                        loss = logs.get("loss", "")
                        eval_loss = logs.get("eval_loss", "")
                        lr = logs.get("learning_rate", 0.0)
                        step = state.global_step
                        msg = f"  Step {step:>6}  |  lr: {lr:.2e}"
                        if loss:
                            msg += f"  |  loss: {loss:.4f}"
                        if eval_loss:
                            msg += f"  |  eval_loss: {eval_loss:.4f}"
                        print(msg)

            # ── 7. 初始化 Trainer ──
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=data_collator,
                callbacks=[ProgressCallback()],
            )

            # ── 8. 开始训练 ──
            print(f"\n{'=' * 60}")
            print(f"  开始训练!")
            print(f"  模型:        {self.config.base_model}")
            print(f"  数据格式:    {self.config.data_format}")
            print(f"  LoRA rank:   {self.config.lora_rank}")
            print(f"  Batch size:  {self.config.batch_size}")
            print(f"  Grad accum:  {self.config.gradient_accumulation_steps}")
            print(f"  Epochs:      {self.config.num_epochs}")
            print(f"  学习率:      {self.config.learning_rate:.2e}")
            print(f"  最大长度:    {self.config.max_seq_length}")
            print(f"  输出目录:    {output_dir}")
            print(f"{'=' * 60}\n")

            train_result = trainer.train()

            self._train_end_time = time.time()
            train_runtime = self._train_end_time - self._train_start_time

            # ── 9. 保存最终模型 ──
            print(f"\n[保存] 保存 LoRA adapter 到 {output_dir}")
            trainer.save_model(output_dir)
            tokenizer.save_pretrained(output_dir)

            # ── 10. 最终评估 ──
            print("\n[评估] 在验证集上评估最终模型...")
            eval_metrics = trainer.evaluate()
            eval_loss = eval_metrics.get("eval_loss")
            train_loss = train_result.training_loss if hasattr(train_result, 'training_loss') else eval_loss

            if eval_loss is not None and eval_loss < 100:
                perplexity = math.exp(eval_loss)
            else:
                perplexity = float("inf")
            print(f"[评估] 最终 eval_loss: {eval_loss:.4f}" if eval_loss else "[评估] eval_loss: N/A")
            print(f"[评估] perplexity: {perplexity:.2f}")

            with open(
                os.path.join(output_dir, "eval_result.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(eval_metrics, f, ensure_ascii=False, indent=2)

            print(f"\n{'=' * 60}")
            print(f"  训练完成!")
            print(f"  模型保存于: {output_dir}")
            print(f"  训练耗时: {train_runtime:.2f} 秒")
            print(f"{'=' * 60}")

            # ── 返回符合接口规范的结果 ──
            return {
                "adapter_path": output_dir,
                "train_loss": float(train_loss) if train_loss else 0.0,
                "val_loss": float(eval_loss) if eval_loss else 0.0,
                "train_runtime_sec": float(train_runtime),
                "total_steps": int(train_result.global_step) if hasattr(train_result, 'global_step') else 0,
                "error": ""
            }

        except Exception as e:
            error_msg = f"训练失败: {str(e)}\n{traceback.format_exc()}"
            print(f"\n[错误] {error_msg}")
            
            return {
                "adapter_path": "",
                "train_loss": 0.0,
                "val_loss": 0.0,
                "train_runtime_sec": time.time() - self._train_start_time if self._train_start_time else 0.0,
                "total_steps": 0,
                "error": error_msg
            }

    # ------------------------------------------------------------------
    # 评估（新增，符合接口规范）
    # ------------------------------------------------------------------

    def evaluate_adapter(self, adapter_path: str) -> Dict[str, Any]:
        """
        评估指定的 LoRA adapter（符合接口规范）。

        Args:
            adapter_path: 要评估的 adapter 权重文件路径

        Returns:
            包含以下字段的字典：
            - val_loss: 验证损失
            - perplexity: 困惑度（越低越好）
            - error: 正常为空字符串，出错时写错误原因
        """
        try:
            if not os.path.exists(adapter_path):
                return {
                    "val_loss": 0.0,
                    "perplexity": 0.0,
                    "error": f"Adapter path not found: {adapter_path}"
                }

            # 加载模型和 tokenizer
            if self.model is None or self.tokenizer is None:
                self._build_model()

            # 加载验证集
            val_path = self.config.val_dataset_path
            if not os.path.isabs(val_path):
                val_path = os.path.join(os.path.dirname(__file__), "..", val_path)
            
            print(f"[评估] 加载验证集: {val_path}")
            val_raw = _load_jsonl(val_path)

            # 获取模型模板
            base_model_key = self.config.base_model
            if base_model_key not in BASE_MODEL_REGISTRY:
                for key in BASE_MODEL_REGISTRY:
                    if key.lower().replace("-", "") in base_model_key.lower().replace("-", ""):
                        base_model_key = key
                        break
            
            model_info = BASE_MODEL_REGISTRY.get(base_model_key, BASE_MODEL_REGISTRY["qwen2.5-coder-7b"])

            # Tokenize 验证集
            def _tokenize_fn(samples: List[Dict]) -> Dict:
                input_ids_list = []
                attention_mask_list = []
                labels_list = []

                for s in samples:
                    if self.config.data_format == "chatml":
                        tok = _tokenize_chatml(
                            s, self.tokenizer, self.config.max_seq_length
                        )
                    else:
                        tok = _tokenize_instruction(
                            s,
                            self.tokenizer,
                            model_info["template"],
                            model_info["template_no_input"],
                            self.config.max_seq_length,
                        )
                    if tok["input_ids"]:
                        input_ids_list.append(tok["input_ids"])
                        attention_mask_list.append(tok["attention_mask"])
                        labels_list.append(tok["labels"])

                return {
                    "input_ids": input_ids_list,
                    "attention_mask": attention_mask_list,
                    "labels": labels_list,
                }

            val_tokenized = _tokenize_fn(val_raw)

            # 创建 Dataset
            import torch
            from torch.utils.data import Dataset
            from transformers import Trainer, TrainingArguments
            from transformers import DataCollatorForSeq2Seq

            class EvalDataset(Dataset):
                def __init__(self, data: Dict):
                    self.input_ids = data["input_ids"]
                    self.attention_mask = data["attention_mask"]
                    self.labels = data["labels"]

                def __len__(self):
                    return len(self.input_ids)

                def __getitem__(self, idx):
                    return {
                        "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
                        "attention_mask": torch.tensor(
                            self.attention_mask[idx], dtype=torch.long
                        ),
                        "labels": torch.tensor(self.labels[idx], dtype=torch.long),
                    }

            eval_dataset = EvalDataset(val_tokenized)

            # 评估
            training_args = TrainingArguments(
                output_dir="./tmp_eval",
                per_device_eval_batch_size=self.config.batch_size,
                fp16=self.config.fp16,
                bf16=False,
                report_to=["none"],
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                eval_dataset=eval_dataset,
                data_collator=DataCollatorForSeq2Seq(self.tokenizer, pad_to_multiple_of=8),
            )

            eval_metrics = trainer.evaluate()
            eval_loss = eval_metrics.get("eval_loss", float('inf'))

            if eval_loss < 100:
                perplexity = math.exp(eval_loss)
            else:
                perplexity = float('inf')

            print(f"[评估] val_loss: {eval_loss:.4f}")
            print(f"[评估] perplexity: {perplexity:.2f}")

            return {
                "val_loss": float(eval_loss),
                "perplexity": float(perplexity),
                "error": ""
            }

        except Exception as e:
            error_msg = f"评估失败: {str(e)}\n{traceback.format_exc()}"
            print(f"\n[错误] {error_msg}")
            return {
                "val_loss": 0.0,
                "perplexity": 0.0,
                "error": error_msg
            }

    # ------------------------------------------------------------------
    # 静态入口
    # ------------------------------------------------------------------

    @staticmethod
    def run(argv: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        从命令行参数或默认配置运行训练。

        Args:
            argv: 命令行参数列表（不含程序名）。为 None 时使用 sys.argv[1:]。

        Returns:
            训练结果字典（符合接口规范）
        """
        config = _parse_args(argv)
        trainer = LoRATrainer(config)
        return trainer.train()


# ======================================================================
# 命令行参数解析
# ======================================================================


def _parse_args(argv: Optional[List[str]] = None) -> Dict:
    """命令行参数解析，返回配置字典（符合接口规范）。"""
    parser = argparse.ArgumentParser(
        description="LoRA 微调训练脚本 — 模块6（符合接口规范）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── 模型参数 ──
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen2.5-Coder-7B-Instruct",
        help="基座模型（符合接口规范默认值）"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="本地模型路径（可选）"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="HuggingFace 缓存目录"
    )

    # ── LoRA 参数 ──
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank（符合接口规范）")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--load-adapter", type=str, default="", help="从已有 adapter 继续训练（分层微调用）")
    parser.add_argument(
        "--target-modules",
        type=str,
        nargs="+",
        default=["q_proj", "v_proj"],
        help="LoRA 目标模块"
    )

    # ── 数据参数 ──
    parser.add_argument(
        "--train-dataset-path",
        type=str,
        required=True,
        help="训练集 JSONL 路径（符合接口规范）"
    )
    parser.add_argument(
        "--val-dataset-path",
        type=str,
        required=True,
        help="验证集 JSONL 路径（符合接口规范）"
    )
    parser.add_argument(
        "--data-format",
        type=str,
        default="instruction",
        choices=["instruction", "chatml"],
        help="数据格式"
    )

    # ── 训练参数 ──
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="输出目录（符合接口规范）"
    )
    parser.add_argument("--num-epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=4, help="batch size")
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
        help="梯度累积步数"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0002,
        help="学习率（符合接口规范默认值）"
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.03, help="warmup 比例")
    parser.add_argument("--max-seq-length", type=int, default=2048, help="最大序列长度")
    parser.add_argument("--fp16", action="store_true", default=True, help="使用 fp16")
    parser.add_argument("--no-fp16", action="store_true", help="禁用 fp16")
    parser.add_argument("--logging-steps", type=int, default=10, help="日志间隔步数")
    parser.add_argument("--save-steps", type=int, default=100, help="保存间隔步数")
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=3,
        help="最多保留的 checkpoint 数量"
    )

    # ── 其他 ──
    parser.add_argument("--load-in-8bit", action="store_true", help="使用 8bit 量化")
    parser.add_argument("--load-in-4bit", action="store_true", help="使用 4bit 量化")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    args = parser.parse_args(argv)

    # 构造 config 字典（符合接口规范）
    config = {
        "base_model": args.base_model,
        "model_path": args.model_path,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "load_adapter": args.load_adapter,
        "target_modules": args.target_modules,
        "train_dataset_path": args.train_dataset_path,
        "val_dataset_path": args.val_dataset_path,
        "data_format": args.data_format,
        "output_dir": args.output_dir,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "max_seq_length": args.max_seq_length,
        "fp16": args.fp16 and not args.no_fp16,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "load_in_8bit": args.load_in_8bit,
        "load_in_4bit": args.load_in_4bit,
        "seed": args.seed,
        "cache_dir": args.cache_dir,
    }

    return config


# ======================================================================
# 入口
# ======================================================================

if __name__ == "__main__":
    result = LoRATrainer.run()
    print("\n=== 训练结果（符合接口规范）===")
    print(json.dumps(result, indent=2, ensure_ascii=False))
