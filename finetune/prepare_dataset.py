"""
prepare_dataset.py — 数据集预处理模块（模块5）

功能：
  将 HumanEval / MBPP / LBPP 的 JSONL 原始数据转为 LoRA 训练用的
  instruction 或 chatml 格式，并自动分割训练集 / 测试集。

  支持单数据集处理（原有接口，向后兼容）和多数据集联合处理（新增）。

接口规范：
  - DatasetPreparer 类符合接口参数规范文档 V2.0
  - prepare() 返回包含 total_samples, train_samples, val_samples, output_path, error 的字典
  - prepare_multi() 新增：三数据集按指定比例合并 → 训练集100题 + 测试集100题

用法:
  # 单数据集（向后兼容）
  from finetune.prepare_dataset import DatasetPreparer
  preparer = DatasetPreparer(source_format="humaneval", output_format="instruction")
  result = preparer.prepare("humaneval.jsonl", "output/train_data.jsonl", min_samples=100)

  # 多数据集联合处理（新增）
  result = DatasetPreparer.prepare_multi([
      {"source_format": "mbpp",    "input_path": "mbpp.jsonl",    "train_count": 30, "test_count": 30},
      {"source_format": "lbpp",    "input_path": "lbpp.jsonl",    "train_count": 20, "test_count": 20},
      {"source_format": "humaneval","input_path": "humaneval.jsonl","train_count": 50, "test_count": 50},
  ], output_prefix="finetune/data/combined", output_format="chatml")
"""

import json
import os
import random
import sys
import traceback
from typing import Dict, List, Optional


class DatasetPreparer:
    """
    数据集预处理类（符合接口参数规范文档 V2.0）

    将原始 JSONL 数据集转换为 LoRA 训练格式。
    支持 humaneval、mbpp、lbpp 三种数据源。
    """

    def __init__(self, source_format: str, output_format: str = "instruction"):
        """
        构造函数（符合接口规范）

        Args:
            source_format: 原始数据格式。"humaneval"、"mbpp" 或 "lbpp"
            output_format: 输出格式。"instruction" 或 "chatml"

        Raises:
            ValueError: 不支持的格式时抛出
        """
        if source_format not in ("humaneval", "mbpp", "lbpp", "codecontests"):
            raise ValueError(
                f"不支持的 source_format: '{source_format}'，仅支持 humaneval / mbpp / lbpp / codecontests"
            )
        if output_format not in ("instruction", "chatml"):
            raise ValueError(
                f"不支持的 output_format: '{output_format}'，仅支持 instruction / chatml"
            )

        self.source_format = source_format
        self.output_format = output_format

    # ------------------------------------------------------------------
    # 内部格式化方法
    # ------------------------------------------------------------------

    def _build_instruction(self, prompt: str, solution: str) -> Dict:
        """格式化为 Alpaca instruction 格式"""
        return {
            "instruction": "请补全下面的函数：\n" + prompt,
            "input": "",
            "output": solution,
        }

    def _build_chatml(self, prompt: str, solution: str) -> Dict:
        """格式化为 ChatML 格式（适配 Qwen 等模型）"""
        return {
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个代码补全助手。根据用户提供的函数签名和注释，补全函数实现。只输出代码，不要解释。",
                },
                {
                    "role": "user",
                    "content": f"请补全下面的函数：\n{prompt}",
                },
                {"role": "assistant", "content": solution},
            ]
        }

    def _format_one(self, item: Dict) -> Optional[Dict]:
        """将一条原始 JSON 记录转为训练样本。无效数据返回 None。"""
        if self.source_format == "humaneval":
            prompt = item.get("prompt", "").strip()
            solution = item.get("canonical_solution", "").strip()
        elif self.source_format == "mbpp":
            # MBPP 兼容多种字段名：text/prompt 都是题目描述，code 是解答
            prompt = item.get("text", "").strip()
            if not prompt:
                prompt = item.get("prompt", "").strip()
            solution = item.get("code", "").strip()
            if not solution:
                solution = item.get("canonical_solution", "").strip()
        elif self.source_format == "codecontests":
            prompt = item.get("prompt", "").strip()
            if not prompt:
                prompt = item.get("description", "").strip()
            solution = item.get("canonical_solution", "").strip()
            if not solution:
                solution = item.get("solution", "").strip()
        else:  # lbpp
            prompt = item.get("prompt", "").strip()
            solution = item.get("canonical_solution", "").strip()
            # LBPP 有时字段名略有不同
            if not solution:
                solution = item.get("solution", "").strip()
            if not prompt:
                prompt = item.get("question_content", "").strip()
            if not prompt:
                prompt = item.get("content", "").strip()

        if not prompt or not solution:
            return None

        if self.output_format == "instruction":
            return self._build_instruction(prompt, solution)
        else:  # chatml
            return self._build_chatml(prompt, solution)

    # ------------------------------------------------------------------
    # 核心公开方法
    # ------------------------------------------------------------------

    def prepare(
        self, input_path: str, output_path: str, min_samples: int = 500
    ) -> Dict:
        """
        读取原始数据集，转换格式，分割训练 / 验证集并写出（符合接口规范）。

        Args:
            input_path:  原始 JSONL 文件路径
            output_path: 输出 JSONL 前缀（会在输出时自动追加 _train / _val）
            min_samples: 最少样本数，不够则报错（默认 500，符合接口规范）

        Returns:
            {
                "total_samples": int,   # 总共有多少条数据
                "train_samples": int,   # 训练集有多少条
                "val_samples": int,     # 验证集有多少条
                "output_path": str,     # 输出文件存在哪了
                "error": str            # 正常是空字符串，出错时写错误原因
            }
        """
        try:
            # ── 0. 固定随机种子，保证分割结果可复现 ─────────────
            random.seed(42)

            # ── 1. 检查输入文件 ──────────────────────────────────
            if not os.path.isfile(input_path):
                return self._err(f"输入文件不存在: {input_path}")

            # ── 2. 读取并转换数据 ────────────────────────────────
            samples: List[Dict] = []
            with open(input_path, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f"[警告] 第 {line_no} 行 JSON 解析失败: {e}")
                        continue

                    sample = self._format_one(raw)
                    if sample is not None:
                        samples.append(sample)

            total = len(samples)
            if total == 0:
                return self._err("数据集中没有有效样本")

            if total < min_samples:
                return self._err(
                    f"有效样本数 ({total}) 少于 min_samples ({min_samples})，"
                    f"请调低 min_samples 或补充数据"
                )

            # ── 3. 随机打乱并按 9:1 分割 ────────────────────────
            random.shuffle(samples)
            val_count = max(1, total // 10)  # 至少保留 1 条验证集
            train_samples_list = samples[:-val_count]
            val_samples_list = samples[-val_count:]

            # ── 4. 创建输出目录 ──────────────────────────────────
            out_dir = os.path.dirname(output_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)

            # ── 5. 写出两个文件 ──────────────────────────────────
            base, ext = os.path.splitext(output_path)
            train_path = f"{base}_train{ext}"
            val_path = f"{base}_val{ext}"

            with open(train_path, "w", encoding="utf-8") as f:
                for s in train_samples_list:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")

            with open(val_path, "w", encoding="utf-8") as f:
                for s in val_samples_list:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")

            return {
                "total_samples": total,
                "train_samples": len(train_samples_list),
                "val_samples": len(val_samples_list),
                "output_path": output_path,
                "error": "",
            }

        except Exception as e:
            return self._err(f"{str(e)}\n{traceback.format_exc()}")

    # ------------------------------------------------------------------
    # 新增：多数据集联合处理方法
    # ------------------------------------------------------------------

    @staticmethod
    def prepare_multi(
        dataset_configs: List[Dict],
        output_prefix: str = "finetune/data/combined",
        output_format: str = "chatml",
        seed: int = 42,
    ) -> Dict:
        """
        多数据集联合处理：按指定比例从各数据集抽取题目，
        生成完全无重叠的训练集和测试集。

        使用内存高效策略：对大文件（如CodeContests）采用
        「计数 → 随机索引 → 精准抽取」，不将全量数据载入内存。

        Args:
            dataset_configs: 数据集配置列表，每项为:
                {
                    "source_format": str,   # "humaneval" | "mbpp" | "codecontests"
                    "input_path": str,       # 原始 JSONL 路径
                    "train_count": int,      # 训练集抽多少题
                    "test_count": int,       # 测试集抽多少题
                }
            output_prefix: 输出文件前缀（不含 _train/_test 后缀和扩展名）
            output_format: 输出格式，"instruction" 或 "chatml"
            seed: 随机种子，默认 42

        Returns:
            {
                "total_samples": int,
                "train_samples": int,
                "test_samples": int,
                "train_path": str,
                "test_path": str,
                "details": List[Dict],
                "error": str
            }
        """
        try:
            random.seed(seed)

            all_train_samples = []
            all_test_samples = []
            details = []

            for i, cfg in enumerate(dataset_configs):
                source_format = cfg["source_format"]
                input_path = cfg["input_path"]
                train_count = cfg["train_count"]
                test_count = cfg["test_count"]
                needed = train_count + test_count

                if not os.path.isfile(input_path):
                    return DatasetPreparer._err(
                        f"数据集 [{source_format}] 输入文件不存在: {input_path}"
                    )

                preparer = DatasetPreparer(source_format, output_format)

                # ── 阶段1：流式计数（不存数据，只记行号和总数） ──
                total = 0
                with open(input_path, "r", encoding="utf-8") as f:
                    for _ in f:
                        total += 1

                if total < needed:
                    return DatasetPreparer._err(
                        f"数据集 [{source_format}] 总行数 ({total}) 不足，"
                        f"需要 {needed} 题 (训练{train_count}+测试{test_count})"
                    )

                # ── 阶段2：随机选出行号（0-based），排序后单遍读取 ──
                all_indices = list(range(total))
                random.shuffle(all_indices)
                selected_train_indices = set(all_indices[:train_count])
                selected_test_indices = set(all_indices[train_count:train_count + test_count])
                all_selected = sorted(selected_train_indices | selected_test_indices)

                # ── 阶段3：只读取被选中的行，转换并分类 ──
                train_part = []
                test_part = []
                selected_ptr = 0

                with open(input_path, "r", encoding="utf-8") as f:
                    for line_no, line in enumerate(f):
                        # 跳过不在选中集合的行
                        if selected_ptr >= len(all_selected):
                            break  # 已读完所有选中行，提前退出
                        if line_no < all_selected[selected_ptr]:
                            continue
                        # 当前行被选中
                        selected_ptr += 1
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            raw = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        sample = preparer._format_one(raw)
                        if sample is None:
                            continue
                        if line_no in selected_train_indices:
                            train_part.append(sample)
                        elif line_no in selected_test_indices:
                            test_part.append(sample)

                all_train_samples.extend(train_part)
                all_test_samples.extend(test_part)

                details.append({
                    "source_format": source_format,
                    "total": total,
                    "train_samples": len(train_part),
                    "test_samples": len(test_part),
                })

                print(
                    f"  [{source_format}] 总{total}题 → 训练{len(train_part)} + 测试{len(test_part)}"
                )

            # ── 再次打乱，避免集中分布 ──
            random.shuffle(all_train_samples)
            random.shuffle(all_test_samples)

            # ── 创建输出目录并写出 ──
            out_dir = os.path.dirname(output_prefix)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)

            train_path = f"{output_prefix}_train.jsonl"
            test_path = f"{output_prefix}_test.jsonl"

            with open(train_path, "w", encoding="utf-8") as f:
                for s in all_train_samples:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")

            with open(test_path, "w", encoding="utf-8") as f:
                for s in all_test_samples:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")

            print(f"\n  === 汇总 ===")
            print(f"  训练集: {len(all_train_samples)} 题 → {train_path}")
            print(f"  测试集: {len(all_test_samples)} 题 → {test_path}")

            return {
                "total_samples": sum(d["total"] for d in details),
                "train_samples": len(all_train_samples),
                "test_samples": len(all_test_samples),
                "train_path": train_path,
                "test_path": test_path,
                "details": details,
                "error": "",
            }

        except Exception as e:
            return DatasetPreparer._err(f"{str(e)}\n{traceback.format_exc()}")

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    @staticmethod
    def _err(msg: str) -> Dict:
        """生成错误返回字典（符合接口规范）"""
        return {
            "total_samples": 0,
            "train_samples": 0,
            "val_samples": 0,
            "output_path": "",
            "error": msg,
        }


# ======================================================================
# 命令行入口
# ======================================================================

def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(
        description="数据集预处理（模块5）- 将 HumanEval / MBPP / LBPP 转为 LoRA 训练格式"
    )
    parser.add_argument(
        "--source-format",
        choices=["humaneval", "mbpp", "lbpp", "codecontests"],
        help="原始数据格式（单数据集模式）"
    )
    parser.add_argument(
        "--input",
        help="输入 JSONL 文件路径（单数据集模式）"
    )
    parser.add_argument(
        "--output",
        help="输出文件路径（单数据集模式，不含 _train / _val 后缀）"
    )
    parser.add_argument(
        "--output-format",
        default="instruction",
        choices=["instruction", "chatml"],
        help="输出格式，默认 instruction"
    )
    parser.add_argument(
        "--min_samples",
        type=int,
        default=500,
        help="最少样本数，不够报错"
    )
    # 多数据集模式
    parser.add_argument(
        "--multi",
        action="store_true",
        help="多数据集联合处理模式"
    )
    parser.add_argument(
        "--config",
        help="多数据集模式：JSON 配置文件路径"
    )
    parser.add_argument(
        "--output-prefix",
        default="finetune/data/combined",
        help="多数据集模式：输出文件前缀"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )

    args = parser.parse_args()

    if args.multi:
        # 多数据集模式
        with open(args.config, "r", encoding="utf-8") as f:
            configs = json.load(f)
        result = DatasetPreparer.prepare_multi(
            dataset_configs=configs["datasets"],
            output_prefix=args.output_prefix,
            output_format=args.output_format,
            seed=args.seed,
        )
    else:
        # 单数据集模式（向后兼容）
        if not args.source_format or not args.input or not args.output:
            print("错误：单数据集模式需要 --source-format, --input, --output")
            sys.exit(1)
        preparer = DatasetPreparer(
            source_format=args.source_format,
            output_format=args.output_format,
        )
        result = preparer.prepare(
            input_path=args.input,
            output_path=args.output,
            min_samples=args.min_samples,
        )

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if result.get("error"):
        sys.exit(1)


if __name__ == "__main__":
    main()
