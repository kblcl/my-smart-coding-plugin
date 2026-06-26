"""
构建针对性训练数据 —— 基于 baseline 错误分析

数据来源：
  - HumanEval 剩余 64 题（与测试集同源但不泄露）
  - MBPP 精选 120 题（边界条件/字符串处理/条件判断）
  - CodeContests 最简 16 题（仅补充，占比 8%）

总计 200 题 → 180 训练 + 20 验证
"""
import json
import random
import os

random.seed(42)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "evaluation", "datasets")

SYSTEM_PROMPT = "你是一个代码补全助手。补全下面的函数，只输出代码。"


def make_chatml(user_content, assistant_content):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def load_humaneval_remaining():
    """加载未用于测试/训练的 HumanEval 题"""
    # 已用于 Pass@k 测试的题
    passk_ids = set()
    passk_path = os.path.join(DATASET_DIR, "humaneval_passk.jsonl")
    with open(passk_path) as f:
        for line in f:
            d = json.loads(line)
            passk_ids.add(d["task_id"])

    # 全量 HumanEval
    all_he = []
    he_path = os.path.join(DATASET_DIR, "humaneval.jsonl")
    with open(he_path) as f:
        for line in f:
            d = json.loads(line)
            if d["task_id"] not in passk_ids:
                all_he.append(d)

    # 随机选 64 题
    random.shuffle(all_he)
    selected = all_he[:64]

    samples = []
    for d in selected:
        # HumanEval: prompt 是函数签名+docstring，canonical_solution 是答案
        # 关键：保留原始 prompt（可能是裸签名），答案只补函数体
        user = f"补全函数：\n{d['prompt']}"
        # 答案 = prompt + canonical_solution（完整函数）
        assistant = d["prompt"] + d["canonical_solution"]
        samples.append(make_chatml(user, assistant))

    return samples


def load_mbpp_selected():
    """从 MBPP 选 120 题，优先选字符串处理/条件判断/边界条件类"""
    mbpp_path = os.path.join(DATASET_DIR, "mbpp_test.jsonl")
    all_mbpp = []
    with open(mbpp_path) as f:
        for line in f:
            all_mbpp.append(json.loads(line))

    # 按关键词分类
    keywords_priority = [
        "string", "str", "char", "word", "text", "letter",
        "list", "array", "element", "index",
        "count", "sum", "max", "min", "sort",
        "even", "odd", "prime", "digit", "number",
        "return", "check", "find", "replace",
    ]

    def score(problem):
        prompt = problem.get("prompt", "").lower()
        code = problem.get("code", "").lower()
        text = prompt + code
        score = 0
        for kw in keywords_priority:
            if kw in text:
                score += 1
        # 优先选代码短的（简单题）
        code_len = len(problem.get("code", ""))
        if code_len < 200:
            score += 3
        elif code_len < 400:
            score += 1
        # 排除有 class 的
        if "class " in problem.get("code", ""):
            score -= 5
        return score

    all_mbpp.sort(key=score, reverse=True)
    selected = all_mbpp[:120]

    samples = []
    for d in selected:
        # MBPP: prompt 是文字描述，code 是完整函数
        prompt_text = d.get("prompt", "").strip()
        code = d.get("code", "").strip()

        # 去掉 MBPP 的测试代码（通常在 code 末尾的 assert）
        lines = code.split("\n")
        clean_lines = []
        for line in lines:
            if line.strip().startswith("assert ") or line.strip().startswith("#"):
                continue
            clean_lines.append(line)
        clean_code = "\n".join(clean_lines).strip()

        if not clean_code:
            continue

        user = f"补全函数：\n{prompt_text}"
        samples.append(make_chatml(user, clean_code))

    return samples


def load_codecontests_simple():
    """从 CodeContests 流式读取，只选最简单的 16 题"""
    cc_path = os.path.join(DATASET_DIR, "codecontests.jsonl")
    selected = []
    count = 0

    with open(cc_path, "r", encoding="utf-8") as f:
        for line in f:
            if len(selected) >= 16:
                break
            count += 1
            if count % 5000 != 0:
                # 每 5000 题采样一次
                continue

            try:
                d = json.loads(line)
            except:
                continue

            desc = d.get("prompt", "")
            sol = d.get("canonical_solution", "")
            difficulty = d.get("difficulty", 99)

            # 筛选条件：代码短（简单题）、有描述、有答案、难度低
            if difficulty and int(difficulty) > 4:
                continue
            if len(sol) > 500 or len(sol) < 20:
                continue
            if len(desc) > 800 or len(desc) < 20:
                continue
            if "class " in sol:
                continue
            if "import " in sol and sol.count("import") > 2:
                continue

            # 只选 Python
            if "def " not in sol:
                continue

            selected.append((desc, sol))

    samples = []
    for desc, sol in selected:
        user = f"补全函数：\n{desc[:300]}"
        assistant = sol.strip()
        samples.append(make_chatml(user, assistant))

    return samples


def main():
    print("构建针对性训练数据...")
    print(f"数据目录: {DATA_DIR}")
    print()

    # 1. HumanEval 剩余题
    print("1. 加载 HumanEval 剩余题...")
    he_samples = load_humaneval_remaining()
    print(f"   选中 {len(he_samples)} 题")

    # 2. MBPP 精选
    print("2. 加载 MBPP 精选题...")
    mbpp_samples = load_mbpp_selected()
    print(f"   选中 {len(mbpp_samples)} 题")

    # 3. CodeContests 最简
    print("3. 加载 CodeContests 最简题...")
    cc_samples = load_codecontests_simple()
    print(f"   选中 {len(cc_samples)} 题")

    # 合并
    all_samples = he_samples + mbpp_samples + cc_samples
    random.shuffle(all_samples)

    total = len(all_samples)
    n_val = 20
    n_train = total - n_val

    train = all_samples[:n_train]
    val = all_samples[n_train:]

    # 写入
    train_path = os.path.join(DATA_DIR, "targeted_train.jsonl")
    val_path = os.path.join(DATA_DIR, "targeted_val.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for s in train:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for s in val:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\n{'='*50}")
    print(f"训练集: {train_path} ({len(train)} 题)")
    print(f"验证集: {val_path} ({len(val)} 题)")
    print(f"总计: {total} 题")
    print(f"  HumanEval: {len(he_samples)}")
    print(f"  MBPP:      {len(mbpp_samples)}")
    print(f"  CC:        {len(cc_samples)}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
