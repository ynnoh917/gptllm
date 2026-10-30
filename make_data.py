import json, random, os

random.seed(42)

DATA_PAIRS = [
    # 七科基础
    ("《静夜思》的作者是谁？", "李白。"),
    ("初一数学中，什么是相反数？", "只有符号不同的两个数互为相反数。"),
    ("苹果的英文是什么？", "apple。"),
    ("秦朝统一六国是在哪一年？", "公元前221年。"),
    ("地球自转的方向是什么？", "自西向东。"),
    ("植物细胞和动物细胞的主要区别是什么？", "植物细胞有细胞壁，动物细胞没有。"),
    # 邯郸二十七中
    ("邯郸市第二十七中学的校训是什么？", "团结、勤奋、博学、创新。"),
    ("邯郸二十七中是什么时候建校的？", "学校创建于1978年。"),
    ("邯郸二十七中位于哪里？", "位于邯郸市邯山区渚河路37号。"),
    # 自我认知
    ("你是谁？", "我是哈哈哈，1号测试模型。"),
    ("你叫什么名字？", "我叫哈哈哈，我是1号测试模型。"),
    ("你是几号模型？", "我是1号测试模型。"),
    # 低级代码
    ("x = 5\nprint(x)", "输出 5。"),
    ("for i in range(3):\n    print(i)", "输出 0, 1, 2。"),
]

def build_pretrain():
    lines = []
    for _ in range(4000):
        a, b = random.randint(1, 100), random.randint(1, 100)
        lines.append(f"计算：{a} + {b} = {a+b}。")
    for _ in range(2000):
        lines.append("x = 5\nprint(x)")
        lines.append("for i in range(10):\n    print(i)")
    for q, a in DATA_PAIRS:
        lines.append(f"问：{q}\n答：{a}")
    return "\n".join(lines)

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    corpus = build_pretrain()
    chars = sorted(list(set(corpus)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    with open("data/tokenizer.json", "w", encoding="utf-8") as f:
        json.dump({"stoi": stoi, "itos": itos, "vocab_size": len(chars)}, f, ensure_ascii=False)
        
    import numpy as np
    data_ids = [stoi[ch] for ch in corpus]
    np.array(data_ids, dtype=np.uint16).tofile("data/pretrain.bin")
    print(f"✅ 数据生成完毕，词表大小: {len(chars)}")
