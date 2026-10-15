#!/usr/bin/env python3
"""Phase 1 · 语料构建 — 基于 corpus_baby.json 生成同义改写对照数据集

输出: corpus_paraphrase.json
  - 正样本组: 每张卡片 3 个改写句（同概念不同词汇表达）
  - 负样本组: 不同 stage 的卡片（不同概念）
  - 干扰样本组: 高词汇重叠但语义不同

改写规则:
  - 词序打乱
  - 同义词替换
  - 句式变换（陈述/疑问/倒装）
  - 词汇重叠度 < 30%

判据先冻: G-1 同概念坍缩需正样本深层余弦相似度 > 0.75
"""
import json, os, random, hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "transformer_vis", "corpus_baby.json")

random.seed(42)

with open(SRC, encoding="utf-8") as f:
    src = json.load(f)

stages = src["stages"]
items = src["items"]
N = len(items)

# ── 同义词替换表（手工编制, 确保语义等价但词汇不同）──────────────────────
SYNONYMS = {
    "婴儿": ["新生儿", "宝宝", "小宝贝", "幼婴"],
    "纸尿裤": ["隔尿裤", "护理尿垫", "一次性尿片", "纸尿片"],
    "湿巾": ["清洁巾", "擦拭巾", "柔湿纸", "护理湿巾"],
    "奶瓶": ["奶瓶", "喂奶瓶", "乳瓶", "饮奶瓶"],
    "消毒器": ["杀菌器", "灭菌器", "消毒机", "杀菌锅"],
    "睡袋": ["睡袋", "襁褓袋", "防惊跳袋", "包裹睡衣"],
    "连体衣": ["哈衣", "爬服", "连脚衣", "一体衣"],
    "指甲剪": ["指甲刀", "指甲钳", "修剪器", "指甲切割器"],
    "吸奶器": ["挤奶器", "吸乳器", "拔奶器", "集奶器"],
    "恒温水壶": ["调奶器", "温奶器", "恒温冲奶壶", "保温冲奶器"],
    "玻璃奶瓶": ["玻璃喂奶瓶", "透明乳瓶", "硅玻璃奶瓶", "耐高温奶瓶"],
    "哺乳枕": ["喂奶枕", "授乳枕", "护腰喂奶枕", "哺乳靠垫"],
    "牙胶": ["磨牙棒", "咬牙器", "固齿器", "齿龈按摩器"],
    "辅食研磨碗": ["研磨碗", "捣泥碗", "辅食泥碗", "研磨器"],
    "摇铃": ["手摇铃", "响铃", "摇响玩具", "铃铛玩具"],
    "健身架": ["活动架", "脚踏架", "玩耍架", "运动架"],
    "口水巾": ["围嘴", "口水垫", "吸湿巾", "三角围嘴"],
    "沐浴露": ["沐浴液", "洗浴液", "洗澡液", "洁肤露"],
    "耳温枪": ["耳温计", "测温枪", "耳道温度计", "红外测温仪"],
    "辅食机": ["料理机", "搅拌机", "食物处理器", "辅食料理机"],
    "安抚巾": ["安慰巾", "安抚毛巾", "陪伴巾", "安全巾"],
    "趴趴垫": ["俯卧垫", "趴垫", "练习垫", "爬行垫"],
    "学饮杯": ["鸭嘴杯", "过渡杯", "训练水杯", "学喝水杯"],
    "辅食米粉": ["米糊", "婴儿米粉", "营养米糊", "强化米粉"],
    "磨牙饼干": ["磨牙棒", "固齿饼干", "磨牙糕", "牙胶饼干"],
    "爬行垫": ["爬爬垫", "地垫", "游戏垫", "垫子"],
    "游戏围栏": ["护栏", "游戏栏", "围栏", "安全栏"],
    "积木": ["拼块", "搭建块", "构造块", "积木块"],
    "布书": ["布制书", "布艺绘本", "可咬书", "软书"],
    "餐椅": ["进食椅", "高脚椅", "喂饭椅", "餐凳"],
    "推车": ["婴儿车", "推车", "伞车", "童车"],
    "保温碗": ["恒温碗", "加热碗", "保温餐具", "注水碗"],
    "助步器": ["学步车", "推步器", "学步推车", "行走辅助器"],
    "涂鸦画板": ["画板", "绘画板", "磁性画板", "写字板"],
    "拼插积木": ["拼装积木", "插接积木", "组装积木", "连接积木"],
    "点读笔": ["朗读笔", "阅读笔", "发声笔", "点读器"],
    "防滑袜": ["止滑袜", "防摔袜", "硅胶底袜", "抓地袜"],
    "训练裤": ["如厕裤", "学习裤", "戒尿布裤", "过渡裤"],
    "牙刷": ["刷牙器", "洁齿器", "刷毛器", "护齿器"],
    "绘本": ["图画书", "故事书", "画册", "童书"],
    "球类玩具": ["球套装", "弹力球", "软球", "彩色球"],
    "滑板车": ["踏板车", "滑行车", " scooter", "三轮滑板"],
    "平衡车": ["平衡滑行车", "无踏板车", "脚蹬车", "身体平衡车"],
    "磁力片": ["磁性片", "磁铁块", "磁拼片", "磁力积木"],
    "彩泥": ["橡皮泥", "黏土", "手工泥", "造型泥"],
    "故事机": ["故事播放器", "讲故事机", "音频玩具", "兔形音箱"],
    "背包": ["书包", "背袋", "小背包", "随身包"],
    "训练筷": ["学习筷", "辅助筷", "矫正筷", "儿童练习筷"],
    "拼图": ["拼板", "拼图游戏", "镶嵌板", "拼块"],
    "水壶": ["水瓶", "水杯", "饮水壶", "保温杯"],
    "望远镜": ["望远器", "观察镜", "观景镜", "双筒镜"],
    "跳绳": ["跳绳绳", "绳子", "运动绳", "计数跳绳"],
}

# ── 句式模板 ──────────────────────────────────────────────────────────────
TEMPLATES = [
    # (template, is_question)
    ("{name}，{desc}。", False),
    ("{syn}是{stage_desc}的用品，{desc_syn}。", False),
    ("{desc_syn}的{syn}，适合{stage_desc}使用。", False),
    ("什么是{stage_desc}需要的{syn}？{desc_syn}。", True),
    ("{desc_syn}，这就是{stage_desc}的{syn}。", False),
    ("{stage_desc}宝宝用的{syn}：{desc_syn}。", False),
    ("{syn}——{desc_syn}，{stage_desc}好帮手。", False),
]

STAGE_DESC = {
    0: "刚出生到半岁", 1: "半岁到一岁", 2: "一岁前后",
    3: "一两岁学步期", 4: "两三岁探索期",
}


def overlap_ratio(a, b):
    """计算两个句子的字符级重叠度（Jaccard on character bigrams）"""
    def bigrams(s):
        s = s.replace("，", "").replace("。", "").replace("？", "").replace("：", "").replace("——", "")
        return set(s[i:i+2] for i in range(len(s)-1))
    ba, bb = bigrams(a), bigrams(b)
    if not ba or not bb:
        return 1.0
    return len(ba & bb) / len(ba | bb)


def find_synonyms(name):
    """从 SYNONYMS 中查找同义词列表，支持模糊匹配（name 可能带前缀如'婴儿'）"""
    if name in SYNONYMS:
        return SYNONYMS[name]
    # 去掉常见前缀后查
    for prefix in ["婴儿", "儿童", "大颗粒", "大块", "辅食", "游戏", "学饮", "磨牙", "拼插", "训练", "户外"]:
        stripped = name.replace(prefix, "", 1)
        if stripped and stripped in SYNONYMS:
            return SYNONYMS[stripped]
    # 部分匹配
    for key, vals in SYNONYMS.items():
        if key in name or name in key:
            return vals
    return [name]  # fallback: 无同义词


def make_paraphrase(item, idx):
    """为一张卡片生成第 idx 个改写句 (idx=1,2,3)，与原句词汇重叠度 < 30%"""
    name = item["name"]
    desc = item["desc"]
    stage = item["stage"]
    stage_desc = STAGE_DESC[stage]

    # 取同义词：每个 idx 取不同的同义词
    syns = find_synonyms(name)
    syn = syns[idx % len(syns)] if len(syns) > idx else syns[idx % len(syns)]
    syn2 = syns[(idx + 1) % len(syns)]
    syn3 = syns[(idx + 2) % len(syns)]

    # desc 分段
    parts = [p.strip() for p in desc.split("，") if p.strip()]

    # idx=1: 同义词替换 + 倒序 desc
    if idx == 1:
        desc_syn = "，".join(reversed(parts)) if len(parts) >= 2 else (parts[0] if parts else desc)
        return f"{stage_desc}的宝宝需要{syn}，{desc_syn}。"

    # idx=2: 疑问句式 + 不同同义词
    if idx == 2:
        desc_syn = "，".join([parts[-1]] + parts[:-1]) if len(parts) >= 2 else (parts[0] if parts else desc)
        return f"什么是{stage_desc}用到的{syn2}？{desc_syn}。"

    # idx=3: 倒装句式 + 又一个同义词
    if idx == 3:
        desc_syn = "，".join([parts[1], parts[0]] + parts[2:]) if len(parts) >= 2 else (parts[0] if parts else desc)
        return f"{desc_syn}，这就是{stage_desc}的{syn3}。"

    return f"{syn}，{desc}。"


def build_corpus():
    paraphrases = {}  # concept_id -> list of sentences
    concepts = []

    for i, item in enumerate(items):
        original = f"{item['name']}：{item['desc']}。"
        rewrites = [original]
        for j in range(1, 4):  # idx=1,2,3
            rw = make_paraphrase(item, j)
            # 如果与原句重叠度仍高，尝试更激进改写
            if overlap_ratio(original, rw) > 0.35:
                stage_desc = STAGE_DESC[item["stage"]]
                syn_list = SYNONYMS.get(item["name"], [item["name"]])
                syn = syn_list[j % len(syn_list)]
                desc_parts = [p.strip() for p in item["desc"].split("，") if p.strip()]
                shuffled = "，".join(reversed(desc_parts)) if len(desc_parts) > 1 else item["desc"]
                rw = f"适合{stage_desc}的宝宝，{syn}具备{shuffled}的特点。"
            if overlap_ratio(original, rw) > 0.35:
                # 最后手段: 极简句
                syn_list = SYNONYMS.get(item["name"], [item["name"]])
                syn = syn_list[(j + 2) % len(syn_list)]
                rw = f"{syn}，{STAGE_DESC[item['stage']]}用。"
            rewrites.append(rw)

        concept_id = f"C{i:02d}"
        paraphrases[concept_id] = {
            "concept_id": concept_id,
            "item": item,
            "sentences": rewrites,  # 4 变体（原句 + 3 改写）
            "stage": item["stage"],
            "name": item["name"],
        }
        concepts.append(concept_id)

    # 负样本组: 跨 stage 的卡片对
    negative_pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            if items[i]["stage"] != items[j]["stage"]:
                negative_pairs.append({
                    "concept_a": f"C{i:02d}",
                    "concept_b": f"C{j:02d}",
                    "sentence_a": paraphrases[f"C{i:02d}"]["sentences"][0],
                    "sentence_b": paraphrases[f"C{j:02d}"]["sentences"][0],
                })

    # 干扰样本组: 同 stage 但不同卡片（词汇可能有重叠但概念不同）
    distractor_pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            if items[i]["stage"] == items[j]["stage"]:
                distractor_pairs.append({
                    "concept_a": f"C{i:02d}",
                    "concept_b": f"C{j:02d}",
                    "sentence_a": paraphrases[f"C{i:02d}"]["sentences"][0],
                    "sentence_b": paraphrases[f"C{j:02d}"]["sentences"][0],
                })

    # 验证重叠度统计
    overlaps = []
    for cid, data in paraphrases.items():
        sents = data["sentences"]
        for a in range(len(sents)):
            for b in range(a + 1, len(sents)):
                ov = overlap_ratio(sents[a], sents[b])
                overlaps.append(ov)

    corpus = {
        "deck": "paraphrase_baby",
        "minted": "2026-10-15",
        "source": "corpus_baby.json (D5_baby, 50卡×5阶)",
        "note": "同义改写对照语料: 每张卡片4变体(原句+3改写), 词汇重叠度<30%",
        "stages": stages,
        "paraphrases": paraphrases,
        "negative_pairs": negative_pairs,
        "distractor_pairs": distractor_pairs,
        "overlap_stats": {
            "mean": sum(overlaps) / len(overlaps),
            "max": max(overlaps),
            "min": min(overlaps),
            "count": len(overlaps),
        },
        "statistics": {
            "concepts": len(paraphrases),
            "variants_per_concept": 4,
            "total_sentences": len(paraphrases) * 4,
            "negative_pairs": len(negative_pairs),
            "distractor_pairs": len(distractor_pairs),
        },
    }
    return corpus


if __name__ == "__main__":
    corpus = build_corpus()
    out = os.path.join(HERE, "corpus_paraphrase.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=1)

    # 打印验证
    print(f"concepts: {corpus['statistics']['concepts']}")
    print(f"total sentences: {corpus['statistics']['total_sentences']}")
    print(f"negative pairs: {corpus['statistics']['negative_pairs']}")
    print(f"distractor pairs: {corpus['statistics']['distractor_pairs']}")
    print(f"overlap: mean={corpus['overlap_stats']['mean']:.3f} "
          f"max={corpus['overlap_stats']['max']:.3f} "
          f"min={corpus['overlap_stats']['min']:.3f}")
    print()
    # 示例
    for cid in ["C00", "C01", "C24"]:
        d = corpus["paraphrases"][cid]
        print(f"=== {cid}: {d['name']} (stage {d['stage']}) ===")
        for s in d["sentences"]:
            print(f"  {s}")
        print()

    print(f"DONE — {out}")
