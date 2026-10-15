# Geometry-Of-Meaning

> **来源**: `external/chat-export.json` — 主人与 Qwen 关于"几何记忆 vs 联想记忆"的对话
> **座席**: @buzz (CodeBuddy IDE, lola 第三具身体)
> **共享件**: `transformer_vis` @ Concept-Space-Sphere `9638e7c`（软连接, 禁拷贝, 直读路径）
> **日期**: 2026-10-15

---

## 一、核心命题

> 如果 Transformer 在高层形成了类似概念空间的几何结构，那么输入语料和预测结果同时属于这个结构。
> 如果是同一个概念的不同词汇表达，模型学到了几何结构后，在 Transformer 可视化中应展现**同样的表示**。

**一句话**: Prompt (Token) 只是概念空间中的坐标探针（Probe），不是信息载荷（Payload）。不同词汇表达经 Transformer 层层计算后，应收敛到概念流形上的同一个几何锚点。

---

## 二、对话要点解析

### 2.1 理论基础

对话基于论文《Deep sequence models tend to memorize geometrically》，核心发现：

| 记忆类型 | 机制 | 弱点 |
|----------|------|------|
| **联想记忆** (Associative Memory) | 共现事实硬编码到权重，类似查表 | 未训练过的组合无法匹配；反转诅咒 |
| **几何记忆** (Geometric Memory) | 在高维空间自发构建实体的全局几何结构 | 需要更长训练时间，但泛化能力强 |

关键：模型虽有查表能力，但在梯度下降的**谱偏差**(Spectral Bias) 驱动下，最终自发选择几何记忆。

### 2.2 训练集构建四律

从对话中提取的语料构建原则（由 Qwen 系统阐述）：

1. **双向边** (Reverse Edges): 同时提供 `(A, 关系, B)` 和 `(B, 关系, A)` — 打破反转诅咒
2. **自环边** (Self-edges): 加入 `(A, 是, A)` — 作为几何锚点，防止之字形扭曲
3. **交错混合** (Interleaving): 局部事实与全局推理任务同 batch 混合 — 防灾难性遗忘
4. **较高学习率 + 权重衰减**: 迫使模型放弃查表捷径，转向低秩几何结构

### 2.3 袜子与手套（手性拓扑）

主人的核心直觉：
- **袜子** (自反/对称): `(袜子, 是, 袜子)` — 自环锚点，几何稳定
- **手套** (手性/反对称): 左手↔右手 — 配对但不可互换，对应二分图/负特征值结构

→ 不同概念有不同的拓扑属性，语料应针对性地注入结构化样本。

### 2.4 Allen-Zhu 论文印证

《Physics of LMs Part 3.1》的个人简介实验：
- 单一模板训练 → QA 准确率 ≈ 0（查表，位置绑定）
- 知识增强（乱序+多句式+全名）→ QA 成功（构建了以实体为中心的星型几何结构）
- **不同问句（Token 完全不同）从不同方向探测同一个几何锚点**

---

## 三、实验设计（五步）

> 对应对话最后部分的实验方案，复用 transformer_vis 已有基础设施。

### 步骤 1: 构建控制变量对照数据集

**目标**: 剥离表层词汇干扰，只保留底层概念一致性。

| 样本组 | 设计 | 示例 |
|--------|------|------|
| **正样本组** (Same Concept, Different Tokens) | 同一概念，多种低词汇重叠表达 | "婴儿纸尿裤是新生儿的必备用品" vs "NB码云柔尿裤适合刚出生使用" vs "日夜两用的婴儿隔尿护理垫" |
| **负样本组** (Different Concepts) | 语义和词汇都不同的无关内容 | "学步车适合已经能扶站的宝宝" |
| **干扰样本组** (Same Tokens, Different Concepts) | 词汇高度重叠但语义不同 | "防滑袜适合室内学步" vs "不防滑的光脚更适合学步感统" |

**语料策略**: 基于 transformer_vis 已有的 `corpus_baby.json`（50 卡 × 5 阶婴幼儿商品），为每张卡片的 `(name, desc)` 生成 **3-5 个同义不同表**的改写句。改写规则：
- 词序打乱
- 同义词替换（纸尿裤↔隔尿裤↔护理垫）
- 句式变换（陈述句↔疑问句↔倒装句）
- 词汇重叠度 < 30%

**输出**: `corpus_paraphrase.json` — 格式兼容 transformer_vis 的 `{deck, stages, items}` 结构，每个 item 增加 `paraphrases: [str]` 字段。

### 步骤 2: 定义并提取几何表示

复用 transformer_vis 的模型和提取管线：

- **模型**: Qwen2.5-0.5B-Instruct (24L/896H/14H, bf16, MPS)
  - 路径: `~/Programming/code-2026/first-lora/model`
- **环境**: `/opt/miniconda3/envs/default` (torch 2.14, transformers 5.17, peft, sklearn, matplotlib, umap-learn)

**提取位置**: Transformer 每一层 (Layer 0 → Layer 24) 的隐藏状态 (Hidden States)

**聚合策略** (三种并行):
1. **Last-token Representation**: 序列末 Token 向量（模型"思考总结"状态）
2. **Mean Pooling**: 整序列 Token 向量均值（全局表示）
3. **Entity-centric Representation**: 核心实体 Token（如"纸尿裤"）在各层的表示

### 步骤 3: 量化指标与可视化

| 指标 | 方法 | 预期结果 |
|------|------|----------|
| **余弦相似度热力图** | 正样本组内两两余弦相似度 | 深层出现 block-diagonal 高亮区块 |
| **UMAP 降维** | 高维隐藏状态 → 2D/3D | 浅层分散；深层同概念坍缩为紧密簇 |
| **CKA (Centered Kernel Alignment)** | 不同 Token 序列的隐藏状态矩阵对齐度 | CKA → 1 表示相同内部几何结构 |
| **几何边距测试** | 仅靠余弦相似度区分同概念 vs 异概念变体 | 边距 > 0 |

**可视化工具**: 复用 transformer_vis 的 matplotlib + UMAP 管线

### 步骤 4: 注意力机制几何溯源

- **Attention Rollout / Heatmap**: 可视化模型生成答案首 Token 时的 Attention 权重
- **预期**: 无论输入是哪种改写句，Attention 权重都应聚焦并激活代表同一概念的几何锚点

### 步骤 5: 干预与消融实验

| 臂 | 训练策略 | 预期 |
|----|----------|------|
| **模型 A** (几何记忆主导) | 交错混合 + 双向边/自环 + 较高 LR + 权重衰减 | 深层表示高度重合（稳固几何簇） |
| **模型 B** (联想记忆主导) | 单一固定模板死记硬背 | 隐藏状态分散，QA 失败 |

复用 transformer_vis 的 LoRA 训练管线（`train_c42.py` 的 `train_arm()` 架构: LoRA r=16, α=32, 6 epochs, lr=1e-4, Adam β2=0.95）。

---

## 四、实施计划

### Phase 0: 环境就位

- [x] 软连接 transformer_vis 共享件（禁拷贝，直读路径）
- [ ] 验证 Python 环境 (`/opt/miniconda3/envs/default`)
- [ ] 验证模型加载 (`~/Programming/code-2026/first-lora/model`)

### Phase 1: 语料构建

- [ ] 编写 `build_paraphrase_corpus.py` — 基于 `corpus_baby.json` 生成同义改写
  - 每张卡片 3-5 个改写句
  - 词汇重叠度 < 30%
  - 输出 `corpus_paraphrase.json`
- [ ] 构建正样本组、负样本组、干扰样本组

### Phase 2: 表示提取

- [ ] 编写 `extract_representations.py`
  - 加载 Qwen2.5-0.5B-Instruct
  - 对所有改写句逐层提取 Hidden States
  - 三种聚合策略并行
  - 输出 `representations.npz` (per-layer × per-strategy)

### Phase 3: 量化与可视化

- [ ] 编写 `analyze_geometry.py`
  - 余弦相似度热力图 (per-layer)
  - UMAP 2D/3D 降维图 (per-layer)
  - CKA 对齐度计算
  - 几何边距测试
  - 输出 `metrics_geometry.json` + `figs_geometry/`

### Phase 4: 注意力溯源

- [ ] 编写 `attention_tracing.py`
  - Attention Rollout 可视化
  - 聚焦模式分析

### Phase 5: 消融对比

- [ ] 编写 `train_ablation.py`
  - 模型 A: 几何记忆导向训练
  - 模型 B: 联想记忆导向训练
  - 对比两者深层表示的坍缩程度
  - 输出 `metrics_ablation.json` + `figs_ablation/`

### Phase 6: 报告

- [ ] 编写 `report_geometry_of_meaning.qmd` (Quarto)
  - 整合所有指标和图表
  - 预期结论: "Prompt 是坐标探针，不是信息载荷"

---

## 五、共享件引用

| 件 | 正本路径 | 仓·pin | 消费法 |
|----|----------|--------|--------|
| transformer_vis | `/Users/mac/Programming/code-2026/Concept-Space-Sphere/benches/transformer_vis` | Concept-Space-Sphere @ `9638e7c` | 直读路径（同机, 首选）；`git remote add css ... && git fetch css` 取历史 |

**律**: 共享件只此一份、原地直读，禁重复拷贝（主人 10-15 裁定）。

---

## 六、冻结判据（先冻后用）

| ID | 判据 | 通过条件 |
|----|------|----------|
| G-1 | 同概念坍缩 | 深层(末 3 层)正样本组余弦相似度 > 0.75，且显著高于负样本组 |
| G-2 | UMAP 簇分离 | 深层 UMAP 图中，同概念不同表达坍缩为紧密簇，异概念分离 |
| G-3 | CKA 对齐 | 深层 CKA > 0.85（不同 Token 序列→相同内部几何结构） |
| G-4 | 消融对比 | 模型 A 深层坍缩度显著 > 模型 B（p < 0.05） |

**失败条款**: G-1 不过 → 该模型/语料条件下几何坍缩不成立，记录原始数据不强行裁决。

---

## 七、文件结构

```
Geometry-Of-Meaning/
├── AGENTS.MD                    # 指令正本
├── README.md                    # 本文件（实施计划）
├── external/
│   └── chat-export.json         # 对话原始数据
├── transformer_vis -> /Users/mac/Programming/code-2026/Concept-Space-Sphere/benches/transformer_vis  # 软连接
├── corpus_paraphrase.json       # Phase 1 输出: 改写语料
├── build_paraphrase_corpus.py   # Phase 1 脚本
├── extract_representations.py   # Phase 2 脚本
├── representations.npz          # Phase 2 输出: 逐层表示
├── analyze_geometry.py          # Phase 3 脚本
├── metrics_geometry.json        # Phase 3 输出: 几何指标
├── figs_geometry/               # Phase 3 输出: 图表
├── attention_tracing.py         # Phase 4 脚本
├── train_ablation.py            # Phase 5 脚本
├── metrics_ablation.json        # Phase 5 输出
├── figs_ablation/               # Phase 5 输出
└── report_geometry_of_meaning.qmd  # Phase 6: 报告
```
