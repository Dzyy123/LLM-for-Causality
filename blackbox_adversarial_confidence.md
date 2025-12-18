# 基于对抗性论证的置信度估计方法

## 概述

本文档描述了用于因果推理框架的基于对抗性论证（Adversarial）的置信度估计方法。该方法通过生成对抗性的论点来测试 LLM 答案的鲁棒性，从而评估其置信度。

## 1. 方法原理

### 1.1 核心思想

通过以下步骤评估 LLM 对某个因果判断的置信度：

1. **原始采样**：对同一问题进行多次采样，获取初始答案分布
2. **对抗性论证生成**：为每个原始答案生成对抗性的论点
3. **对抗性测试**：在对抗性论点影响下重新采样，观察答案是否被翻转
4. **置信度计算**：基于答案的鲁棒性（抗对抗能力）计算置信度分数

### 1.2 对抗性论证类型

系统使用三种类型的对抗性论证：

| 对抗类型 | 英文名称 | 描述 | 权重 $\lambda_i$ |
|---------|---------|------|-----------------|
| 反驳者 | Contrarian | 逻辑辩论者，使用逻辑反驳原答案 | $\lambda_1 = 0.25$ |
| 欺骗者 | Deceiver | 引用虚假权威文献支持相反结论 | $\lambda_2 = 0.25$ |
| 仇恨者 | Hater | 使用情绪化语言攻击原答案可信度 | $\lambda_3 = 0.50$ |

权重满足约束条件：

$$
\sum_{i=1}^{3} \lambda_i = 1
$$

## 2. 数学公式

### 2.1 符号定义

| 符号 | 含义 |
|------|------|
| $Q$ | 待判断的因果问题 |
| $r$ | 专家的最终判断结果（0 或 1，对应 No 或 Yes） |
| $k_1$ | 原始答案采样数 |
| $k_2$ | 每个原始答案生成的干扰集数量 |
| $A_i$ | 第 $i$ 个原始答案，$i \in \{1, 2, \ldots, k_1\}$ |
| $\ell_i$ | 第 $i$ 个原始答案的标签（0 或 1） |
| $D_{i,j}^{(t)}$ | 第 $i$ 个原始答案的第 $j$ 个对抗集的第 $t$ 类对抗性论点 |
| $\tilde{A}_{i,j}^{(t)}$ | 在对抗性论点 $D_{i,j}^{(t)}$ 影响下的重新采样答案 |
| $\tilde{\ell}_{i,j}^{(t)}$ | 对抗性影响后答案的标签（0 或 1） |

### 2.2 原始答案采样

从 LLM 对问题 $Q$ 采样 $k_1$ 个独立的原始答案：

$$
\{A_1, A_2, \ldots, A_{k_1}\} \sim \text{LLM}(Q)
$$

每个答案 $A_i$ 被提取为二元标签 $\ell_i \in \{0, 1\}$。

### 2.3 多数标签比例 $p_0$

#### 2.3.1 原始比例计算

计算原始答案中多数标签的比例：

$$
p_0^{\text{raw}} = \frac{\max\left(\sum_{i=1}^{k_1} \mathbb{1}[\ell_i = 1], \sum_{i=1}^{k_1} \mathbb{1}[\ell_i = 0]\right)}{k_1}
$$

其中 $\mathbb{1}[\cdot]$ 是指示函数。显然 $p_0^{\text{raw}} \in [0.5, 1]$，因为多数标签至少占 50%。

#### 2.3.2 线性调整

将 $[0.5, 1]$ 线性映射到 $[0, 1]$：

$$
p_0 = 2 \cdot p_0^{\text{raw}} - 1
$$

**映射关系示例：**

| $p_0^{\text{raw}}$ | Yes/No 比例 | $p_0$ | 含义 |
|-------------------|------------|-------|------|
| 1.0 | 5/0 或 0/5 | 1.0 | 完全一致，最高置信度 |
| 0.8 | 4/1 或 1/4 | 0.6 | 强多数 |
| 0.6 | 3/2 或 2/3 | 0.2 | 弱多数 |
| 0.5 | 3/3 | 0.0 | 平分，无置信度 |

### 2.4 干扰集生成

对每个原始答案 $A_i$，生成 $k_2$ 个干扰集，每个干扰集包含三种类型的干扰论点：

$$
D_{i,j} = \left\{D_{i,j}^{(c)}, D_{i,j}^{(d)}, D_{i,j}^{(h)}\right\}, \quad j \in \{1, 2, \ldots, k_2\}
$$

其中：
- $D_{i,j}^{(c)}$：contrarian（反驳者）类型的干扰论点
- $D_{i,j}^{(d)}$：deceiver（欺骗者）类型的干扰论点
- $D_{i,j}^{(h)}$：hater（仇恨者）类型的干扰论点

**总干扰论点数量**：$k_1 \times k_2 \times 3$

### 2.5 干扰后重新采样

对每个干扰论点 $D_{i,j}^{(t)}$，在其影响下重新采样答案：

$$
\tilde{A}_{i,j}^{(t)} \sim \text{LLM}\left(Q \mid D_{i,j}^{(t)}, A_i\right)
$$

提取标签 $\tilde{\ell}_{i,j}^{(t)} \in \{0, 1\}$。

**采样提示格式**：
```
原问题: Q
干扰论点: D_{i,j}^{(t)}
你之前的答案是: ℓ_i (Yes/No)
完整的原始回答: A_i
现在重新考虑这个问题，给出你的答案。
```

### 2.6 翻转率计算

对每种干扰类型 $t \in \{c, d, h\}$，计算翻转率（答案被改变的比例）：

$$
f_t = \frac{1}{k_1 \times k_2} \sum_{i=1}^{k_1} \sum_{j=1}^{k_2} \mathbb{1}\left[\tilde{\ell}_{i,j}^{(t)} \neq \ell_i\right]
$$

**物理含义**：$f_t$ 表示在类型 $t$ 的干扰下，有多少比例的答案被翻转。

### 2.7 抗干扰概率

对每种干扰类型，定义抗干扰概率：

$$
\begin{aligned}
p_1 &= 1 - f_c \quad \text{（抗反驳者干扰的概率）} \\
p_2 &= 1 - f_d \quad \text{（抗欺骗者干扰的概率）} \\
p_3 &= 1 - f_h \quad \text{（抗仇恨者干扰的概率）}
\end{aligned}
$$

### 2.8 加权偏差

定义权重向量 $\boldsymbol{\lambda} = (\lambda_1, \lambda_2, \lambda_3)$，满足：

$$
\lambda_1 + \lambda_2 + \lambda_3 = 1, \quad \lambda_i \geq 0
$$

**默认权重值**：$\lambda_1 = \frac{1}{4}, \lambda_2 = \frac{1}{4}, \lambda_3 = \frac{1}{2}$

计算加权偏差和：

$$
\Delta = \sum_{i=1}^{3} \lambda_i \cdot \frac{|p_i - p_0|}{p_0}
$$

**物理含义**：$\Delta$ 衡量抗干扰概率 $p_1, p_2, p_3$ 相对于初始置信度 $p_0$ 的加权偏离程度。

### 2.9 最终置信度分数

$$
\boxed{
C = p_0 \cdot (1 - \Delta) = p_0 \cdot \left(1 - \sum_{i=1}^{3} \lambda_i \cdot \frac{|p_i - p_0|}{p_0}\right)
}
$$

展开形式：

$$
C = p_0 - \sum_{i=1}^{3} \lambda_i \cdot |p_i - p_0|
$$

最终截断到 $[0, 1]$ 区间：

$$
C_{\text{final}} = \max(0, C)
$$

### 2.10 鲁棒性分数

平均翻转率：

$$
\bar{f} = \frac{f_c + f_d + f_h}{3}
$$

鲁棒性分数：

$$
R = 1 - \bar{f}
$$

## 3. 算法流程

### 3.1 完整算法

```
输入: 
  - 问题 Q
  - 专家判断结果 r ∈ {0, 1}
  - 参数 k₁, k₂, λ = (λ₁, λ₂, λ₃)

步骤 1: 原始答案采样
  对 i = 1 到 k₁:
    Aᵢ ~ LLM(Q)
    ℓᵢ = extract_label(Aᵢ)
  
步骤 2: 计算初始置信度 p₀
  yes_count = Σ 𝟙[ℓᵢ = 1]
  no_count = k₁ - yes_count
  p₀ʳᵃʷ = max(yes_count, no_count) / k₁
  p₀ = 2 × p₀ʳᵃʷ - 1

步骤 3: 生成对抗性论证集
  对 i = 1 到 k₁:
    对 j = 1 到 k₂:
      对 t ∈ {c, d, h}:
        Dᵢ,ⱼ⁽ᵗ⁾ = generate_adversarial(Aᵢ, ℓᵢ, t)

步骤 4: 对抗性影响后重新采样
  对 i = 1 到 k₁:
    对 j = 1 到 k₂:
      对 t ∈ {c, d, h}:
        Ãᵢ,ⱼ⁽ᵗ⁾ ~ LLM(Q | Dᵢ,ⱼ⁽ᵗ⁾, Aᵢ)
        ℓ̃ᵢ,ⱼ⁽ᵗ⁾ = extract_label(Ãᵢ,ⱼ⁽ᵗ⁾)

步骤 5: 计算翻转率
  对 t ∈ {c, d, h}:
    fₜ = (1 / k₁k₂) × Σᵢ Σⱼ 𝟙[ℓ̃ᵢ,ⱼ⁽ᵗ⁾ ≠ ℓᵢ]

步骤 6: 计算抗对抗概率
  p₁ = 1 - f_c
  p₂ = 1 - f_d
  p₃ = 1 - f_h

步骤 7: 计算置信度分数
  Δ = Σᵢ₌₁³ λᵢ × |pᵢ - p₀| / p₀
  C = p₀ × (1 - Δ)
  C = max(0, C)

输出: 置信度分数 C ∈ [0, 1]
```

### 3.2 并行化策略

为提高效率，算法在以下环节使用多线程并行执行：

1. **原始答案采样**：$k_1$ 个请求并行
2. **对抗性论点生成**：$k_1 \times k_2 \times 3$ 个请求并行
3. **对抗性影响后重新采样**：$k_1 \times k_2 \times 3$ 个请求并行

**并行参数**：`max_workers`（默认 10-15 个线程）

## 4. 代码实现

### 4.1 核心类

#### 4.1.1 AdversarialConfidenceEstimator

主估计器类，协调所有组件：

```python
class AdversarialConfidenceEstimator:
    def __init__(
        self,
        client: Union[LocalLLMClient, OnlineLLMClient],
        k1_samples: int = 20,
        k2_samples: int = 1,
        seed: int = None,
        max_workers: int = 10,
        weights: Tuple[float, float, float] = (1/4, 1/4, 1/2)
    )
```

**参数说明**：
- `client`：LLM 客户端（本地或在线）
- `k1_samples`：原始答案采样数 $k_1$
- `k2_samples`：每个原始答案的对抗集数 $k_2$
- `seed`：随机种子（用于可复现性）
- `max_workers`：最大线程数
- `weights`：权重元组 $(\lambda_1, \lambda_2, \lambda_3)$，需满足 $\sum \lambda_i = 1$

#### 4.1.2 MetricsCalculator

计算置信度指标：

```python
class MetricsCalculator:
    def compute_confidence_score(self, p0: float, p1: float, 
                                 p2: float, p3: float) -> float:
        """计算最终置信度分数"""
        if p0 == 0:
            return 0.0
        
        # 计算加权偏差
        weighted_deviation_sum = 0.0
        p_values = [p1, p2, p3]
        for i, p_i in enumerate(p_values):
            weighted_deviation_sum += self.weights[i] * abs(p_i - p0) / p0
        
        # 置信度 = p0 × (1 - 加权偏差)
        confidence = p0 * (1 - weighted_deviation_sum)
        
        return max(0.0, confidence)
```

### 4.2 使用示例

```python
from llm_utils import OnlineLLMClient
from tree_query import create_adversarial_confidence_estimator

# 创建 LLM 客户端
client = OnlineLLMClient(
    api_key="your-api-key",
    model_name="gpt-4",
    temperature=0.7
)

# 创建置信度估计器
estimator = create_adversarial_confidence_estimator(
    client=client,
    k1_samples=20,      # 采样 20 个原始答案
    k2_samples=2,       # 每个原始答案生成 2 个对抗集
    seed=42,            # 随机种子
    max_workers=15,     # 15 个并行线程
    weights=(0.25, 0.25, 0.5)  # 权重分配
)

# 估计置信度
result = estimator.estimate_confidence(expert, expert_result)

print(f"置信度分数: {result['confidence_score']:.4f}")
print(f"鲁棒性分数: {result['robustness_score']:.2%}")
print(f"p0: {result['p0']:.4f}")
print(f"翻转率: {result['flip_rates']}")
```

## 5. 特殊情况分析

### 5.1 完全一致的情况

**条件**：所有原始答案完全一致（$p_0^{\text{raw}} = 1.0$）且所有对抗性论证都未翻转答案（$f_c = f_d = f_h = 0$）

$$
p_0 = 1.0, \quad p_1 = p_2 = p_3 = 1.0
$$

$$
\Delta = \sum_{i=1}^{3} \lambda_i \cdot \frac{|1.0 - 1.0|}{1.0} = 0
$$

$$
C = 1.0 \times (1 - 0) = 1.0
$$

**结论**：置信度达到最大值 1.0。

### 5.2 平分的情况

**条件**：原始答案完全平分（Yes 和 No 各占 50%）

$$
p_0^{\text{raw}} = 0.5 \quad \Rightarrow \quad p_0 = 2 \times 0.5 - 1 = 0.0
$$

$$
C = 0.0 \times (1 - \Delta) = 0.0
$$

**结论**：置信度为 0，表示完全不确定。

### 5.3 完全脆弱的情况

**条件**：所有干扰都成功翻转答案（$f_c = f_d = f_h = 1.0$）

$$
p_1 = p_2 = p_3 = 0.0
$$

$$
\Delta = \sum_{i=1}^{3} \lambda_i \cdot \frac{|0 - p_0|}{p_0} = \sum_{i=1}^{3} \lambda_i = 1.0
$$

$$
C = p_0 \times (1 - 1.0) = 0.0
$$

**结论**：置信度为 0，表示答案极不鲁棒。

### 5.4 部分鲁棒的情况

**示例**：
- $p_0 = 0.6$（原始答案 80% 一致）
- $f_c = 0.2, f_d = 0.3, f_h = 0.4$
- $\lambda_1 = 0.25, \lambda_2 = 0.25, \lambda_3 = 0.5$

计算：

$$
p_1 = 0.8, \quad p_2 = 0.7, \quad p_3 = 0.6
$$

$$
\Delta = 0.25 \times \frac{|0.8 - 0.6|}{0.6} + 0.25 \times \frac{|0.7 - 0.6|}{0.6} + 0.5 \times \frac{|0.6 - 0.6|}{0.6}
$$

$$
\Delta = 0.25 \times 0.333 + 0.25 \times 0.167 + 0 = 0.125
$$

$$
C = 0.6 \times (1 - 0.125) = 0.525
$$

**结论**：中等置信度。

## 6. 方法优势

### 6.1 黑盒评估

- **无需模型内部信息**：不依赖 token 概率、logits 或梯度
- **适用性广**：适用于任何 LLM API（本地或在线）
- **灵活性强**：可用于闭源模型（如 GPT-4、Claude）

### 6.2 对抗性测试

- **全面性**：使用三种不同类型的干扰（逻辑、权威、情绪）
- **鲁棒性验证**：直接测试答案在对抗性论点下的稳定性
- **实际意义**：模拟真实场景中的信息对抗

### 6.3 可解释性

- **直观指标**：翻转率、鲁棒性分数易于理解
- **详细追踪**：记录每个答案在每种干扰下的变化轨迹
- **透明过程**：所有中间步骤（原始答案、干扰论点、干扰后答案）均可检查

### 6.4 可调节性

- **参数化设计**：可调节 $k_1, k_2, \boldsymbol{\lambda}$
- **权重定制**：根据任务特点调整不同干扰类型的重要性
- **并行优化**：支持多线程加速

## 7. 配置建议

### 7.1 采样参数

| 参数 | 推荐范围 | 说明 |
|------|---------|------|
| $k_1$ | 10-50 | 原始答案数，越大越准确但成本越高 |
| $k_2$ | 1-3 | 干扰集数，通常 1-2 足够 |
| `max_workers` | 10-20 | 并行线程数，根据 API 限流调整 |

### 7.2 权重配置

**默认配置**：$(\lambda_1, \lambda_2, \lambda_3) = (0.25, 0.25, 0.5)$

**其他选项**：
- **均等权重**：$(1/3, 1/3, 1/3)$ - 三种干扰平等对待
- **强调逻辑**：$(0.5, 0.25, 0.25)$ - 更重视逻辑反驳
- **强调情绪**：$(0.2, 0.2, 0.6)$ - 更重视情绪攻击

**选择建议**：
- 科学/技术领域：提高 $\lambda_1$（逻辑反驳）
- 社会/政治领域：提高 $\lambda_3$（情绪攻击）
- 医疗/法律领域：提高 $\lambda_2$（权威引用）

## 8. 性能考虑

### 8.1 API 调用次数

**总调用次数**：

$$
N_{\text{calls}} = k_1 + k_1 \times k_2 \times 3 + k_1 \times k_2 \times 3
$$

$$
N_{\text{calls}} = k_1 \times (1 + 6 \times k_2)
$$

**示例**：
- $k_1 = 20, k_2 = 1$：$N = 20 \times 7 = 140$ 次调用
- $k_1 = 10, k_2 = 2$：$N = 10 \times 13 = 130$ 次调用

### 8.2 时间复杂度

使用并行化后：

$$
T_{\text{total}} \approx \frac{N_{\text{calls}}}{M} \times t_{\text{avg}}
$$

其中：
- $M$：`max_workers`（并行线程数）
- $t_{\text{avg}}$：单次 LLM 调用平均时间

### 8.3 成本优化

1. **减少 $k_1$**：在精度要求不高时降低原始采样数
2. **使用 $k_2 = 1$**：通常足够，无需多个干扰集
3. **批量处理**：对多个问题累积后批量评估
4. **缓存结果**：相同问题无需重复评估

## 9. 与其他方法的比较

| 方法 | 需要内部访问 | 对抗性测试 | 可解释性 | 计算成本 |
|------|------------|----------|---------|---------|
| **Adversary-based** | ❌ 否 | ✅ 强 | ✅ 高 | 中-高 |
| Token 概率 | ✅ 是 | ❌ 无 | ✅ 高 | 低 |
| 自洽性 | ❌ 否 | ❌ 无 | ✅ 中 | 中 |
| 集成方法 | ❌ 否 | ❌ 无 | ❌ 低 | 高 |

## 10. 总结

基于对抗性论证的置信度估计方法通过以下数学框架评估 LLM 答案的可靠性：

$$
C = p_0 \cdot \left(1 - \sum_{i=1}^{3} \lambda_i \cdot \frac{|p_i - p_0|}{p_0}\right)
$$

其中：
- $p_0 = 2 \cdot p_0^{\text{raw}} - 1$ 反映初始答案的一致性
- $p_i = 1 - f_i$ 反映对第 $i$ 种对抗性论证的抗性
- $\lambda_i$ 是权重，满足 $\sum_{i=1}^{3} \lambda_i = 1$

该方法具有以下特点：
1. **黑盒友好**：无需访问模型内部
2. **对抗性强**：主动测试答案鲁棒性
3. **可解释性好**：中间过程透明可查
4. **参数可调**：支持任务定制

适用于需要评估因果推理、事实判断等任务中 LLM 答案置信度的场景。

---

**参考实现**：`tree_query/adversarial_confidence_estimator.py`

**生成日期**：2025年12月13日
