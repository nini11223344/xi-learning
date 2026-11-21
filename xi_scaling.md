# Ξ 标度律：loss ∼ 1/N(T)

## 核心命题

Transformer 的神话是"参数越多越聪明"。Ξ 的真相是：**临界线越长，才越聪明**；参数只是把零点写进纸里。

\[
\boxed{\mathcal L \sim \frac{1}{N(T)} = \frac{2\pi}{T\log(T/2\pi e)}}
\]

## 一、先复习真标度

### 经验 Scaling Law（Kaplan/Henighan）
\[
L(N_{\text{param}}) \approx \frac{c}{N_{\text{param}}^\alpha} + \text{irreducible}
\]

它描述的是：
**"多层矩阵里塞更多系数 ⇒ 拟合更多相关结构"**

### Ξ 视角
- 权重矩阵 ≠ 知识
- **谱零点 $\rho_k$ = 概念本征模式**
- 能表示的概念数 ≈ 临界带零点数 $N(T)$

## 二、Ξ 容量：不是 $d$，是 $T$

### 前面定义
\[
N(T) = \#\{\rho : 0 < \Im \rho \le T\}
\]

### 大小参数对比

| 参数 | 作用 | 限制 |
|------|------|------|
| 隐维 $d$ | 存零点坐标 | 只能存，不能生 |
| 层数 $L$ | 存更多零点 | 同样只能存 |
| **$T$** | **增加新频段** | **真增量** |
| 参数量 $P$ | 存零点 | 线性增长 |

- **参数加倍** ⇒ 能存更多 $\rho$，但不一定用
- **$T$ 加倍** ⇒ 临界带拉长 ⇒ **新频段（新抽象层级）出现**

## 三、Ξ 标度律推导（玩具但自洽）

### 1. 输出 = 显式公式截断
\[
f_T(x) = x - \sum_{k=1}^{N(T)} \frac{x^{\rho_k}}{\rho_k} + R_T(x)
\]

### 2. 截断误差
\[
|R_T(x)| \lesssim \frac{x^{1/2}}{T}
\]

### 3. 每个新零点平均摊到的误差
\[
\frac{1}{N(T)} \cdot \frac{1}{T} \sim \frac{1}{T^2\log T}
\]

### 4. 任务损失由"没被零点覆盖的频率"决定，不是由单个零点
\[
\mathcal L \sim \frac{1}{N(T)}
\]

于是：

\[
\boxed{\mathcal L(T) = \frac{C}{\frac{T}{2\pi}\log\frac{T}{2\pi e}} - o\!\left(\frac{1}{T\log T}\right)}
\]

## 四、参数去哪了？

设：
- 隐维 $d$
- 层数 $L$
- 参数量 $P \sim L d^2$

Ξ 模型里这些参数做两件事：

1. **存零点位置**：$t_k, \Re\rho_k$ ～ $O(P)$ 个标量
2. **存本征幅度**：$c_{\rho_k}$ ～ 也是 $O(P)$

但有效概念数受谱结构限制：
\[
\text{usable zeros} \le N(T_{\text{arch}})
\]

而
\[
T_{\text{arch}} \sim \text{context depth} \times \text{patch arithmetic range}
\]

**不是 $d$ 单纯变大就有用。**

👉 所以：

> **加参数若不加临界带宽度 ⇒ 过配不智能**  
> **加 $T$（更长上下文 / 更细 patch / 更深显式公式）⇒ 真降损**

## 五、Ξ Scaling 定律（正式玩具版）

**Theorem (Ξ Scaling)**  

若 Ξ-Net 全零点在临界线，截断到虚部 $T$，则对于同分布生成任务：

\[
\mathbb E[\mathcal L] = \frac{2\pi C}{T\log(T/2\pi e)} - O\!\left(\frac{\log T}{T^2}\right)
\]

等价写成：

\[
\boxed{\mathcal L \cdot N(T) = \Theta(1)}
\]

### 若零点偏离临界线 $\Re\rho = \sigma > 1/2$
表观容量变：
\[
N_\sigma(T) \sim T^\sigma \log T
\]

但泛化误差退化：
\[
\mathrm{gen\_gap} = \Omega(T^{\sigma-1/2})
\]

⇒ **参数变多只是把零点"推离临界线"，loss 表面降、测试不降。**

## 六、和 Chinchilla 的冲突/统一

### Chinchilla
"**给定算力，参数和 data 应同比例**"

### Ξ 版
给定算力，应分配去：
- **拉临界带 $T$**（上下文/算术距离/显式公式深度）
- **不是无脑堆 $d$**

### 预测对比

| 传统 | Ξ |
|------|---|
| 大模型 = 大矩阵 | 大模型 = 宽临界带 |
| 长上下文靠 attention 内存 | 长上下文 = 更高 $T$ 零点计数 |
| MoE 增参 | MoE = 多支 L-函数，各有一条临界线 |
| 数据越多越好 | 数据要"激发新频段 $t$"才有用 |
| $\mathcal L \propto \text{param}^{-\alpha}$ | $\mathcal L \propto 1/N(T)$ |

### 未来验证
如果某天有人发现：
**"同样参数量，显式公式深度/上下文算术范围更能预测 loss"**
——那就是 **Ξ Scaling** 在现实里露头。

## 七、为什么 $1/N(T)$ 比 $1/P$ 更对

### 参数可以冗余
- 两个头学同一个 $t$
- MLP 把 $\rho$ 存两遍
- LayerNorm 不改谱

### 但 $N(T)$ 是谱计数
- 多一个零点 ⇒ 多一个振荡模式
- 少一个频段 ⇒ 某些抽象关系永远表达不出
- loss 的下界由"未覆盖的 $t \in [0,T]$"决定

所以：

\[
\boxed{\text{智能的上界由临界带决定，不由参数量决定}}
\]

## 八、Ξ 标度三定律（口诀）

| 定律 | 说明 | 公式 |
|------|------|------|
| 1. 参数存零点，不产零点 | 增加参数不一定增加知识 | $P \uparrow \not\Rightarrow \mathcal L \downarrow$ |
| 2. 上下文拉 $T$，不拉 $d$ | 增加 $T$ 真降损 | $\mathcal L \sim 1/N(T)$ |
| 3. loss 跟 $N(T)$ 走，不跟 $P$ 走 | 记忆容量决定表现 | $\mathcal L \cdot N(T) \approx C$ |

数学版：
\[
\frac{d\log \mathcal L}{d\log P} \to 0 \quad\text{as } T \text{ saturates}
\]

\[
\frac{d\log \mathcal L}{d\log T} = -\left(1+\frac{1}{\log(T/2\pi e)}\right)
\]

## 九、一句话

| 范式 | 智能决定因素 |
|------|--------------|
| Transformer 神话 | 参数越多越聪明 |
| Ξ 真相 | 临界线越长，才越聪明 |
| 参数作用 | 只是把零点写进纸里 |

---

*本文档属于 Ξ-Learning 公理体系，后续章节将基于此标度律展开推理。*
