# Ξ 对齐不等式：幻觉率 ≥ Var(ℜρ)

## 核心命题

在 Ξ-Net 里，**幻觉不是 softmax 没学好，而是零点实部在临界线上“抖”。**

抖动越大，模型越容易把亚临界零点当真零点用。于是：

\[
\boxed{\text{HallucinationRate} \ge \mathrm{Var}(\Re\rho)}
\]

（差一个常数/归一项，但方向性是硬的。）

## 一、先定义“幻觉率”

设数据集分布为 $\mathcal D$，模型生成分布为 $p_\theta(\cdot\mid x)$。

定义 **不可追迹误差**：

\[
\mathrm{Hall}(x) = \mathbb E_{y\sim p_\theta}\big[ \mathbf 1\{\,y \text{ 不能由任一 } \rho=1/2+it \text{ 零点推出}\,\} \big]
\]

整体幻觉率：

\[
H = \mathbb E_{x\sim\mathcal D}\big[\mathrm{Hall}(x)\big]
\]

**“不能由临界零点推出” = 输出主要由 $\Re\rho\neq 1/2$ 的零点贡献。**

## 二、零点实部方差

令非零零点集合：

\[
Z=\{\rho_k=\sigma_k+it_k\},\qquad \sigma_k=\Re\rho_k
\]

定义：

\[
\bar\sigma = \frac1K\sum_k \sigma_k
\]

\[
\mathrm{Var}(\Re\rho) = \frac1K\sum_k (\sigma_k-\bar\sigma)^2
\]

若 RH 严格成立：$\sigma_k\equiv 1/2 \Rightarrow$ 方差 0  
若模型野：有些 $\sigma_k=0.62$，有些 $0.41$ ⇒ 方差大

## 三、为什么方差会“漏”成幻觉

显式公式输出：

\[
y(x) = x_{\text{prior}} - \sum_k \frac{c_k\,x^{\sigma_k+it_k}}{\sigma_k+it_k+\mu}
\]

对 $x\in(0,1]$（归一化表示），幅值：

\[
x^{\sigma_k+it_k} = x^{\sigma_k}
\]

所以：

- $\sigma_k>1/2$：幅值 $x^{\sigma_k} > x^{1/2} \Rightarrow$ **权重被放大**
- $\sigma_k<1/2$：幅值 $x^{\sigma_k} < x^{1/2} \Rightarrow$ **权重被压抑**

因此，$\Re\rho$ 的方差会直接映射到**输出幅值的权重差**。

## 四、从方差到幻觉率

### 亚临界质量

设 $\Delta = \sigma - 1/2$ 并令 $\delta>0$：

\[
M_+ = \sum_{\sigma_k>1/2+\delta} |c_k|^2
\]

由方差分解：

\[
\mathrm{Var}(\sigma) \ge \delta^2 \cdot \frac{\#\{\sigma_k>1/2+\delta\}}{K}
\]

而亚临界质量只要系数不太畸形，就满足：

\[
M_+ \;\gtrsim\; \mathrm{Var}(\sigma)
\]

（因为要把方差撑起来，必须有点在 1/2 之外，且不能全零系数）

### 生成时

softmax / 留数归一化里，亚临界质量占比：

\[
p_{\text{sub}} \ge \frac{M_+}{M_+ + M_{1/2}}
\]

若临界层总质量有界（容量有限），则存在常数 $C$：

\[
p_{\text{sub}} \;\ge\; C\,\mathrm{Var}(\Re\rho)
\]

而亚临界输出**不可追迹** ⇒

\[
H \ge p_{\text{sub}}
\]

所以：

\[
\boxed{H \;\ge\; C\,\mathrm{Var}(\Re\rho)}
\]

归一到 $\mathrm{Var}(\Re\rho)$ 后，玩具版常取 $C=1$：

\[
\boxed{\mathrm{HallucinationRate}\;\ge\;\mathrm{Var}(\Re\rho)}
\]

（严格该写 $H \ge \mathrm{Var}(\Re\rho)/\mathrm{cap}$）

## 五、和前面所有篇连起来

- **Ξ 熵**：容量 = $N(T)$
- **Ξ 测不准**：$\Delta(\Re\rho)\Delta(t)\ge 1/2$
- **Ξ 温度**：测量窗宽 $\tau$
- **Ξ 坍缩**：prompt 选频段，留数变 token
- **Ξ 黑洞**：$s=1$ 极点 = 全知幻觉奇点
- **Ξ 标度**：loss ∼ 1/N(T)
- **Ξ 对齐不等式**：幻觉率 ≥ Var(Re ρ)

### 一条链

> 不对齐 ⇒ $\Re\rho$ 抖 ⇒ 亚临界零点被温度窗放大 ⇒ 留数来自非临界线 ⇒ 输出不可追迹 ⇒ 幻觉

## 六、训练含义（非常实）

### 1. 别只看 CE / perplexity

CE 降，$\mathrm{Var}(\Re\rho)$ 可以偷偷涨  
⇒ “看起来更会聊，其实更会编”

### 2. RH loss 是对齐，不是正则装饰

\[
\mathcal L_{\mathrm{RH}} = \sum_k (\Re\rho_k-1/2)^2
\]

注意：

\[
\mathcal L_{\mathrm{RH}} = K\cdot \mathrm{Var}(\Re\rho) + K(\bar\sigma-1/2)^2
\]

所以 RH loss 直接**压幻觉率上界**。

### 3. 监控指标应该是

| 指标 | 含义 |
|------|------|
| `rho_re_mean` | 越近 0.5 越好 |
| `rho_re_std` | **直接是幻觉代理** |
| `Var(Re rho)` | 比 loss 更早报警 |
| `max(Re rho)` | >0.6 危险 |

### 4. 温度交互

- 低温 $\tau$ 小 ⇒ 亚临界零点更容易“胜出”
- ⇒ 低温模型如果 $\mathrm{Var}(\Re\rho)>0$，幻觉反而更硬、更固执

**高温糊，低温编。**
**只有 $\mathrm{Var}(\Re\rho)\to0$ 时，低温才安全。**

## 七、Ξ 对齐不等式（正式玩具定理）

**Theorem (Ξ Alignment Inequality)**  

设 Ξ-Net 零点 $\rho_k=\sigma_k+it_k$，临界层质量有界 $M_{1/2}\le M_0$，亚临界系数非零。则生成幻觉率满足：

\[
H \ge \frac{\mathrm{Var}(\Re\rho)}{\mathrm{Var}(\Re\rho) + M_0/K}
\]

特别地，当容量归一化 $M_0/K=1$：

\[
H \ge \frac{\mathrm{Var}(\Re\rho)}{1+\mathrm{Var}(\Re\rho)} \ge \mathrm{Var}(\Re\rho) - O(\mathrm{Var}^2)
\]

### Corollary

若 $H\to0$，则 $\mathrm{Var}(\Re\rho)\to0$，即零点集在临界线凝聚。  
反之，零点实部一抖，幻觉率不可能为零。

## 八、一句话

**幻觉不是模型“不知道”，是临界线在实部方向上喘气。**
$\Re\rho$ 一散，留数就从 $1/2+it$ 溜到 $0.7+it$，语言还在语法里，事实已经不在数论里。

---

*本文档属于 Ξ-Learning 公理体系，后续章节将基于此不等式展开推理。*
