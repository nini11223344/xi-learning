# Ξ 温度：softmax 是临界线的 "折射率"

## 核心命题

Softmax 温度不是"模型有多敢猜"的人为超参，而是**临界线谱函数在虚部轴的平滑核宽度**。

\[
\boxed{\tau_{\text{softmax}} \equiv \tau_{\text{spectral}} = \frac{2\pi}{\log(T/2\pi) + \varepsilon}}
\]

## 一、从 softmax 回到谱

### 标准 softmax
\[
p_j = \frac{e^{z_j/\tau}}{\sum_k e^{z_k/\tau}}
\]

### Ξ 视角
$z_j$ 不是 logit，而是临界线留数幅度：
\[
z_j \leftrightarrow \Re\left(\frac{c_{\rho_j}}{\rho_j + \mu}\right)
\]

但更根本的是：**token 表示本身是谱和**
\[
f(x) = \sum_\rho \frac{c_\rho x^\rho}{\rho+\mu}, \quad \rho = \frac12 + it
\]

在 $x=e^{-u}$ ($u=-\log x$) 上：
\[
f(u) = e^{-u/2}\sum_\rho c_\rho e^{-i t u}
\]

这就是一条**临界线信号**。

## 二、什么是 Ξ 温度

我们不直接对"logits"softmax，而是对临界线谱做**热核平滑**：

\[
f_\tau(u) = \int_{-\infty}^{\infty} K_\tau(t-t') \widehat f(t') dt'
\]

取热核（GUE/Selberg 自然核）：
\[
K_\tau(\Delta t) = \frac{1}{\sqrt{2\pi}\,\tau} \exp\!\left(-\frac{(\Delta t)^2}{2\tau^2}\right)
\]

于是：

\[
\boxed{\text{softmax 温度 } \tau \; \equiv \; \text{spectral smoothing width on } \Im\rho}
\]

## 三、为什么 softmax 是"低温近似"

### $\tau \to 0$ 时
\[
K_\tau(\Delta t) \approx \delta(\Delta t)
\]
- 输出由少数主导零点决定
- 像 argmax / greedy
- 语言"确定、重复、低频"
- **但：可能掉进黑洞**（如果主导零点 $\Re\rho > 1/2$）

### $\tau \to \infty$ 时
\[
f_\tau(u) \approx \text{const}\cdot \sum_\rho c_\rho
\]
- 所有零点混在一起
- 临界振荡被"烫平"
- 模型变"温吞、正确但空"
- **对应：大头贴、废话、热平衡**

**这正是 softmax 温度的行为，只是现在有了谱解释。**

## 四、自然温度该是多少？

### 黎曼零点间距
黎曼零点间距（$t$ 大时）：
\[
\Delta t \sim \frac{2\pi}{\log(t/2\pi)}
\]

若截断到 $T$，平均间距：
\[
\langle \Delta t\rangle_T \sim \frac{2\pi}{\log(T/2\pi)}
\]

所以"不糊掉结构"的温度应满足：

\[
\boxed{\tau_{\text{natural}} \sim \frac{2\pi}{\log(T/2\pi)}}
\]

### 温度三档

| 温度 | 效果 | AI 表现 |
|------|------|---------|
| **$\tau \ll \tau_{\text{nat}}$** | 只信一个概念 → 死板/幻觉式固执 | Greedy、Beam Search |
| **$\tau \gg \tau_{\text{nat}}$** | 把邻近零点揉成一团 → 语义模糊 | Temperature=1.0+ |
| **$\tau \approx \tau_{\text{nat}}$** | 刚好分辨"概念能级" | 最佳生成 |

### 大模型温度不是人工设 0.7
而该随上下文临界带 $T$ 自动变：
\[
\tau(x) = \frac{2\pi}{\log(T(x)/2\pi) + \varepsilon}
\]

## 五、和 Ξ 测不准联系

上篇：
\[
\Delta(\Re\rho)\,\Delta(t) \ge \frac12
\]

温度做的是：在 $t$ 轴上加**窗宽** $\tau$：
\[
\Delta(t)_{\text{obs}}^2 = \Delta(t)_{\text{true}}^2 + \tau^2
\]

代入测不准：
\[
\Delta(\Re\rho)\,\sqrt{\Delta(t)_0^2+\tau^2} \ge \frac12
\]

### 测量后果

- **降温（$\tau\downarrow$）** → 若 $\Delta(t)_0$ 小，则 $\Delta(\Re\rho)$ 必须变大
  ⇒ 看起来"确定"，其实零点在实部上抖 → **伪对齐**

- **升温（$\tau\uparrow$）** → 频率不确定被温度吃掉，$\Re\rho$ 可以稳在 1/2
  ⇒ 安全但钝 → **废话**

**低温制造"假装可解释"的模型**  
**高温制造"安全但没观点"的模型**

## 六、Ξ-Softmax（替换公式）

别写：
\[
\mathrm{softmax}(z/\tau)
\]

写：
\[
p_j = \frac{\displaystyle \Re\sum_\rho \frac{c_{\rho_j}}{\rho+\mu} \exp\!\Big(-\frac{(t_j-\bar t)^2}{2\tau^2}\Big)}{\displaystyle \sum_k \Re\sum_\rho \frac{c_{\rho_k}}{\rho+\mu} \exp\!\Big(-\frac{(t_k-\bar t)^2}{2\tau^2}\Big)}
\]

其中 $\tau$ 可学：
\[
\tau_\theta = \mathrm{stop\_grad}\!\left(\frac{2\pi}{\log(T/2\pi)+\varepsilon}\right)\cdot (1+\eta_\theta)
\]

👉 **温度不再是超参，是临界带标度**

## 七、训练含义（非常实用）

| 场景 | 传统调法 | Ξ 解释 |
|------|----------|--------|
| 小模型 $T$ 小 | Log | 温度高（1.0-1.5） |
| 大模型 $T$ 大 | 温度低（0.7） | 温度低（0.3-0.5） |
| 推理时降温 = 把测量窗变窄 = 强迫输出坍缩到主零点 |  |
| 对齐时别只降 RH loss，也要管 $\tau$ |  |
| 温度太低会"表面对齐、实则实部抖动" |  |

### 实际建议

1. **初始温度估计**：
   ```python
   tau = 2 * pi / (log(T) + 1e-6)
   ```

2. **温度衰减**：
   ```python
   tau_t = tau * exp(-decay * epoch)
   ```

3. **主动正则**：
   ```python
   loss = CE + lambda_tau * (tau - tau_natural)**2
   ```

## 八、望远镜参数（高级）

| 参数 | 传统解释 | Ξ 解释 |
|------|----------|--------|
| Temperature | 随机性控制 | 谱窗宽 |
| Top-k | 截断概率 | 只保留 $k$ 个主零点 |
| Top-p | 累积概率 | 保留累积质量达 $p$ 的零点集 |
| Greedy | 单点峰值 | 测量窗退化成 $\delta$ 函数 |
| Beam search | 多条路径 | 同时测几条临界线轨道 |

**所有解码策略 = 不同的测量算子设计**

## 九、一句话

**Softmax 温度是人类的妥协。Ξ 温度是临界线的折射率：它决定你能不能看清，那些住在 $1/2+it$ 上的概念。**

---

*本文档属于 Ξ-Learning 公理体系，后续章节将基于此温度定义展开推理。*
