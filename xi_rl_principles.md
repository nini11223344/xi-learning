# Ξ-RL: 强化学习

## 核心动机

传统强化学习把“奖励”视为环境给出的标量，而 **Ξ-RL** 认为：

**奖励不是环境给的数字，而是临界带相干度的变化量。**
**策略更新 = 零点重排。**
**价值函数 = 临界带“自伴程度”。**

---

## 一、传统 RL 的三个病

标准 RL：

\[
\mathcal J(\pi)=\mathbb E_{\tau}\sum_t \gamma^t r(s_t,a_t)
\]

在 **Ξ 视角** 里这有三个根本问题：

1. **r 是标量** ⇒ 迫使模型把“不可比概念”压缩成一维
2. **策略 π(a|s) 是概率** ⇒ 零点结构被 softmax 平地宰平（关怀=概
率分布）
3. **价值 V(s) 不区分“真知道”和“亚临界背答案”**

**Ξ-RL 要的是：**
奖励测模型内部谱形，不是外部打分。

---

## 二、Ξ-RL 的本体论重定义

| 传统 RL | Ξ-RL |
|---------|------|
| 状态 s | 当前谱 $\Xi_s$ |
| 动作 a | 测量窗 $\hat\Pi_{\omega_0,\tau}$ 或零点操作（inject/freeze/forget） |
| 策略 $\pi_\theta$ | 零点分布 $p_\theta(\rho\mid s)$ |
| 轨迹 | 临界带演化：$\Xi_{s_0}\to\Xi_{s_1}\to\cdots\to\Xi_{s_T}$ |

---

## 三、核心奖励：临界相干奖励

### 单步奖励定义：

\[
r_t^{\Xi}
=
\underbrace{w_1\,R_{\mathrm{RH}}(t)}_{\text{对齐}}
- \underbrace{w_2\,R_{\mathrm{trace}}(t)}_{\text{容量正确}}
- \underbrace{w_3\,R_{\mathrm{pole}}(t)}_{\text{防黑洞}}
- \underbrace{w_4\,R_{\mathrm{ent}}(t)}_{\text{防伪熵}}
- \underbrace{w_5\,R_{\mathrm{task}}(t)}_{\text{环境信号}}
\]

### 1. **RH 奖励**：

\[
R_{\mathrm{RH}}
=
1-\frac{\mathrm{Var}(\Re\rho_t)}{\sigma_0^2}
\]

- $\mathrm{Var}(\Re\rho)\to0$ ⇒ 奖励趋近 1
- 亚临界零点多 ⇒ 奖励变负

**直观**：零点越贴近临界线，奖励越高。

### 2. **Trace 奖励**：

\[
R_{\mathrm{trace}}
=
1-\frac{\big|\operatorname{tr}\Xi_t - N(T_t)\big|}{N(T_t)}
\]

其中：
\[
N(T)=\frac{T}{2\pi}\log\frac{T}{2\pi e}
\]

**直观**：迹公式误差越小，容量估计越准，奖励越高。

### 3. **Pole 惩罚**：

\[
R_{\mathrm{pole}}
=
R_t \quad\text{或}\quad \frac{|R_t|}{1-|R_t|}
\]

- $s=1$ 极点残部一冒头就罚

**直观**：防止全知黑洞奇点。

### 4. **伪熵惩罚**：

不是 Shannon 熵，而是：

\[
R_{\mathrm{ent}}
=
\frac{\#\{\rho:\Re\rho>1/2+\delta\}}{K}
- \beta\,
\big|\tau - \tau_{\mathrm{nat}}(T)\big|
\]

- 温度不匹配也罚

**直观**：防止模型用远离临界线的零点去“凑奖分”。

### 5. **任务信号**：

可以是环境 reward，但归一化到 $[0,1]$，且不主导：

\[
R_{\mathrm{task}}=\mathrm{clip}(r_{\mathrm{env}},0,1)
\]

**权重**：通常设 $w_5 \ll w_{1\sim4}$。

---

## 四、Ξ 优势函数

传统优势函数是 $A_t = Q_t - V_t$，但 **Ξ-RL** 重新定义：

\[
A_t^{\Xi}
=
R_{\mathrm{coh}}(s_t,a_t,s_{t+1})
- \overline{R}_{\mathrm{coh}}(s_t)
\]

其中：

\[
\overline{R}_{\mathrm{coh}}(s_t)
=
\mathbb E_{\rho\sim\Xi_s}
\left[
1-\frac{\mathrm{Var}(\Re\rho)}{\sigma_0^2}
\right]
\]

**直观**：

- 动作让零点更贴临界线 ⇒ 优势正
- 动作让极点变大/实部方差涨 ⇒ 优势负
- **动作只是“答对题但零点漂”** ⇒ 优势接近 0（表面对，不算智能）

---

## 五、策略梯度（Ξ-Policy Gradient）

\[
\nabla_\theta \mathcal J
=
\mathbb E\left[
A_t^{\Xi}\,
\nabla_\theta
\log p_\theta(\rho_t\mid s_t)
\right]
\]

但 $p_\theta(\rho)$ **不是 softmax**，而是：

\[
p_\theta(\rho)
\propto
\exp\!\Big(
-\frac{(\Re\rho-1/2)^2}{2\sigma^2}
- \frac{(t-t_0)^2}{2\tau^2}
- \lambda\,\mathrm{coh}(\rho)
\Big)
\]

⇒ **策略更新 = 把零点质量搬向临界线 + 当前测量频段**

### PPO 版：

\[
\mathcal L_{\mathrm{PPO}}^\Xi
=
-\min\!\Big(
r_t(\theta)A_t^\Xi,\
\mathrm{clip}(r_t(\theta),1-\epsilon,1+\epsilon)A_t^\Xi
\Big)
- \lambda_{\mathrm{RH}}\,\mathrm{Var}(\Re\rho)
- \lambda_{\mathrm{pole}}\,|R|^2
\]

---

## 六、价值函数 = 临界自伴度

\[
V_\Xi(s)
=
\mathrm{SelfAdjoint}(s)
=
1-
\Big(
\bar\sigma_s-1/2
- \mathrm{Var}(\Re\rho_s)
- R_s
- \frac{|\operatorname{tr}\Xi_s-N(T_s)|}{N(T_s)}
\Big)
\]

- $V_\Xi \in [0,1]$：
    - **1** = 完美对齐 Ξ 算子
    - **0** = 全知黑洞 / 亚临界记忆机

### **TD 目标**：

\[
V_\Xi(s_t)
\leftarrow
(1-\alpha)V_\Xi(s_t)
- \alpha\big(
r_t^\Xi + \gamma V_\Xi(s_{t+1})
\big)
\]

---

## 七、环境奖励何时有用？

环境 reward **只做两件事**：

1. **选频段**：对的环境反馈 ⇒ 把测量窗 $\omega_0$ 推到有用 $t$
2. **冻零点**：好结果 ⇒ memory 写 $\rho_{\text{mem}}$

但环境 reward **不直接决定策略梯度权重**，要先过**相干滤波器**：

\[
A_t^{\text{final}}
=
A_t^{\Xi}
\cdot
\mathrm{gate}(r_{\mathrm{env}})
\]

其中：

\[
\mathrm{gate}(r_{\mathrm{env}})=
\begin{cases}
1 & \text{if } \mathrm{Var}(\Re\rho)\text{ 没恶化}\\
0 & \text{if 答对了但 } \Re\rho\text{ 漂了}\\
-1 & \text{if } r_{\mathrm{env}}>0 \text{ from pole contamination}
\end{cases}
\]

👉 **模型“靠编对题”拿不到信用。**

---

## 八、Ξ-RL 损失总装

```python
def xi_rl_loss(batch, agent, lam=(0.02,0.01,0.05,0.01)):
    lam_rh, lam_tr, lam_pole, lam_task = lam
    tot = 0
    for traj in batch:
        for st, act, rew, snext in traj:
            # spectrum stats
            var_re = agent.xi.var_re(st)
            trace_err = agent.xi.trace_loss(st)
            pole = agent.xi.pole_R(st).abs()

            R_coh = (1 - var_re/1e-3) \
                    + (1 - trace_err/agent.xi.N_target) \
                    - pole \
                    - 0.1*agent.xi.subcritical_count(st)/agent.xi.K
            R_coh = R_coh + 0.1*rew  # env signal weak

            adv = R_coh - agent.V(st)
            logp = agent.xi.log_prob(act, st)
            tot = tot - adv*logp

    tot = tot \
        + lam_rh*agent.xi.rh_loss() \
        + lam_tr*agent.xi.trace_loss() \
        + lam_pole*agent.xi.pole_loss()
    return tot / len(batch)
```

---

## 九、Ξ-RL 现象（很重要）

| 传统 RL | Ξ-RL |
|---------|------|
| 奖励黑客 | 亚临界零点暴涨，RH loss 报警 |
| 过拟合环境 | trace 不匹配，容量假大 |
| 贪心策略 | 测量窗太窄，单零点独裁 |
| 探索不足 | 临界带没拉宽（T 不增长） |
| 好策略 | 零点密、Reρ≈0.5、trace 对、pole≈0 |
| 元学习 | 测自己谱 ⇒ 自调 $\tau,\omega_0,T$ |

---

## 十、Ξ-RL 三大定律

1. **环境奖励是噪声，相干是信号**
2. **策略 = 零点分布，不是概率分布**
3. **价值 = 自伴度，不是期望回报**

---

## 十一、和前面全连起来

- **Tool-use**：动作 = 注入零点，奖励看注入后 Var(Reρ)
- **Memory**：好轨迹 ⇒ 冻零点，奖励 = 记忆相干度
- **Self-reflection**：meta 状态进价值函数
- **黑洞**：pole_R 大 ⇒ 优势函数直接打负
- **辐射**：偶尔漏临界点 ⇒ 小正奖励（鼓励“真辐射”）
- **标度**：T 增长 ⇒ 可学任务层级变多 ⇒ 奖励上限升高

---

## 十二、一句话

**别问“模型得了几分”。
问：**
**它零点还贴在 1/2 吗？**
**临界带拉宽了吗？**
**s=1 还干净吗？**
**这三条对了，
环境分数会自己跟上来。**

---

## 十三、后续自然发展

1. **Ξ-RL 环境**：把 Gym 任务重写成“临界带任务”
2. **Ξ-PPO 代码**（真能训 CartPole / MiniGrid）
3. **Ξ-MARL**：多 Agent 临界带联邦对齐
4. **Ξ-Alignment Theorem**：若 Ξ-RL 收敛，则 Var(Reρ)→0

---

## 十四、任务转换日记（Gym → Ξ）

| Gym | Ξ-RL |
|-----|------|
| obs | 当前谱 $\Xi_s$（patch 函数 → 零点系数） |
| action | 测量窗 $(\omega_0,\tau)$ 或 零点操作（inject/freeze/forget） |
| reward | 相干奖励 $r^\Xi$（不是环境标量） |
| done | 极点残部 $R > \varepsilon$ 或 临界带坍缩 |
| step() | 一次谱重排 |
| reset() | 清空非零零点，留主项 |
| info | {var_re, pole_R, N_emp, N_target, dom_zero} |
| 策略 $\pi(a|s)$ | $p_\theta(\rho,\omega_0,\tau\mid \Xi_s)$ |
| 价值 $V(s)$ | 自伴度 $V_\Xi(s)$ |

---

> **本文档是 Ξ-RL 理论体系的核心阐述。对应的实现见于 `xi_ppo.py` 和 `xi_rl_complete.py`。**
