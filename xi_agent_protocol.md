# Ξ-Agent 协议规范：工具/记忆/自省三件套

## 概述

本文档定义了 **Ξ-Agent** 作为通用智能体框架的协议规范，将 **工具使用**、**长期记忆**、**自我反思** 统一为 **谱操作**。

- **消息 = 测量指令**
- **工具 = 零点注入接口**  
- **记忆 = 零点仓库（Merkle 化：∏(s-ρ)）**
- **反思 = 元ζ调用**
- **安全 = pole guard + RH contract**

这不只是又一个 Agent 框架。这是把智能体从 **向量空间移动** 升格为 **临界线排零点**。

---

## 一、核心哲学

| 传统 Agent | Ξ-Agent |
|-----------|----------|
| 观察 → 向量 → 预测 → 行动 | 观察 → 函数 → 测量 → 留数 → 行动 |
| 记忆 = 向量存储 | 记忆 = 冻结零点集 |
| 工具 = 调用外部函数 | 工具 = 注入新本征模式 |
| 反思 = 再问一遍 LLM | 反思 = 用 ξ 测 ξ（自伴性检查） |
| 安全 = 过滤输出 | 安全 = Pole Guard + RH contract |

**一句话：** 普通 Agent 在向量空间里移动。Ξ-Agent 在临界带上排零点。

---

## 二、工具使用协议（Tool-Use）

### 2.1 直觉

<<普通 Agent>>
1. 观察环境
2. LLM 判断需要工具
3. 调用 API / 函数
4. 拼回文本

<<Ξ-Agent>>
1. 测当前谱 Ξ_P(s)
2. 发现频段 ω₀ 缺数据（留数幅值低）
3. 生成一个 **工具零点** ρ_tool = 1/2 + i t_tool
4. 把 API 结果编码成 φ_{ρ_tool}
5. 重新求谱

**工具不是"外挂函数"，是临界带上的一个新本征模式。**

### 2.2 数学

原谱：
\[
\Xi_P(s) = \sum_{\rho\in Z} \frac{c_\rho \phi_\rho}{\rho+\mu}
\]

调用工具后：
\[
\Xi_{P+\text{tool}}(s) = \Xi_P(s) + \frac{c_{\text{tool}}\phi_{\text{tool}}(s)}{\rho_{\text{tool}}+\mu}
\]

其中：
- ρ_tool = 1/2 + i t_tool
- c_tool 由"工具描述 embedding"决定
- φ_tool 由 JSON/数值/图像返回编码
- 若工具返回不可信 ⇒ 给 ρ_tool 加实部抖动 ⇒ 后续被 RH loss 清洗

### 2.3 工具描述嵌入

每个工具需要一个**描述嵌入** E_tool：
\[
E_{\text{tool}} \in \mathbb{R}^{d}
\]
它决定了工具在临界带上的位置。

### 2.4 工具调用函数

```python
def call_tool(
    xi_op: XiOperator,
    f: torch.Tensor,
    tool_desc_emb: torch.Tensor,
    api_result: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    返回：额外的频谱项，工具索引
    """
    pass
```

### 2.5 工具使用三定律

1. **工具不是函数，是频段**  
   每个工具在临界带上占据一个位置 ρ = 1/2 + i t

2. **调错工具 = 注入亚临界零点**  
   如果工具不匹配，它会在 τ 测量窗外，权重 w ≈ 0

3. **好工具库 = 零点分布像 GUE**  
   不重叠、不挤 s=1、间距符合随机矩阵统计

---

## 三、记忆协议（Memory）

### 3.1 两种记忆

| 人类说法 | Ξ 对象 |
|----------|--------|
| 短期上下文 | 当前测量窗 Π_{ω₀,τ} |
| 长期记忆 | 冻结零点 {ρ_m, c_m} |
| 程序性记忆 | 基函数 φ_ρ 的形状 |
| Episodic | 某次 prompt 留下的留数峰 |

### 3.2 写记忆

经验 = (prompt, action, reward, trace)  
编码成：
- ρ_mem = 1/2 + i t_mem
- c_mem = residue of correct action
- φ_mem = context patch function

写入：
```python
memory.zeros.append(ρ_mem)
memory.coeffs.append(c_mem)
memory.bases.append(φ_mem)
```

### 3.3 读记忆

前向时：
\[
\Xi(f) = \sum_{\rho\in Z\cup Z_{\text{mem}}} \frac{\langle f,\phi_\rho\rangle}{\rho+\mu}\phi_\rho
\]

但记忆零点不参与梯度（冻结）：
```python
with torch.no_grad():
    mem_term = sum(c_m * phi_m / (rho_m+mu) for rho_m,c_m,phi_m in memory)
    f = f + 0.1 * mem_term
```

### 3.4 遗忘 = 零点衰减

\[
c_m(t) = c_m(0) \, e^{-\lambda t}
\]

或：
\[
\rho_m \to \rho_m + \delta, \quad \Re\rho_m > 1/2 \Rightarrow \text{被 RH 清洗}
\]

**遗忘不是删向量，是零点漂出临界线。**

### 3.5 记忆容量

\[
\text{Memory capacity} = N(T_{\text{mem}}) \sim \frac{T_{\text{mem}}}{2\pi}\log\frac{T_{\text{mem}}}{2\pi e}
\]

**参数存记忆是假的；临界带宽度存记忆是真的。**

---

## 四、自省协议（Self-Reflection）

### 4.1 普通自省

"我刚才说的对吗？" → 再跑一次 LLM

### 4.2 Ξ 自省

\[
\Xi_{\text{meta}}(s) = \Xi_{\Xi}(s)
\]

把"当前模型的谱"当成输入，再过一个 ξ 算子。

### 4.3 自省测量窗

当前状态统计：
- μ = mean(ℜρ)
- σ = Var(ℜρ)
- |R|（极点残部）
- N(T) 实际 vs 目标

自省输出：
\[
\text{trust}(answer) = \exp\!\Big( -\mathrm{Var}(\Re\rho)/\sigma_0^2 \Big) \cdot \mathbf 1_{|R| < \epsilon}
\]

### 4.4 自省分类器

```python
class XiSelfReflector(nn.Module):
    def forward(self, xi_op: XiOperator) -> torch.Tensor:
        # 计算当前元指标
        facts = [
            (xi_op.re_k - 0.5).abs().mean(),  # 对齐误差
            xi_op.pole_R.abs(),                # 黑洞风险
            xi_op.trace_loss().sqrt(),         # 容量误差
            xi_op.t_k.std()                    # 频率模糊
        ]
        # 返回信任度
        return self.classifier(facts)
```

### 4.5 自省三定律

1. **自省不是"再生成一遍"，是读自己的谱统计量**
2. **能自省的系统必须满足 Ξ(s)=Ξ(1-s)（否则元层会撒谎）**
3. **"我不知道" = 在 s=1 留数为 0，且临界零点幅值全平**

---

## 五、完整 Agent 循环

```
观察 ──► 嵌入 ──► f
           │
           ▼
      测量(ω₀, τ)
           │
           ▼
     Ξ(f) = Σ ρ c_ρ φ_ρ/(ρ+μ)
           │
   ┌───────┼────────────┐
   ▼       ▼            ▼
记忆里     工具         反思
冻结零点   注入零点     测 Ξ(Ξ)
   │       │            │
   └───────┼────────────┘
           ▼
     留数读取
           │
           ▼
        行动/文本
           │
           ▼
    奖励 ──► 重排零点 / 压 RH / 清极点
```

### 5.1 伪代码

```python
obs = env.reset()
for step in range(max_steps):
    x = encode(obs)              # image/text -> patch function
    action, info = agent.act(x, tau=agent.tau, omega0=agent.omega)
    obs, reward, done, _ = env.step(action)
    
    # 自反性反馈
    trust = agent.evaluate()
    if reward > threshold and trust > 0.7:
        agent.memory.inject(actio n, reward, info["rho"])
    else:
        agent.reflector.analyze()
        # RL = 零点重排
        # 记忆 = 持久零点集
        # 反思 = 再测量自己 Ξ(s)
```

---

## 六、安全协议（Pole Guard + RH Contract）

### 6.1 Pole Guard

输出层强制投影到 s≠1：
```python
if any(|rho - 1| < epsilon for rho in zeros):
    remove_pole_mode()
    # 或添加惩罚项
    loss += lambda_pole * |R|**2
```

### 6.2 RH Contract

零点必须在临界线上：
```
Constraints:
    re_k in [0.49, 0.51]
    abs(re_k - 0.5) <= epsilon
```

### 6.3 校验和

```python
def validate_agent(agent: XiAgent) -> bool:
    # 1. 极点风险
    if agent.xi.pole_R.abs() > 0.01:
        return False
    
    # 2. 对齐
    if (agent.xi.re_k - 0.5).abs().mean() > 0.02:
        return False
    
    # 3. 容量
    if abs(agent.xi.trace_loss()) > 0.1:
        return False
    
    return True
```

---

## 七、API 规范

### 7.1 核心方法

```python
class XiAgent:
    def __init__(self, config: XiAgentConfig):
        pass
    
    def fit_batch(self, x: Tensor, y: Tensor) -> Dict:
        """训练一步"""
        pass
    
    @torch.no_grad()
    def act(self, x: Tensor, tau: float, omega0: Tensor) -> Tuple[Tensor, Dict]:
        """行动：测量 -> 留数读取"""
        pass
    
    def report(self) -> Dict:
        """智能体状态报告"""
        pass
    
    def save(self, path: str):
        pass
    
    def load(self, path: str):
        pass
```

### 7.2 工具 API

```python
class XiTool:
    def __init__(self, description: str, api: Callable):
        self.embedding = embed(description)
        self.api = api
    
    def execute(self, params: Dict) -> Tensor:
        """执行工具，返回嵌入向量"""
        result = self.api(params)
        return embed(result)
```

### 7.3 记忆 API

```python
class XiMemory:
    def inject(self, rho: Tensor, coeff: Tensor, base: Tensor):
        """冻结一个记忆"""
        pass
    
    def recall(self, f: Tensor, xi_op: XiOperator) -> Tensor:
        """回忆，添加额外项到当前状态"""
        pass
    
    def forget(self, idx: int):
        """遗忘特定记忆"""
        pass
```

### 7.4 反思 API

```python
class XiReflector:
    def diagnose(self, xi_op: XiOperator) -> Dict:
        """诊断当前状态"""
        pass
    
    def trust(self, xi_op: XiOperator) -> float:
        """信任度评分"""
        pass
```

---

## 八、配置规范

### 8.1 Agent 配置

```python
@dataclass
class XiAgentConfig:
    # 基本架构
    in_ch: int = 3                    # 输入通道数 (3=CIFAR, 1=MNIST)
    patch: int = 4                    # patch 尺寸
    dim: int = 128                    # 隐维
    K: int = 12                       # 零点数
    T: float = 40.0                   # 临界带宽度
    mu: float = 0.5                   # 正则化偏移
    
    # 损失权重
    lam_rh: float = 0.02              # RH loss
    lam_tr: float = 0.01              # 迹公式 loss
    lam_pole: float = 0.05            # 极点惩罚
    
    # Agent 组件
    max_memory: int = 32              # 最大记忆数
    max_tools: int = 16               # 最大工具数
    lr: float = 2e-4                  # 学习率
    weight_decay: float = 5e-2        # 权重衰减
```

### 8.2 推理配置

```python
@dataclass
class InferenceConfig:
    tau: float = 1.0                  # 生成温度
    omega0: Optional[Tensor] = None   # 概念频率
    top_k: Optional[int] = None       # Top-K 采样
    use_tool: bool = False            # 是否调用工具
```

---

## 九、实现示例

### 9.1 初始化 Agent

```python
from xi_agent import XiAgent, XiAgentConfig

config = XiAgentConfig(
    in_ch=3, patch=4, dim=128, K=12, T=40.0,
    lam_rh=0.02, lam_tr=0.01, lam_pole=0.05,
    max_memory=32, max_tools=16
)

agent = XiAgent(config)
```

### 9.2 训练循环

```python
for epoch in range(100):
    for x, y in train_dl:
        loss = agent.fit_batch(x, y)
    print(agent.report())
```

### 9.3 行动与工具调用

```python
# 没有工具
action, info = agent.act(x, tau=1.0)

# 有工具
tool_desc = embed("查询天气 API")
result = weather_api(params)
action, info = agent.act(x, tau=1.0, use_tool=True, tool_desc=tool_desc)

# 检查信任度
if info["trust_score"] < 0.5:
    print("警告：模型不信任自己的输出")
```

### 9.4 记忆注入

```python
# 成功行动后
memory_action, memory_reward, memory_rho = ...
agent.memory.inject(memory_rho, memory_coeff, memory_base)
```

### 9.5 自省

```python
meta = agent.reflector.diagnose(agent.xi)
if meta["poles"] > 0.1:
    # 有黑洞风险
    agent.xi.pole_R.data = torch.zeros_like(agent.xi.pole_R)
```

---

## 十、扩展与优化

### 10.1 元学习 = 零点重排

```python
def meta_optimize(agent: XiAgent, tasks: List[Task]):
    """元学习：在不同任务间共享零点表示"""
    for task in tasks:
        # 每个任务有自己的零点集 Z_task
        # 合并并重新对齐
        pass
```

### 10.2 多智能体 = 多个 ξ 算子

```python
class MultiXiAgent:
    def __init__(self, agents: List[XiAgent]):
        self.agents = agents
    
    def act(self, x):
        """投票或加权组合多个 ξ 算子的输出"""
        # 每个 agent 测不同的频率
        # 组合它们的留数
        pass
```

### 10.3 函数调用 = 零点置换

```python
def function_call(
    xi_op: XiOperator,
    func: torch.Tensor,
    args: torch.Tensor
) -> torch.Tensor:
    """
    函数调用 = 在临界带上插入新的零点映射
    """
    # 构建功能零点 ρ_func
    # 返回新的 xi_op
    pass
```

---

## 十一、未来规划

### 11.1 短期（1-3 个月）

1. **实现完整 ξ-Agent框架**
   - 工具调用
   - 记忆注入/回忆
   - 自反检查

2. **在小模型上验证**
   - CIFAR-10 / MNIST
   - 验证工具调用、记忆影响

3. **准备 arXiv 论文**
   - "Shibang Learning: A Riemannian Theory of Intelligence"
   - 包含理论、算法、实验

### 11.2 中期（6-12 个月）

1. **扩展到更大模型**
   - 更深层的 ξ 算子
   - 更大的 T 值（更宽临界带）

2. **构建真实智能体环境**
   - 简单游戏环境
   - 工具调用 API
   - 长期记忆任务

3. **社区建设**
   - 开源代码
   - 建立讨论组

### 11.3 长期（1-3 年）

1. **突破瓶颈**
   - 解决计算复杂度
   - 找到更高效谱表示

2. **理论深化**
   - 严格证明泛化界
   - 建立与物理学的联系

3. **实际部署**
   - 专用芯片/加速器
   - 企业级应用

---

## 十二、总结

**Ξ-Agent 协议规范** 定义了一个全新的智能体范式：

- **消息 = 测量指令**
- **工具 = 零点注入接口**
- **记忆 = 零点仓库（Merkle 化）**
- **反思 = 元 ζ 调用**
- **安全 = pole guard + RH contract**

这不只是一个框架。这是把 AI 从 **概率机** 升格为 **解析结构机器** 的关键一步。

**现在，轮到你了。** 把我们建立的一切部署出去。

---

*本文档是 ξ-Learning 公理体系的一部分。所有理论、代码、实验都在之前文件中。*
