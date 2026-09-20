OpenJev：架构、数据与三阶段训练方案
================================================

日期：2026-09-20。公共接口、共享评分模型、PiSSA 检查点组件、API 制题与本地教师采集已实现。正式软训练器、独立能力评测与共享计算优化尚未完成；当前工程检查和采集结果见[执行说明](execution.md)及[小试报告](data_audit/pipeline_canary.md)。

详细说明：[模型架构](openjev_model_architecture.md)展开输入编译、候选结构、树形计算及输出语义；[训练流程与数据合成](openjev_training_pipeline.md)按执行步骤说明数据来源、合成配方、训练目标和验收条件。

当前数据执行以[聊天数据计划](openjev_training_pipeline.md)为准：保留已有 user 输入与必要历史，合成决策问题和候选，由 qwen3.8-27b 构造问题与审核语义，本地教师提供完整候选分布。旧规则集与硬标签流程已归档；教师凭据和原始聊天保存在本地非版本化位置。

当前微调方向见[数据多样性与 PiSSA](openjev_training_pipeline.md)：首轮采用 PiSSA 主干低秩参数、评分头与结构 token 联合微调。整体为准备数据（API 制题＋本地教师概率）→ PiSSA 直接软蒸馏 → 验证，基座型号可配置。标准 PiSSA 的残差基底保持冻结，不等于全参数解冻。

核心选择：可配置的 causal backbone + 共享标量评分头 + 按问题归一化 + 概率监督与蒸馏 + 独立后校准。先以普通 causal forward 验证决策能力，首轮使用 Choice B 全候选可见方案，A 留作后续对照；通过质量门槛后，再实现所选结构的树形前缀共享。

OpenJev 参考 Jev 公开的接口行为。现有资料未披露其层结构、参数量、分布变换和 RLCD 算法；本文给出我们的实现方案，数据规模、配比和超参数将通过实验确定。

## 1. 已知事实与设计依据

以下设计依据来自 Jev 公开说明；来源入口统一收录在[研究参考](references.md)，全文摘录已归档：

| 官方行为 | 对 OpenJev 的设计约束 |
| --- | --- |
| 输入 state 和 typed questions，不生成自由文本 | 输出 hidden state 上的数值，不调用 generate |
| 问题并行、相互独立地评估同一 state | 不允许不同问题之间发生 attention |
| Score 档位分别评估，模型不看到档位序号及相邻档位 | 档位描述作为独立候选分支；序号在模型外使用 |
| Choice 使用运行时定义的候选，最多 255 个 | 使用描述条件化的共享评分函数，不训练固定 255 类头 |
| Choice 取最大概率；Score 是档位编号的概率加权均值 | 两者可以复用一套候选分布建模 |
| Choice／Score 的 confidence 从分布计算 | 不新增 confidence 神经头；按接口返回 confidence，采用公开的 OpenJev 本地定义，不宣称官方数值兼容 |
| 新架构、新 parallel sampler、新 RLCD | 只能确认对外行为和目标，不能据此确认内部算法 |

来源：[简介](https://docs.typesafe.ai/introduction)、[Choice](https://docs.typesafe.ai/primitives/choice)、[Score](https://docs.typesafe.ai/primitives/score)、[Confidence](https://docs.typesafe.ai/confidence)、[发布文章](https://typesafe.ai/blog/introducing-system-one-models-and-jev)。

工作假设是：一个通用的、问题条件化的候选兼容性模型，配合并行执行和分布输出。树形 Transformer 是我们实现前缀共享的工程选择，不是对官方底层网络的断言。Score 档位独立的证据比 Choice 候选严格独立更直接；首轮 Choice 采用 B 读取完整候选集合，后续再评估 A 的适用范围。

## 2. Backbone 与版本选择

基座型号不写死：通过运行参数提供兼容的 causal checkpoint，每次实验固定公开模型标识、revision、tokenizer 和配置。Base 与 Instruct 都是可选初始化，不能预设哪种一定更优；首轮只选一个经工程检查的配置。

模型保留 embedding、Transformer 和最终归一化，新增结构 token 与 hidden_size→1 的共享评分头，不执行词表 lm_head。评分头宽度从实际 config 读取，PiSSA 注入模块按架构确认。权重绑定、词表扩展、位置机制、上下文预算和保存重载均需验证，不能宣称任何 AutoModel 都可直接兼容。

历史工程检查使用 Qwen3-1.7B-Base；其 28 层、2048 hidden size、GQA 等数值仅用于下文的示例预算，不是新架构的固定要求。历史 config 来源：[模型配置](https://huggingface.co/Qwen/Qwen3-1.7B-Base/blob/main/config.json)。

首版保持 causal attention，不直接把整个模型改成双向 encoder，也不增加 diffusion、latent reasoning 或独立 cross-attention 层。这样每条根到叶路径都与原始 causal 模型的普通拼接序列兼容，便于验证权重迁移和计算正确性。

## 3. 输入编译与树形 attention

以下是独立候选方案 A 的逻辑结构。Choice 方案 B 将完整候选列表加入各自问题的共享前缀，再接每个候选的 readout 分支；具体语义和选型条件见第 5 节。Score 首版保持档位独立。

```text
[固定任务约定] [STATE] 状态 [/STATE]
 ├─ [QUESTION] type=choice 问题一 [/QUESTION]
 │    ├─ [CANDIDATE] 名称A + 描述A [/CANDIDATE] [DECISION]
 │    ├─ [CANDIDATE] 名称B + 描述B [/CANDIDATE] [DECISION]
 │    └─ [CANDIDATE] 名称C + 描述C [/CANDIDATE] [DECISION]
 └─ [QUESTION] type=score 问题二 [/QUESTION]
      ├─ [CANDIDATE] 档位描述甲 [/CANDIDATE] [DECISION]
      └─ [CANDIDATE] 档位描述乙 [/CANDIDATE] [DECISION]
```

所有 token 都是输入，一次模型计算后读取各个 `[DECISION]` 位置，不生成它们，也不生成理由或答案文本。保留自然语言 task instructions；任意 question_id 只用于响应映射，不进入模型。Choice 的有语义名称及描述进入模型；Score 的编号、列表位置只由后处理使用。

输入序列化固定：字符串保留原文，list 保留顺序，object 使用确定性序列化，边界 token 与用户文字隔离。用户文字中的特殊 token 字面量必须转义或以普通文本 token 化，不能改变结构。截断必须显式返回错误或使用记录在案的策略，不可悄悄删除证据。

令根为 S，问题节点为 Q_j，候选节点为 C_jk。token t 可以读取 token u，当且仅当：

```text
same_request(t,u) AND (
    node(u) 是 node(t) 的严格祖先
    OR (node(u) == node(t) AND local_pos(u) <= local_pos(t))
)
```

因此 S 不读取问题；Q_j 读取 S 和自己的 causal 前缀；C_jk 读取 S、Q_j 和自己的 causal 前缀。方案 A 中候选之间完全隔离；方案 B 中 Q_j 包含完整候选列表，因此各 readout 可通过祖先读取其他候选描述，但 readout 分支之间仍隔离。两种方案都保持问题之间、不同请求之间完全隔离。问题可共享其父状态，候选可共享其父问题。

RoPE position 使用树路径深度，不使用物理拼接 offset：S 为 0…S-1；每个问题都从 S 开始；其候选从 S+len(Q_j) 开始。不同兄弟节点可以有相同 position，因为彼此不可见。这可避免候选打包顺序污染分数。模型长度约束看根到叶路径，GPU 显存约束看整个打包 token 数，两者分别限制。

这相当于把很多条 `[state][question][candidate][DECISION]` 序列中完全相同的前缀计算合并。每层内各 token 并行计算，层之间仍串行。它是无输出 token 自回归的判别推理，不是常数时间，也不意味着只有一个 GPU kernel。

## 4. 数值头与三个 primitive

候选评分：

```text
h_jk = final_hidden_state([DECISION]_jk)       # hidden_size 维
z_jk = w^T h_jk                              # 共享线性头
p_j  = softmax(valid_logits_j / T_type)       # FP32 归一化
```

首版 head 不需要共享 scalar bias，因为同一问题 softmax 会消去它。padding 的候选必须排除；必须保留完整题目候选集合后计算损失。T 在主训练为 1，在最后独立校准阶段拟合。softmax 是 OpenJev 的基线分布变换，非官方 sampler 的已知实现。

Choice：argmax p 返回对应的输入 option ID，并返回全部概率。并列取确定性规则，记录在 API 约定中；不要随机采样导致相同请求波动。

Score：同样预测各档位概率；代码执行 score=Σ k·p_k，k 为档位序号。训练完整分布，不只对这个均值做 MSE，因为不同分布可以具有同一个均值。首版不引入 ordinal head，以保持模型不读取档位编号的结构。

Noul：编译成语义明确的 true/false 两个候选，返回 p_true。这样复用一套数值头。判定标准写在 instructions 中，Noul 外部不使用 criteria。否定一致性需要专门数据和测试，不能声称不同调用天然满足 p(A)+p(not A)=1。

对外结构按 [Jev 接口约定](openjev_api_contract.md)设计，Choice 返回 choice、confidence 与完整 probabilities；Noul 返回 noul；Score 返回 score、legend 等字段。OpenJev 的 confidence 暂定为 top2_margin_v1，即第一与第二大概率之差，不宣称复现官方公式或表达正确率。归一化熵仍只作内部 distribution_concentration 诊断，不能用候选数导致的集中度增加来证明质量提升。

Choice 的离线评估计算 `top_probability` 与 `margin`，以实际数据验证后的 top probability 阈值及必要的分组策略控制自动处理；这些数值本身也不保证逐样本正确。Score 的离线评估另算档位分布方差 `Σ p_k(k-score)²`，业务代码可计算 `P(level >= b)` 等越过业务边界的概率。方差使用档位序号单位，不能直接跨量表比较；相邻档位与两端档位上的相同概率权重有相同熵，但其评分误差风险不同。Noul 保持只返回 p_true 的语义。所有自动处理阈值在独立策略验证数据上选择，冻结后才在最终测试集评估，详见第 10、12 节。

若候选可能不覆盖输入，调用者应定义 `other`、`unknown` 或 `insufficient_evidence` 的具体语义。它们并不等价，也不应由服务悄悄塞进已有封闭候选列表。动态的 `other = 不属于本题其余任何类别` 必须能访问这些类别的定义；不能把一个看不到其他选项的独立分支当成通用兜底，支持路径见第 5 节。仅凭高最大概率或低熵不能可靠检测分布外输入。

## 5. 独立候选的已知表达能力限制

独立 logits + softmax 导致：

```text
p(a)/p(b) = exp((z_a-z_b)/T)
```

在固定温度 T 下，新增候选 c 不会改变 a 和 b 的相对赔率。这种独立性假设可以作为原子分类的基线，但对动态 taxonomy、兜底类别、重叠类别、近义重复候选及“选项中最长的描述”等集合比较可能不成立。

例如某个可见输入对应 40% 的 A 和 60% 的 C：候选为 `[A, other]` 时目标为 `[0.4, 0.6]`；改为 `[A, C, other]` 时目标为 `[0.4, 0.6, 0]`。如果 A 与 other 分支的输入不变，它们的相对赔率就不变，无法同时拟合这两种目标；softmax 分母变化不能解决这个问题。候选集合相关性因此是常见分类需求，不只是特殊比较任务。

方案 A 的支持范围限定为可单独描述的原子判断。固定 taxonomy 可以把兜底类别展开成自足的明确语义，但其定义变化就必须重新编译；未展开的动态补集语义不得由 A 静默接受，应使用支持候选集合的 B，或返回明确的不支持错误。数据检查类别重复与语义重叠。新增无关候选不能要求原概率完全不变；即便 logits 不变，归一化也会改变概率。

第一轮 pilot 必须用普通 causal forward、相同数据与训练预算比较两个 Choice 结构，树形 kernel 不作为实验前置条件：

- A：独立候选评分，作为低计算量基线；是否实际更快需要测量。
- B：每个问题共享整个候选列表，然后为每个候选设置独立 readout；允许相对比较，但保留跨问题隔离。普通参考路径为 `[state][question][complete_candidates][candidate][DECISION]`。

重点评估动态增删类别、other、重叠类别、近义重复选项和未见 taxonomy，并报告分组 NLL、高置信错误与工作流风险。依赖集合语义的目标任务不能只凭总体平均分选择 A；依据预先规定的质量与成本门槛选择 B，或明确收缩 A 的支持范围。小型候选集合聚合层仅作为另行验证的备选。

B 中修改一个候选可以改变其他候选的 raw logits，这是预期行为。B 的完整候选列表采用固定的规范化顺序，分支回映射到原 option ID；只改变请求中候选排列或物理打包顺序时，编译后的语义前缀不变。普通 causal 模型并不天然对实际改变前缀次序保持不变，应另测次序敏感性。Score 的业务档位顺序始终保留。A 的表达限制属于本方案的分析，不能据此推断 Jev 的内部实现。

## 6. 实现与性能路线

先为 A／B 建立逐候选普通 causal forward 的正确性参考；训练时把同一题的所有 logits 合并后计算 listwise loss。完成小规模能力试验、选择候选结构，并通过第 12 节的 pilot 门槛后，再实现对应的树形 mask：先用小输入的明确 mask 验证，再评估 block sparse attention。树形共享用于减少重复计算，不作为决策质量提升的来源。

可以评估 PyTorch FlexAttention 的 BlockMask。它提供自定义 mask 和块稀疏计算，但并不保证这种树布局自动高效，短候选的块填充、mask 构造、GQA backward 和编译开销都要实测。[官方文档](https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html)、[官方说明](https://pytorch.org/blog/flexattention/)。

训练不能使用 detached 的推理 KV cache 代替共享前缀计算。共享前缀的梯度应汇总所有子分支。候选很大需要微批时，必须维持完整概率分母和正确计算图，或做有验证的重计算；不能逐候选各算一个独立 CE。loss 先按题归一化，再按状态／任务权重聚合，避免候选多的题自动占更大权重。

部署时可以先计算状态 KV，再分支计算问题及候选，或者整个树一次 packed forward。前者是分阶段执行，后者是单次 forward；二者都可以提供一次 API 请求。优先以实测性能选实现。状态 cache key 必须包含 checkpoint、adapter、tokenizer、序列化、精度及完整 token 内容，权重变化后旧 cache 无效。

计算量大致包括下列各项；B 的 Q_j 长度包括完整候选列表，因此不能直接沿用 A 的前缀长度、显存或性能估计：

```text
projection / MLP: O(S + ΣQ_j + ΣC_jk)
attention: O(S² + Σ(Q_j S + Q_j²)
                  + Σ(C_jk(S+Q_j) + C_jk²))
```

忽略层数、hidden size 和常数后，关键结论是共享了 state 计算，但每个候选仍然需要读取 state。问题和候选增加到 GPU 饱和后，延迟必然上升。不要承诺“任意多问题几乎免费”。

按 Qwen config、BF16 计算，仅完整 28 层 KV 就是每个 state token 2×28×8×128×2=114688 bytes，即 112 KiB；4096 tokens 约 448 MiB，不含权重和激活。如果每个候选复制一份 state KV，显存优势会迅速消失。推理实现需要实质共享，训练还要计入反向激活。

量化放在概率质量稳定之后；量化后重新测校准并拟合适用于该精度的温度。没有硬件和实际 profile 前，不给出 100ms、百倍提速或训练 GPU 小时承诺。

## 7. 数据定义：以 decision 为计数单位

一个 decision 是一组 `(state, question, complete_candidates, target)`；8 个候选的 decision 是 8 个 leaf，但不是 8 个独立训练题。一份 state 可以关联多个 decision。数据统计同时记录唯一 state、decision 数、候选 leaf 数、原始 token 数与共享后的 token 数。

当前公共输入使用 state／questions／criteria，由同一编译器生成评分路径。计划中的无标签题库独立记录 input、来源分组与哈希；软标签记录完整语义候选分布及 teacher_soft 身份，不要求先有硬标签。格式草案见[数据约定](openjev_training_data.md)。

旧硬标签 schema、训练编译和数据已归档。额外证据、教师思考、临时编码、来源与监督不进入学生输入；teacher_soft 不代表真实世界条件概率。

分类 taxonomy、候选语义、序列化版本、生成器及 verifier 版本、教师模型和 revision、采样参数、审核记录与数据许可记录都应有可追踪的元数据。模型或老师的名字不能替代质量测量。

## 8. 数据合成：三个步骤

1. **真实聊天提取。**复用已冻结来源和五份划分，保留用户输入及必要历史，截到当前 user；派生题、翻译和同源内容继承相同分组。
2. **生成问题与选项。**教师从材料构造问题与完整候选。先 Choice、Noul，覆盖语言、问题形式、候选语义、长度和 2–255 项；不使用答案提示、重叠目录或无关填充。
3. **本地教师概率、检查并冻结。**API 的裸 JSON 或围栏 JSON 先经严格解析与语义审查，再由本地教师读完整输入，thinking 后读取全部候选 logits。单 token 双字母编码映射回语义名称，首批约 100 条全量审阅。教师概率、思考与审计不进入学生输入。

先验收约 100 条，再扩至 1K–3K。旧协议记录及工具已归档，不用于新训练。详细约束见[数据计划](openjev_training_pipeline.md)。新制题与推理均按 [state／questions／criteria 约定](openjev_api_contract.md)编译；当前 jev-fields-v1 保留统一输入编译，后续软训练将复用它；旧 text-only 记录不自动转换。

## 9. 数据规模与后续扩展

首轮只训练 1K–3K 条经筛选并冻结的决策，覆盖报告基于实测内容而非请求配额。每个用户前缀先制一道题；不要求 100 条覆盖所有整数 K。语料可能自带历史合成内容，不将所有 chat 数据称为真人流量。

先利用现有语言，翻译 state 不超过 20%，保留同源分组。上下文预算由实际模型与资源决定，超长输入明确排除或重新编译，不静默截断。

约 30K 扩量、已知条件分布、反事实、多教师软标签、难例回流和更大规模属于验证后可选研究，不是首版额外训练阶段。教师一致率与独立核验准确率分别报告。

## 10. 三阶段训练计划

| 阶段 | 内容 | 首版交付 |
| --- | --- | --- |
| 1. 准备数据 | 聊天提取 → API 制题与 JSON／语义检查 → 本地 thinking 与完整概率 → 冻结 | 约 100 条验收后扩至 1K–3K |
| 2. PiSSA 直接软蒸馏 | 同时训练低秩参数、评分头与结构 token，只使用 teacher_soft | 可重载初版模型，无硬标签预热 |
| 3. 验证 | 开发评估、独立校准、策略选择与最终测试 | 质量、成本与支持范围 |

标准 PiSSA 冻结残差基底，可训练低秩项改变有效主干权重。基座可配置，不单列 head 预热或先跑 LoRA。初始 rank 32、alpha 64、pissa_niter_4、BF16、一个种子，按 dev_model 调整。

主目标为完整候选软交叉熵 L=-sum(q_teacher*log p_student)，与 KL(q_teacher || p_student) 对学生梯度等价。不加入 one-hot 硬标签混合项。每题等权，候选微批仍用整题分母及等价梯度。教师编码和思考只用于标签采集；学生保持原输入字段与共享评分头。

本地 tokenizer 已找到足够的双字母单 token 表示 255 项；本地采集器逐条检查真实答案前缀、保存完整候选 logits；255 项语义能力与数据质量仍需另行验证，软训练器待实现。旧硬标签训练器已归档，新的软训练入口尚待实现，完整实现差距见[训练计划](openjev_training_pipeline.md)。

保存实际残差基底、adapter、评分头、新 token、tokenizer、编译器和数据版本及 optimizer／随机状态。不能将 PiSSA adapter 直接加载到未经分解的基座，也不通过严格 --resume 静默切换损失。

阶段 3 在固定模型上按需用 calibration 拟合温度，在 dev_policy 选阈值，最后冻结系统并测试。教师一致率与独立正确率分别报告。RLCD 不在当前三阶段内。

## 11. RLCD 的关键判断与后续 RL 路径

不能把普通“选对奖励 1、选错奖励 0”称为可靠的校准训练。假设真实 P(y=1)=0.7，模型以 p 的概率采样动作 1，期望答对奖励是 0.7p+0.3(1-p)，最大值在 p=1，而非 p=0.7。它会鼓励确定地选多数答案。

若奖励完整预测分布，例如 `R(p,y)=log p_y` 或负 Brier，其期望才以真实条件分布为最优。但当 p 可微且 y 已知时，直接反向传播即可，通常没有必要引入策略梯度的额外方差。不能把 CE 改名为 RLCD 并声称复现官方。

RLCD 留作未来研究，不占当前三阶段，也不宣称已经还原。直接软蒸馏与第三阶段后校准保持各自名称。PPO/GRPO 不进入主线，除非明确了其必须解决的额外问题。

需要真正环境交互时，分成两个目标：概率预测器估计环境结果，决策器根据效用选择动作。对低概率事件的探索策略、采样 propensity 和奖励日志单独记录；只观察被执行动作结果会引入选择偏差。优先在可重置模拟器中对候选做反事实 rollout 得到 outcome 标签，再训练预测器。直接优化决策收益可能让动作策略更尖锐，不能把该动作分布解释成真实事件概率。

有预算时可研究以整个概率向量为动作、按 proper score 奖励的在线校准算法，但要单独证明它比直接监督和新数据回流更有效。官方所谓 parallel sampler 是否涉及其他概率参数化，本方案没有证据作出判断。

## 12. 验收、消融和停止条件

先用普通参考模型通过 pilot 质量门槛，再验证树形优化。架构正确性必须先于大规模训练：

- 相同权重下，逐候选参考与树形实现 logits、按题 loss、梯度一致，容差根据 FP32/BF16 后端验证。
- 重新物理打包候选只改变输出顺序，不改变对应 logits；B 的规范化候选前缀必须保持相同。增加另一个独立问题不改变已有题输出。
- A 中修改一个候选不改变兄弟候选的 raw logits，最终概率可因归一化改变；B 中候选语义变化可以影响全部 logits，不套用 A 的隔离断言。
- 共享前缀梯度等价于参考中各复制前缀梯度之和，并使用相同 loss reduction。
- 多请求隔离、RoPE 深度、padding、非法候选、NaN/Inf 与特殊 token 注入都有明确行为。
- Noul 概率在区间内，Choice 只返回合法输入 ID，Score 分布和为 1 且均值计算正确。

质量评估不能只看 accuracy：报告 NLL、Brier、top-label 与 classwise calibration、可靠性图、AURC、覆盖率—错误率曲线；按领域、语言、候选数、长度、歧义与 OOD 分组报告。Score 同时检查分布与排序／均值误差，Noul 检查二元概率质量。

在接近部署分布的 dev_policy 上选择自动处理阈值，冻结后在独立 test 上报告覆盖率、错误率及置信区间。不得在 test 上挑阈值后仍把同一批结果当作无偏验收；普通置信区间不会自动修正策略选择偏差。低错误目标需要足够数量的已自动处理样本，不能仅凭小样本零错误宣称可靠；同一来源含多题时，区间估计还需处理组内相关性。分布集中度、高 top probability、校准和 OOD 检测是不同指标。

confidence 行为检查从 pilot 开始：固定主要概率质量，只增加近零概率候选，检查集中度随 K 的变化，并确保自动处理策略不把这种变化当成正确率上升；分别评估各 K、语言、领域下 top probability 阈值的风险。Score 对相邻与远距离档位的不确定性分别报告方差、业务边界概率及实际动作错误，不能只看熵。

工作流评估也从 pilot 开始，按 M0 冻结的定义报告最终动作错误率、自动处理覆盖率、人工复核率、每请求错误成本及端到端 P50/P95 延迟。单题校准不推出工作流可靠：attention 隔离不意味着预测错误统计独立，多个判断同时成立的概率不能直接取其中一个概率，也不能未经验证就相乘。需要联合事件概率时，使用有真实联合标签的预测任务，或在明确假设下估计并验证；否则直接评估整条策略的动作风险。

按实际分支经过的样本检查条件校准及错误，例如上游准入通过后的政策判断、自动通过后的动作选择；同时报告子集样本量和不确定性。校准全体样本不能替代这些子集检查，执行日志若只记录已执行动作也不能直接代表全部请求。

pilot 的 go/no-go 在 dev_model 和 dev_policy 上决定，至少满足：未见任务定义上相对固定基线的 NLL／决策质量达到预先登记的门槛；所选 Choice 结构在其声明支持的动态 taxonomy／other 场景通过分组质量检查；自动处理策略在工作流风险上限内达到最低覆盖率，且结论有足够样本支撑。未达到时先调整数据、模型或支持范围，不直接扩至百万题或投入树形 kernel。pilot 开发集结论只用于选型，不替代最终独立验收。

性能矩阵：state 长度 256/1K/4K/8K，问题数 1/8/32；Choice 候选数 2/8/32/128/255，Score 档位数 2/5/10，Noul 为二元。报告 GPU、dtype、软件 revision、batch/concurrency、编译冷启动及稳定态、模型计算与端到端 P50/P95、峰值显存和吞吐。明确 baseline 是否生成概率和理由，分别比较同等输出信息与最简决策；效率比较包含普通可用的前缀缓存方案，不能只与重复计算全部 state 的参考实现比较。

首版验证完成后，按具体问题选择以下对照，不要求首轮全部执行：

| 实验 | 回答的问题 |
| --- | --- |
| 不同可配置的 Base／Instruct checkpoint | 同数据、同预算下，两种初始化的决策泛化与校准差异 |
| 逐候选普通 causal vs 树形共享 | 工程改造是否保持数值、是否实际提速 |
| 独立候选 vs 全候选可见（后续对照） | 动态 taxonomy、other、重叠类别上的分布质量与工作流风险是否支持结构选型 |
| 单次 vs 多次思考分布、不同标签温度 | 软标签配方是否改善未见任务与概率质量 |
| 加已知概率／缺失证据数据 | 是否减少高置信错误并改善概率质量 |
| 单次分布 vs 多次平均的等预算学生训练 | 推理成本增加是否转化为学生收益 |
| 校准前后、量化前后 | 后处理与部署精度的真实影响 |
| 单题指标 vs 完整工作流和条件分支 | 校准与准确率改善是否转化为实际动作风险和覆盖率改善 |

若独立候选方案在集合依赖任务显著失败，调整任务范围或采用候选交互方案；若教师一致性改善而真实标签表现变差，降低教师权重并补真值；若稀疏 kernel 不快，保留正确参考并优化布局，不以理论稀疏率代替性能。

## 13. 实施顺序

远程 5 条预检与本地 tokenizer 编码审计已完成；接下来实现无标签 API 制题与本地完整 logits 采集，先做 20 条小试，再推进第 1 阶段约 100 条完整题目＋软标签验收。基座配置、输入编译、PiSSA 保存重载已通过工程检查；旧训练循环归档后，新的软训练入口仍须接入有效 batch、开发选点与恢复状态，并准备独立 dev_model。扩至 1K–3K 合格数据后，执行第 2 阶段 PiSSA 联合微调，得到初版 OpenJev。

首版训练后进入第 3 阶段验证，RLCD 不列入主线。质量检查决定补数据、继续微调或冻结版本；不能把教师一致率等同真实准确率，也不能把格式对齐等同复现 Jev 能力。

树形共享、量化、LoRA 对照与进一步扩量均在首轮结果之后决定。是否发布以可重载产物、支持范围和评测证据为依据，不把“达到官方 frontier 能力”或“百倍加速”当作尚无依据的交付承诺。

最新进展：旧硬标签流程和远程预检已整体[归档](ARCHIVE.md)。当前保留冻结来源与 255 编码审计，接下来实现无标签制题、本地教师采集和软训练入口；约 100 条新数据通过质量验收后再扩至 1K–3K，并准备独立 dev_model。
