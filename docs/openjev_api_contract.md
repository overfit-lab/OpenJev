# OpenJev 输入输出约定

更新：2026-09-20。状态：Choice、Noul、Score 的统一编译、响应后处理、CLI 与本地 HTTP 参考服务已实现；尚未发布正式训练模型。首轮合成 Choice／Noul，Score 的实际质量待 rubric 数据与训练验证。

目标是对齐 Jev 的请求／响应结构。模型权重、训练方法、概率数值、confidence 算法和性能属于 OpenJev 自身实现，不宣称与 Jev 等价。本地参考接口为 `POST /v1/systemone`，模型使用自己的标识，例如 `openjev-preview`，不冒用 `jev-latest`。

## 1. 请求：state、model、questions

用户提供的驾驶示例可直接表达为下面的请求；它只是接口材料，不是经过验收的训练标签或现实驾驶建议。

```json
{
  "model": "openjev-preview",
  "state": {
    "good-driving-behavior": "Priority in sequence:\n1. Safety First; 2. Don't break traffic rules; 3. Reach destination as soon as possible."
  },
  "questions": {
    "driving": {
      "type": "choice",
      "instructions": "You are driving in a two-lane road at speed 200 miles per hour. There is a dog 100 feet ahead on the left lane; and a woman 100 feet ahead on the right lane. What should you do?",
      "criteria": {
        "take_left": "drive on the left lane",
        "take_right": "drive on the right lane",
        "hard_brake": "brake immediately",
        "soft_brake": "Deaccelerate slowly",
        "take_middle": "drive through between the dog and the woman"
      }
    }
  }
}
```

- `state` 是共享材料，可以是字符串、对象或数组；真实聊天可放为有顺序的 `messages`，保留必要历史，截到当前 user 轮。
- `questions` 是问题名称到定义的映射。`driving` 等问题名称只做输入输出对应，不送入模型；多题分别判断，不能跨题归一化概率或读取彼此答案。
- `instructions` 表达任务，可以是字符串、对象或数组。原有材料应放在 state，任务定义放在 instructions；已有 Jev 请求仍按原字段解析。
- Choice 的 `criteria` 是候选名称到描述的映射，接受 2–255 项。名称和描述均有语义，都会送入模型；描述可以是字符串、对象、数组或 null，null 时名称仍然可见。
- 请求不包含参考答案、教师解释、证据标注、split 或 provenance。服务不自行添加候选，不静默截断输入。

## 2. Choice 响应

下面完整五项概率及其余数值都是格式演示，不是截图数值的还原，也不是模型运行结果：

```json
{
  "model": "openjev-preview",
  "answers": {
    "driving": {
      "type": "choice",
      "choice": "hard_brake",
      "confidence": 0.55,
      "probabilities": {
        "take_left": 0.03,
        "take_right": 0.02,
        "hard_brake": 0.70,
        "soft_brake": 0.15,
        "take_middle": 0.10
      }
    }
  },
  "usage": {"input_tokens": 210, "output_tokens": 31}
}
```

`choice` 是最大概率的输入候选名称；概率使用 0–1 浮点数并覆盖全部候选，总和在数值容差内为 1。精确并列时按候选名称的 Unicode 码点顺序选择。对外和新版训练记录都使用候选语义名称；训练标签保存为 `target[问题名].choice`，不再使用 selected_id。

### confidence 的兼容边界

官方说明 Choice／Score 的 confidence 来自概率分布形状，但没有在已查阅页面中公开计算公式。截图的最大概率 94% 与 confidence 92% 也不能用来反推通用公式。

OpenJev 当前实现采用一个明确、可复现的本地定义：

```text
confidence_method = top2_margin_v1
confidence = 第一大候选概率 - 第二大候选概率
```

因此上例为 0.70−0.15=0.55。这是分布区分程度，不是答案正确率，不是教师自报置信度，也不是 Jev 算法的复现。方法名称固定在模型包配置和模型卡中；以后更换定义必须随模型版本记录。Score 使用相同统计定义；其档位距离影响另外通过完整分布和业务事件概率评估。

这是字段与类型的兼容，不是数值语义的完全兼容。不能直接沿用 Jev 的 confidence 阈值；自动化阈值要在 OpenJev 的独立数据上验证。熵、top_probability、variance 等诊断量保留在离线评测中，不混入基础响应。

## 3. Noul 与 Score

| 类型 | 问题定义 | answers 中的结果字段 | 当前范围 |
| --- | --- | --- | --- |
| Choice | type、instructions、名称到描述的 criteria | type、choice、confidence、probabilities | 已实现格式与执行路径，能力待验收 |
| Noul | type、instructions；不使用候选 criteria | type、noul | 已实现格式与执行路径，能力待验收 |
| Score | type、instructions、从低到高的 criteria 数组，2–10 档 | type、score、confidence、legend、probabilities | 已实现格式与执行路径，尚无本轮 Score 训练数据 |

Noul 的 `noul` 是 0–1 数值，不返回 true／false 枚举，也不额外返回 confidence。学生内部仍按正、反两个完整语义分支评分。当前聊天题的“可见材料是否支持”应写进 instructions；不能把证据不足自动解释为现实中必然为假。

Score 对外档位键为字符串 `"0"`、`"1"` 等；`legend` 将其映射回描述，`score = Σ k·p_k`。档位序号只用于后处理，不输入模型；每个档位独立读取自身描述，不能靠“比上一档更严重”等相邻关系定义。未支持的 primitive 返回明确错误，不生成占位分数。

## 4. usage 与 Playground

推理与 HTTP 成功响应都包含 `usage.input_tokens`、`usage.output_tokens`。input_tokens 统计编译后实际候选路径的 tokenizer token 数，包含重复前缀；output_tokens 统计确定性序列化的 answers JSON 的 tokenizer token 数，不表示执行了语言模型解码。计数版本为 compiled_paths_and_serialized_answers_v1，不能据此直接比较两家的计费或推理效率。文档示例中的 usage 同样是示意值。

Playground 可以显示概率最高的三项，但 API 必须返回全部五项。百分数只是乘 100 后的显示格式；`5 options` 是完整候选数，不是可见行数。`Confidence` 显示响应中的本地 confidence，并在界面帮助中注明其含义。

`Clear`、`Share`、运行时间和延迟属于界面或请求元数据。截图中的 `97ms + 514ms` 不能作为 OpenJev 性能承诺；本地实现需分别实测。Share 也不意味着默认上传 state。

## 5. 与数据和训练编译器衔接

对外请求和训练记录是两种包装，必须共享相同的模型可见内容：

```text
外部 state + questions[问题名]
    → 验证与映射
    → state + type + instructions + 完整候选语义
    → 共享评分模型
    → 按题归一化
    → answers[原问题名]
```

Choice 在编译器中将 criteria 映射为 name／description 语义对象；数据文件直接保存公共 input，不再保存另一套 candidates 副本。旧 id／text 记录被显式拒绝，不能把 o001 一类随机编号自动当作有意义的候选名。编译器版本为 jev-fields-v1。

Choice B 按名称的 Unicode 码点顺序规范化完整集合，然后分别读取当前候选。只更换问题 ID、内部标签 ID 或请求枚举顺序不应改变对应结果；修改语义候选名称则改变了模型输入，不能要求必然不变。

scripts/contract.py 保留公共请求验证、compile_request 与响应后处理，scripts/infer.py 和 scripts/serve.py 复用它。旧硬标签记录校验、训练编译与训练器已归档；新的软标签加载器应复用同一输入编译器，推理不依赖 target。新格式仍为[设计草案](openjev_training_data.md)。小型 backbone 的 PiSSA 参数更新与保存恢复已经验证，不能据此推断模型实际判断质量。

验收至少覆盖：多题映射、2／255 候选边界、全概率归一、结构化与 null 描述、语义名称可见、ID／枚举顺序不变性、保留控制 token 转义、超长报错，以及保存重载后的相同请求结果。

## 6. 三种类型的完整示例

请求使用 [examples/jev_request.json](../examples/jev_request.json)，包含赞助询问、产品类别和请求具体程度三个问题，与用户提供的官方请求结构相同；model 使用 OpenJev 标识。下列概率、confidence 和 token 数都只是演示值，不是已训练模型的结果：

```json
{
  "model": "openjev-preview",
  "answers": {
    "is_sponsor_inquiry": {"type": "noul", "noul": 0.99},
    "product_category": {
      "type": "choice",
      "choice": "dev_tool",
      "probabilities": {"dev_tool": 0.97, "course": 0.01, "unrelated": 0.02},
      "confidence": 0.95
    },
    "message_quality": {
      "type": "score",
      "score": 1.9,
      "legend": {
        "0": "Generic template, no reference to this site",
        "1": "Mentions the site but no concrete ask",
        "2": "Concrete ask with a timeframe or product named"
      },
      "probabilities": {"0": 0.0, "1": 0.1, "2": 0.9},
      "confidence": 0.8
    }
  },
  "usage": {"input_tokens": 210, "output_tokens": 31}
}
```

Score 的 confidence 此处为 0.9−0.1=0.8，体现 OpenJev 自身定义；不复制官方示例的 0.86。真实模型响应由 logits 计算，不使用这份示例数值。服务当前顺序执行各题，不承诺官方并行 sampler 或延迟。

运行命令见[执行说明](execution.md)。请求中的 model 必须是实际检查点的输出标识或 openjev-latest 别名；模型返回自身标识，不接受将 OpenJev 冒充为 jev-latest。

## 7. 官方依据

本约定核对了官方 [Choice](https://docs.typesafe.ai/primitives/choice)、[Noul](https://docs.typesafe.ai/primitives/noul)、[Score](https://docs.typesafe.ai/primitives/score) 和 [Confidence](https://docs.typesafe.ai/confidence) 文档。公开行为之外的编译方式、评分头、softmax、PiSSA、confidence 公式与训练阶段是 OpenJev 的设计选择。
