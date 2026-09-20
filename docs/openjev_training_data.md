# 无标签题库与软标签数据约定

状态：无标签题库、软记录采集和离线回放验证器已实现；正式训练集审核冻结与学生训练加载器待完成。旧硬标签 schema、编译分支、示例及数据已移入[归档](ARCHIVE.md)，当前代码不再维护兼容入口。官方未公开内部训练格式，本文件是 OpenJev 的设计。

## 公共输入保持一致

学生使用 Jev 的 state／questions／criteria 字段。下面只是格式示例，不属于真实训练数据：

```json
{
  "state": {"messages": [{"role": "user", "content": "请把这段英文翻译成中文。"}]},
  "questions": {
    "intent": {
      "type": "choice",
      "instructions": "用户希望完成什么任务？",
      "criteria": {"translate": "翻译文本", "summarize": "概括文本"}
    }
  }
}
```

API 只生成问题及完整候选；本地挂接原始 state，不采用生成器的答案、自报概率或重写聊天。支持裸 JSON 或一个 Markdown 围栏包裹的 JSON；严格解析后检查字段、候选数、重复键、非法数值和实际输入预算，再做语义审核。scripts/synthesize.py 负责 API 调用与任务调度，scripts/question_data.py 负责题目与审核验证，scripts/chat_records.py 负责严格解析和来源去敏。

## 当前两个记录版本

| 记录 | 必需内容 | 用途 |
| --- | --- | --- |
| openjev-question-v1 | id、公共 input、input 哈希、group_id、split、制题与语义审核引用；没有 target | 无标签题库 |
| openjev-distillation-v1 | 题目记录 ID、input 哈希、每题完整语义候选概率、teacher_soft 身份与审计引用 | 软监督 |

概率键与原候选完全对应：Choice 使用 criteria 名称，Noul 使用 false／true，Score 待可信 rubric 就绪再按档位映射。概率必须有限、非负、完整且归一化；缺项不能补零，不能隐式 argmax 成硬标签。每题独立计数，K 不决定题目权重。

学生直接拟合冻结分布，损失为 -sum(q*log p)。不要求先做硬标签训练，不把旧标签转换成 one-hot 冒充本地教师分布。教师分布不等于真实世界条件概率。

## 本地文件布局

```text
questions_run/
  questions.jsonl        # openjev-question-v1，无 target
  plan.json              # 固定来源组与配方槽位
  requests/              # 去敏后的 API 请求／响应
  results/               # 接受、跳过或拒绝原因
  summary.json
  fingerprint.json
  code_snapshot/
  quality_hold.json
  manifest.json
teacher_run/
  soft_labels.jsonl      # openjev-distillation-v1
  records/               # 每题一份软记录
  traces/                # 每次思考、编码映射和原始候选 logits
  fingerprint.json
  code_snapshot.py
  summary.json
  quality_hold.json
  manifest.json
```

软记录通过 question_record_id 和 input_sha256 连接题库，不复制一份可分叉的公共输入。target[question_id] 内部含 label_type=teacher_soft 和 probabilities；这些内部字段不是 Jev 响应格式。教师 T=1 的原始分布和 T=2／4 的对照都保留在 variants 中，尚未确定正式训练采用哪组。

两个入口均支持严格参数绑定的断点恢复，离线验证不调用 API 或教师。操作命令见[执行说明](execution.md)。manifest 是文件完整性凭证，不是训练验收证明。

## 教师审计

scripts/collect_teacher.py 保存每条 thinking 的原始候选 logits、合法候选质量、权重／tokenizer／模板哈希、量化与后端版本、随机种子、编码映射、答案前缀哈希和耗时。variants 保存 M=1／4、T=1／2／4 的比较分布；target 暂指向本次 M 的 T=1 结果，尚非最终训练配方。记录引用 trace_files 与 teacher_fingerprint，manifest 绑定文件内容；思考文本只保存在忽略的本地产物。

本地教师使用经验证的双字母单 token 编码，每题实际答案前缀须再次验证。现有[255 编码审计](data_audit/teacher_codebook.md)只验证模板前缀上的表示；编码与推理轨迹不进入学生输入。

## 共同编译与冻结

scripts/contract.py 保留 validate_request、compile_request 和 response_from_logits；编译版本为 jev-fields-v1。后续软训练加载器必须复用 compile_request，并单独对齐候选概率，不引入第二套输入序列化。

Choice 按语义名称排序且名称与描述可见；Score 保留外部档位映射，每条路径只读自身描述；Noul 使用 No／Yes 分支。路由模型名、问题 ID、标签和审计不进入 token。结构标记字面量转义，超预算拒绝，不静默截断。

冻结前校验题目／分布哈希、来源划分、所有题的质量处置和完整映射，写入新目录及 manifest。保护集合不能训练，quality_hold 未解除的记录不能进入训练版本。来源、共享目录及派生记录保持分组；未冻结数据不能凭概率形状合法就被批准。

当前没有已验收的 teacher_soft 训练集。先完成约 100 条题目与分布的验收，再扩到 1K–3K，并准备独立 dev_model。完整路线见[训练计划](openjev_training_pipeline.md)。
