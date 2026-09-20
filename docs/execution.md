# 本地执行入口

主线为准备数据 → PiSSA 直接软蒸馏 → 验证。旧硬标签制题、冻结、训练与远程接口预检已归档，恢复位置见[归档记录](ARCHIVE.md)。当前保留来源处理、公共输入编译、评分模型、PiSSA 检查点、CLI／HTTP 推理和本地教师编码检查。

**API 制题和本地教师采集已有入口。** 正式训练集审核冻结与学生训练器仍待实现。当前小试尚不能训练，质量结果见[小试报告](data_audit/pipeline_canary.md)。

## 环境

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements-training.txt
```

这是学生评分参考、制题与测试的环境。教师使用独立环境：本次实测 vLLM 0.24.0、Transformers 5.13.0、A800 80GB；NVFP4 权重通过 Marlin W4A16 后端推理，不是 A800 原生 FP4 计算。不要把教师依赖安装到学生环境。模型、tokenizer 和数据目录均通过运行时参数传入；机器路径与凭据不写入仓库。

## 新数据合成：两个执行入口

运行前在终端设置 ANTHROPIC_BASE_URL、ANTHROPIC_AUTH_TOKEN、OPENJEV_TEACHER_DIR 和 OPENJEV_TEACHER_PYTHON。最后一个变量指向独立 vLLM 环境的 Python。不要把实际值写进配置或日志。API 模型名称由 configs/data/questions.json 指定，服务返回模型必须与之匹配。

先执行有预算上限的制题小试：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/synthesize.py run \
  --splits outputs/chat_splits_v2_combined --config configs/data/questions.json \
  --tokenizer "$OPENJEV_TEACHER_DIR" --output outputs/questions_new --max-jobs 12
```

默认配方规划 100 个验收槽位，但 --max-jobs 限制本次尝试的来源数，**不是保证成功条数**。每个来源至少一次制题调用，通过格式检查才额外调用语义审核。传输至多重试两次；每槽最多三个不同来源，失败不拼补 JSON。中／英／日为首轮语言，可在新配置中扩展。当前按来源组唯一分配，不复用同一组凑配额。

继续同一运行时，重复命令并加 --resume；配置、代码、tokenizer、来源 manifest 和接口身份必须相同。已完成的调用缓存复用；修改提示或配方须使用新输出目录。全程只读取 train。

离线检查并导出完整样例：

```bash
.venv/bin/python scripts/synthesize.py validate \
  --run outputs/questions_new --tokenizer "$OPENJEV_TEACHER_DIR"
.venv/bin/python scripts/synthesize.py inspect \
  --run outputs/questions_new --tokenizer "$OPENJEV_TEACHER_DIR" \
  --output outputs/questions_review_new
```

题库在 questions.jsonl；requests 保存去除运行时秘密的请求／响应，results 保存逐次拒绝原因，summary.json 记录实际 K 与长度覆盖，code_snapshot 和 manifest 绑定运行版本。inspect 输出 samples.md。语义审核为另一轮 API 调用，不等于独立人工核验。所有记录保持 training_eligible=false。

对完成检查的题库采集本地概率：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 "$OPENJEV_TEACHER_PYTHON" \
  scripts/collect_teacher.py --questions outputs/questions_new \
  --model "$OPENJEV_TEACHER_DIR" --codebook outputs/teacher_codebook_local_v1 \
  --output outputs/teacher_new --limit 20 --traces 4 --thinking-tokens 4096
```

第一次运行会读取权重计算哈希。可用 --traces 1 做最小调试；正式对照使用 4，自动保存首条 M=1 与全部 M=4 的 T=1／2／4 分布。thinking 未正常结束、上下文超限或实际答案前缀不兼容时拒收，不用截断思考强制打分。输出目录必须为新的相对路径；完全相同参数可加 --resume，已完成和明确拒收的题不会重复执行。参数改变使用新目录。

采集器只在一个答案位置读取完整词表 raw logits，再提取所有合法编码。vLLM 0.24 的 logprob_token_ids 上限为 128，因此这里使用 logprobs=-1，支持提取 255 个编码，不靠 top-k 补零。采集和温度变换没有选中学生基座。

soft_labels.jsonl 保存按原候选名称映射的分布；traces 保存实际 thinking、原始 logits 和合法候选质量；fingerprint.json 记录权重、模板、tokenizer、后端与参数。模型启动输出可能包含运行时路径；不要直接重定向到项目内的可发布日志。

无需加载模型的分布回放：

```bash
python3 scripts/collect_teacher.py --validate-only \
  --questions outputs/questions_new --codebook outputs/teacher_codebook_local_v1 \
  --output outputs/teacher_new
```

该检查验证哈希、输入身份、编码映射、全候选覆盖，并从 logits 重算每组分布。它不能证明题目正确。正式冻结前仍需独立审核、选择分布配方并验收覆盖；当前产物的 quality_hold 不应手动删除或将 eligibility 改成 true。

## 本地教师编码检查

设置运行时 OPENJEV_TEACHER_DIR 后，运行不加载权重的审计：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/teacher_codes.py \
  --tokenizer "$OPENJEV_TEACHER_DIR" --output outputs/teacher_codebook_new
```

输出目录必须不存在。现有 outputs/teacher_codebook_local_v1 包含 tokenizer 哈希、测试前缀、255 个不同单 token 编码、代码快照及 manifest。实际 thinking 的答案前缀仍需逐条验证，详见[编码报告](data_audit/teacher_codebook.md)。

## 学生推理与 HTTP

以下命令需要已有且兼容的检查点；仓库尚未发布正式训练模型。

```bash
.venv/bin/python scripts/infer.py --checkpoint "$OPENJEV_CHECKPOINT" \
  --input examples/jev_request.json --device cuda:0
.venv/bin/python scripts/serve.py --checkpoint "$OPENJEV_CHECKPOINT" \
  --device cuda:0 --port 8000
```

本地服务提供 POST /v1/systemone，绑定回环地址，顺序执行请求。Choice／Noul／Score 的格式、输入编译和响应后处理已实现，但接口正确不等于模型能力已验收。confidence 采用 OpenJev 的前两项概率差，非已知的 Jev 私有公式。

scripts/model.py 保留共享标量评分头、PiSSA 参数构建、残差基底和 adapter 保存／重载。完整软标签训练器待实现；不使用归档的硬标签训练入口启动新路线。

## 验证

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  scripts.test_chat_records scripts.test_prepare_chat_pool \
  scripts.test_contract scripts.test_teacher_codes \
  scripts.test_questions scripts.test_teacher_collection
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 .venv/bin/python -m unittest \
  scripts.test_decision_adapter scripts.test_checkpoint scripts.test_http
python3 scripts/check_docs.py
```

数值测试使用小型 Qwen／Llama，验证软分布损失下候选微批的梯度一致性、联合参数更新、保存重载和服务输出。它们不等于完整软标签训练流水线。测试产物移入带时间戳的 trash，不直接删除。

## 来源准备

当前继续制题应复用 outputs/chat_splits_v2_combined；名称中的 v2 是当前冻结来源版本，不是待删除的旧训练数据。五份划分和来源哈希保持不变。

仅在需要独立新来源实验时运行：

```bash
.venv/bin/python scripts/prepare_chat_pool.py --root "$OPENJEV_DATA_ROOT" \
  --config configs/data/source_pool.json --output outputs/source_pool_new
.venv/bin/python scripts/freeze_chat_splits.py --root "$OPENJEV_DATA_ROOT" \
  --pool outputs/source_pool_new --config configs/data/source_splits.json \
  --output outputs/source_splits_new
```

source_pool.json 是新来源准备配方，不是当前合并划分的重建命令。source_exclusions.json 保留旧探索触及的原文锚点，防止重新流入保护集合；不能因旧实验已归档而移除这些防泄漏约束。

## 接下来执行

提高制题通过率与大目录质量 → 20 条合格题比较教师分布 → 约 100 条审阅冻结 → 1K–3K 数据和独立开发标签 → PiSSA 软训练。数据约定见[格式设计](openjev_training_data.md)，配额、监督与验收见[训练计划](openjev_training_pipeline.md)。
