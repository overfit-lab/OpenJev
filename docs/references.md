# 研究参考

本项目参考 Jev 公开描述设计接口与实验。以下是研究使用的官方来源入口，不表示 OpenJev 复现了其内部实现或实测性能；协议兼容范围以[接口约定](openjev_api_contract.md)为准。

| 来源 | 用途 |
| --- | --- |
| [TypeSafe AI](https://typesafe.ai/) | Jev 产品介绍 |
| [Introducing System One Models & Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) | Diogo Almeida 发布文章，System One 与 RLCD 的公开描述 |
| [Introduction](https://docs.typesafe.ai/introduction) | 请求与类型化决策概述 |
| [Choice](https://docs.typesafe.ai/primitives/choice) | 动态候选接口 |
| [Score](https://docs.typesafe.ai/primitives/score) | 描述性档位接口 |
| [Confidence](https://docs.typesafe.ai/confidence) | 置信度的公开说明 |

旧的网页与文章全文摘录已移入本地 trash，恢复位置见[归档记录](ARCHIVE.md)。源码分发保留来源链接和我们自己的设计说明，不再携带全文摘录。

公开描述未提供足够信息确认 Jev 的层结构、参数规模或 RLCD 训练算法。OpenJev 的共享标量评分头、候选上下文和直接软蒸馏是本项目的实现选择。模型及数据的上游许可见[第三方说明](../THIRD_PARTY_NOTICES.md)。
