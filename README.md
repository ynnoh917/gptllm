# 哈哈哈 1号测试模型 (100M GPT-2)

基于 Hugging Face GPT-2 架构从零训练的 100M 中文小模型。

## 云端训练
1. HF 创建 Model 仓库，配置 GitHub Secrets (`HF_TOKEN`) 和 Variables (`HF_CKPT_REPO`)。
2. Actions 触发 `Train 100M HaHaHa`，自动执行 10 个 2.5 小时 Job 并生成 `ha-ha-ha-100m.gguf`。

## 本地推理
1. 导入 GGUF 到 PocketPal 即可直接对话。
2. 需要联网时，创建 `api.json` 并运行 `serve.py`。
