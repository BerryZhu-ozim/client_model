# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] – 2025-05-26

### Added
- 添加项目骨架：智能客服系统（Project 1）和语音机器人系统（Project 2）客户端选型与部署框架。
- 新增文档 **README.md**：包含项目背景、环境配置、模型启动及 FastAPI 服务部署说明。
- 新增 **.gitignore**：忽略 Python 缓存、虚拟环境、IDE 配置、日志文件、大模型权重及本地模型目录内容，仅保留空 `.gitkeep`。
- 新增部署脚本及示例：`sglang.launch_server` 启动命令、`uvicorn` FastAPI 服务命令示例。
- 新增推理脚本目录：`client_model/voice_agent`、`client_model/customer_service`。

### Breaking Changes
- 无

## [0.1.1] – 2025-05-30
### Changed

voice_agent：调整 detect-client 与 chat-client 的联动逻辑，使其符合决策树（decision tree）流程。

customer_service：优化提示工程（prompt engineering），改进工具调用逻辑，并新增 CLI 工具 cs_cli.py。

### Breaking Changes
- 无

