# 火电两票协同智能审查与设备运维多 Agent 系统

事件驱动 + 编排式多 Agent 架构，面向火电厂设备运维与安全生产场景。

## 核心能力

1. **24 小时设备状态感知**：监测 Agent 常驻消费测点数据，规则引擎（阈值/趋势/变化率/跨测点关联）检测异常，告警去抖与双条件归并，输出事件时间线。
2. **跨班组故障会诊**：主控 Agent 按设备归属动态激活班组专家（锅炉/汽机/电气/热控/输煤），并行检索知识库输出诊断意见，整合研判素材、梳理分歧要点、生成结构化参考处置方案。
3. **两票智能辅助审查**：工作票/操作票填写防错（必填项/危险点/安全措施/关联证件校验），常见票型辅助生成草稿。
4. **处置闭环与知识沉淀**：人工确认后复盘入库，知识库增量更新，系统越用越准。

## 设计原则

- 无 AI 决策：所有结论为参考，最终由运维人员决定（分级人工确认）
- 受控自主：编排式多 Agent，确定性交给代码，智能性交给 LLM，底线交给人工
- 私有化合规：生产断网私有化运行（72B + vLLM），数据不出域
- 算力按需：规则常驻，LLM 仅在异常时唤醒

## 架构

```
DCS 测点流 → 监测 Agent（感知）→ 主控 Agent（编排整合）
  → 班组专家 ×5（并行检索诊断）→ 两票辅助 Agent（合规）
  → 复盘 Agent（学习闭环）→ 分级人工确认节点
```

详见 [DESIGN.md](DESIGN.md)。

## 快速开始（开发/演示）

1. 创建 conda 环境并安装依赖：`conda env create -f environment.yml`（或 `pip install -r requirements.txt`）
2. 配置 `.env.local`（模型网关：deepseek 模式用于开发）
3. 构建知识库：`python knowledge_base/scripts/build_kb.py`（首次向量索引由检索器惰性构建并缓存）
4. 启动后端：`python -m uvicorn backend.main:app --port 8000`
5. 演示面板：`streamlit run web/app.py`（三页：监控看板/诊断工作台/知识闭环）
6. 一条龙闭环演示：`python scripts/demo_workflow.py`（告警→会诊→两票→人工确认→复盘入库）

## 评测

```bash
python eval/build_eval_set.py        # 从 FAQ 构建 100 条评测集
python eval/run_eval.py              # 混合检索指标（Top-1/3/5 命中率，秒级）
python eval/run_eval.py --rerank     # Reranker 对比实验
```

当前基线：Top-1 63% / Top-3 96% / Top-5 100%（0.3s/100 条）。
Reranker 对比结论：字面查询场景为负优化（Top-1 54%），按语义查询场景可选开启。

## 部署

| 模式 | 模型 | 说明 |
|---|---|---|
| 开发/演示 | DeepSeek API | 模型网关配置切换，本地验证流程 |
| 生产 | 72B + vLLM（私有化） | 内网断网运行，对接 DCS 数据接口，满足等保合规 |

容器化：`docker compose up -d`（backend + streamlit + mysql + milvus）；
追踪审计：`docker compose -f docker-compose.langfuse.yml up -d` 后配置 `LANGFUSE_ENABLED=true`。

## 目录结构

```
backend/           后端服务（FastAPI）
  agents/          各 Agent 模块（monitor/experts/orchestrator/ticket_assist/debrief）
  api/             REST API
  base/            基础设施（配置/模型网关/日志/重试/追踪）
  graphs/          LangGraph 图（诊断图 + 完整工作流图）
knowledge_base/    知识库（FAQ/规程文档/索引缓存）
models/            本地模型（bge-m3 嵌入 / bge-reranker 重排）
scripts/           演示与工具脚本（demo_workflow 一条龙）
web/               Streamlit 演示面板
eval/              评测集与评估脚本
docs/              架构与部署文档
```

## 测试与评估

`pytest` 覆盖核心链路；`eval/` 含评测集与指标脚本（端到端通过率、裁定准确率对比单 Agent 基线、人工确认次数、端到端时延）。

## 免责说明

演示数据为同规格模拟工况数据 + 脱敏历史案例；生产对接脱敏 DCS 接口。
