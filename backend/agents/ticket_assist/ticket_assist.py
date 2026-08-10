"""
两票辅助 Agent：新人填票防错（规则硬约束 + LLM 语义查漏）+ 常见票型草稿生成。

边界：不介入真实 ERP 开票审核流程；定位为"开票前的辅助检查"。
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from backend.agents.ticket_assist.rules import aggregate_rules, run_rule_checks
from backend.agents.ticket_assist.schemas import TicketCheckResult, TicketDraft
from backend.agents.ticket_assist.templates import generate_draft, match_template
from backend.base.llm_factory import get_structured_llm
from backend.base.logger import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = """你是一位火电厂安全专工，负责审查工作票/操作票草稿的语义完整性。

在规则校验（必填项/证件/危险点覆盖）基础上，重点审查：
1. 危险点分析是否具体（不能只说"高处坠落"，要说明是"脚手架未验收"还是"临边防护缺失"）
2. 安全措施是否与危险点一一对应
3. 操作步骤是否逻辑正确（先停后修、先隔离后作业、关键节点有确认环节）
4. 是否存在漏项或跳步

输出审查结果（JSON）：问题清单 + 修改建议。"""


class TicketAssistAgent:
    """两票辅助 Agent。"""

    def __init__(self, agent_type: str = "ticket_assist"):
        self.agent_type = agent_type

    def check_rule_only(self, draft: TicketDraft) -> TicketCheckResult:
        """纯规则校验（不调 LLM，硬约束）。"""
        result = aggregate_rules(run_rule_checks(draft))
        logger.info(
            "ticket.rule_check", ticket_id=draft.ticket_id,
            passed=result.passed, score=result.score, issues=len(result.issues),
        )
        return result

    async def check_full(self, draft: TicketDraft) -> TicketCheckResult:
        """规则 + LLM 语义审查（双重保障）。"""
        # 1. 规则硬校验
        rule_result = self.check_rule_only(draft)
        if not rule_result.passed:
            logger.info("ticket.llm_skipped", reason="rule_blocked")
            return rule_result  # 硬性失败不再调 LLM

        # 2. LLM 语义查漏
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=(
                f"【票面内容】\n票号:{draft.ticket_id}\n类型:{draft.ticket_type}\n"
                f"任务:{draft.task}\n风险等级:{draft.risk_level}\n"
                f"危险点:{draft.hazard_analysis}\n安全措施:{draft.safety_measures}\n"
                f"步骤:{draft.procedures}\n关联证件:{draft.attachments}"
            )),
        ]
        llm_result: TicketCheckResult = await get_structured_llm(
            self.agent_type, TicketCheckResult
        ).ainvoke(messages)

        # 合并规则与 LLM 结果
        all_issues = rule_result.issues + llm_result.issues
        all_suggestions = rule_result.suggestions + llm_result.suggestions
        score = min(rule_result.score, llm_result.score)
        passed = rule_result.passed and llm_result.passed
        logger.info("ticket.full_check", ticket_id=draft.ticket_id, passed=passed, score=score)
        return TicketCheckResult(
            passed=passed, score=score, issues=all_issues,
            suggestions=all_suggestions,
            required_attachments=rule_result.required_attachments,
        )

    def assist_draft(self, device: str, task_keyword: str, location: str = "") -> dict | None:
        """按常见票型生成草稿（人工确认后走 ERP）。"""
        template = match_template(task_keyword)
        if template is None:
            logger.info("ticket.no_template", keyword=task_keyword)
            return None
        draft = generate_draft(template, device, location)
        logger.info("ticket.draft_generated", template=template.name, device=device)
        return draft
