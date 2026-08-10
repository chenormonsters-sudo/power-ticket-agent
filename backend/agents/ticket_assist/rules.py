"""
两票规则引擎：必填项校验 + 关联证件校验 + 危险点覆盖校验（不调 LLM，硬约束）。

与 LLM 语义审查形成"规则硬兜底 + LLM 查漏补缺"双重保障。
"""
from __future__ import annotations

import re

from backend.agents.ticket_assist.schemas import TicketCheckResult


def check_required_fields(draft) -> TicketCheckResult:
    """必填项校验：票号/任务/危险点/措施/步骤/人员（任何缺失 → 硬性不通过）。"""
    issues, suggestions = [], []
    score = 100
    missing_required = False

    if not re.search(r"(WP|GK|DQ)[-\s]?\d+", draft.ticket_id or ""):
        issues.append("缺少规范票号（WP/GK/DQ 前缀）")
        score -= 15; missing_required = True
    if not draft.task or len(draft.task.strip()) < 4:
        issues.append("缺少工作任务描述")
        score -= 15; missing_required = True
    if not draft.hazard_analysis or len(draft.hazard_analysis.strip()) < 10:
        issues.append("缺少危险点分析")
        score -= 20; missing_required = True
    if not draft.safety_measures or len(draft.safety_measures.strip()) < 10:
        issues.append("缺少安全措施")
        score -= 20; missing_required = True
    if not draft.procedures or len(draft.procedures.strip()) < 10:
        issues.append("缺少操作步骤")
        score -= 15; missing_required = True
    if not draft.personnel:
        issues.append("缺少人员签名栏（工作负责人/许可人）")
        score -= 10; missing_required = True

    return TicketCheckResult(
        passed=not missing_required, score=max(0, score),
        issues=issues, suggestions=[f"请补充：{i}" for i in issues],
    )


# 风险作业 → 必须附带的证件映射
_RISK_ATTACHMENTS: dict[str, list[str]] = {
    "动火": ["动火证"],
    "受限空间": ["气体检测记录", "受限空间作业审批"],
    "高处": ["高处作业票"],
    "电气": ["停电操作票", "验电记录"],
    "起重": ["起重作业方案"],
}


def check_attachments(draft) -> TicketCheckResult:
    """关联证件校验：按风险关键词检查必附证件。"""
    issues, missing = [], []
    text = f"{draft.task} {draft.safety_measures} {draft.hazard_analysis}"
    for keyword, required in _RISK_ATTACHMENTS.items():
        if keyword in text:
            for req in required:
                if not any(req in a for a in draft.attachments):
                    missing.append(f"{keyword}作业缺少{req}")

    if missing:
        issues.extend(missing)
    return TicketCheckResult(
        passed=len(missing) == 0, score=100 if not missing else 60,
        issues=issues, suggestions=missing,
        required_attachments=missing,
    )


# 危险点覆盖阶段关键词（准备/执行/恢复）
_PHASE_KEYWORDS = {
    "准备": ["隔离", "停电", "挂牌", "验电", "泄压", "置换"],
    "执行": ["监护", "通风", "检测", "防坠", "防火"],
    "恢复": ["恢复", "验收", "试运", "拆除", "清理"],
}


def check_hazard_coverage(draft) -> TicketCheckResult:
    """危险点覆盖校验：是否覆盖作业全流程（准备/执行/恢复）。"""
    issues = []
    text = draft.hazard_analysis or ""
    for phase, keywords in _PHASE_KEYWORDS.items():
        if not any(k in text for k in keywords):
            issues.append(f"危险点分析未覆盖{phase}阶段（缺少{'/'.join(keywords)}等关键词）")
    return TicketCheckResult(
        passed=len(issues) == 0, score=80 if not issues else 60,
        issues=issues, suggestions=[f"补充{phase}阶段危险点" for phase in _PHASE_KEYWORDS if not any(k in (draft.hazard_analysis or "") for k in _PHASE_KEYWORDS[phase])],
    )


def run_rule_checks(draft) -> list[TicketCheckResult]:
    """运行全部规则校验。"""
    return [
        check_required_fields(draft),
        check_attachments(draft),
        check_hazard_coverage(draft),
    ]


def aggregate_rules(results: list[TicketCheckResult]) -> TicketCheckResult:
    """聚合规则结果：任一硬性失败即不通过。"""
    all_issues, all_suggestions, all_missing = [], [], []
    score = 100
    for r in results:
        all_issues.extend(r.issues)
        all_suggestions.extend(r.suggestions)
        all_missing.extend(r.required_attachments)
        score = min(score, r.score)
    return TicketCheckResult(
        passed=all(r.passed for r in results),
        score=score, issues=all_issues,
        suggestions=all_suggestions, required_attachments=all_missing,
    )
