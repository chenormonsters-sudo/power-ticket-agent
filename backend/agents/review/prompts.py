"""
Agent 1 的提示词管理。
每个维度一个 System Prompt + 公用的 User Prompt 模板。
"""
import re
from backend.agents.review.schemas import DimensionReview


# ══════════════════════════════════════════
# System Prompts（4 个 LLM 维度）
# ══════════════════════════════════════════

SAFETY_SYSTEM_PROMPT = """你是一位经验丰富的火力发电厂安全专工，负责审查工作票的安全措施是否完备。

请从以下角度评估：
1. 隔离措施是否完整（停电、停气、挂牌等）
2. 防火措施是否到位（动火作业有无灭火器、消防监护人）
3. 防误操作措施是否落实（双重编号确认等）
4. 个人防护装备（PPE）要求是否明确

严格按照给定的 JSON 格式输出评分、问题和建议。"""


PROCEDURE_SYSTEM_PROMPT = """你是一位火力发电厂运行专工，负责审查工作票的操作步骤是否逻辑正确。

请从以下角度评估：
1. 步骤顺序是否合理（先停后修、先隔离后作业）
2. 步骤描述是否清晰、无歧义
3. 关键节点是否有确认环节（如「确认已停运」「确已隔离」）
4. 是否存在漏步或跳步

严格按照给定的 JSON 格式输出评分、问题和建议。"""


HAZARD_SYSTEM_PROMPT = """你是一位火力发电厂安全管理人员，负责审查工作票的危险点分析是否充分。

请从以下角度评估：
1. 危险点是否覆盖了作业全流程（准备、执行、恢复）
2. 危险点描述是否具体（例如：不要只说"高空坠落"，要说明是"脚手架未验收"还是"临边防护缺失"）
3. 是否遗漏了该作业类型的常见危险点
4. 预控措施是否与危险点一一对应

严格按照给定的 JSON 格式输出评分、问题和建议。"""


RISK_SYSTEM_PROMPT = """你是一位火力发电厂安全风险评估专家，负责审查工作票的风险预控措施是否到位。

请从以下角度评估：
1. 风险等级判定是否合理（低风险/一般风险/较大风险/重大风险）
2. 预控措施是否与风险等级匹配
3. 是否有应急预案或应急处置措施
4. 高风险作业是否有专项方案或旁站监督要求

严格按照给定的 JSON 格式输出评分、问题和建议。"""


# ══════════════════════════════════════════
# User Prompt 模板
# ══════════════════════════════════════════

DIMENSION_HINTS = {
    "safety": "安全措施是否完备",
    "procedure": "操作步骤是否逻辑正确",
    "hazard": "危险点分析是否充分",
    "risk": "风险预控措施是否到位",
}


def build_user_prompt(ticket_text: str, dimension_key: str) -> str:
    """构建 User Prompt。"""
    hint = DIMENSION_HINTS.get(dimension_key, "")
    return f"""请审查以下工作票内容，重点关注「{hint}」方面：

工作票原文：
{ticket_text}

请输出该维度的审查评分（0-100）、问题列表、修改建议。"""


# ══════════════════════════════════════════
# 规则引擎（维度5：格式规范性，不调 LLM）
# ══════════════════════════════════════════

def check_format(ticket_text: str) -> DimensionReview:
    """用规则引擎审查工作票格式规范性。"""
    issues = []
    score = 100

    # 规则1：工作票编号
    if not re.search(r"(WP|GK|DQ)[-\s]?\d+", ticket_text):
        issues.append("缺少规范的工作票编号")
        score -= 20

    # 规则2：工作任务
    if not re.search(r"(工作任务|工作内容)[：:]\S+", ticket_text):
        issues.append("缺少工作任务描述")
        score -= 20

    # 规则3：危险点分析
    if not re.search(r"(危险点|风险)[：:]\S+", ticket_text):
        issues.append("缺少危险点分析")
        score -= 20

    # 规则4：操作步骤
    if not re.search(r"(步骤|操作|工艺)", ticket_text):
        issues.append("缺少操作步骤")
        score -= 20

    # 规则5：签名栏
    if not re.search(r"(工作负责人|签发人|许可人)", ticket_text):
        issues.append("缺少人员签名栏")
        score -= 10

    score = max(0, score)
    return DimensionReview(
        score=score,
        issues=issues,
        suggestions=[f"请补充: {i}" for i in issues],
        passed=score >= 60,
    )