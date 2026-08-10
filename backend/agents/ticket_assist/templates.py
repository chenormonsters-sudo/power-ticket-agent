"""
常见票型模板库：典型检修场景的票面模板（非全覆盖，变量预填 + 人工确认）。

模板只沉淀高频票型（磨煤机检修、一次风机振动、干渣机检修、汽机轴瓦检查、电气电机检修），
避免"全量模板规划"的不现实——不同设备/地点的内容差异由人工确认修正。
"""
from __future__ import annotations

from backend.agents.ticket_assist.schemas import CommonTicketTemplate

COMMON_TEMPLATES: list[CommonTicketTemplate] = [
    CommonTicketTemplate(
        template_id="TPL-01",
        name="磨煤机检修",
        ticket_type="工作票",
        task_template="{device}检修：检查磨辊/磨盘磨损、轴承状态，更换磨损件",
        hazard_template="机械伤害、高温烫伤、煤粉自燃、起重伤害、受限空间（磨煤机内部）",
        safety_template="隔离煤源、停电挂牌、泄压、磨煤机内部通风检测合格后进入、专人监护",
        procedures_template="1.办理工作票并许可 2.隔离煤源、停电挂牌 3.磨煤机内部通风、气体检测 4.检修作业（专人监护）5.验收恢复、试运确认",
        required_attachments=["动火证", "受限空间作业审批", "气体检测记录"],
    ),
    CommonTicketTemplate(
        template_id="TPL-02",
        name="一次风机振动处理",
        ticket_type="工作票",
        task_template="{device}振动异常检查处理：检查叶轮积灰/磨损、轴承、对中",
        hazard_template="机械伤害、高处坠落（风机平台）、转动设备伤人",
        safety_template="停电挂牌、叶轮锁止、高处作业系安全带、检修平台围栏检查",
        procedures_template="1.办理工作票 2.停电挂牌、叶轮锁止 3.检查叶轮/轴承/对中 4.处理缺陷 5.恢复试运、测振动",
        required_attachments=["高处作业票"],
    ),
    CommonTicketTemplate(
        template_id="TPL-03",
        name="干渣机检修",
        ticket_type="工作票",
        task_template="{device}检修：检查链条/刮板/减速机，更换磨损部件",
        hazard_template="机械伤害、高温烫伤（渣仓区域）、粉尘",
        safety_template="隔离渣源、停电挂牌、冷却后作业、佩戴防尘口罩",
        procedures_template="1.办理工作票 2.隔离渣源、停电挂牌 3.冷却通风 4.检修作业 5.验收恢复、试运",
        required_attachments=[],
    ),
    CommonTicketTemplate(
        template_id="TPL-04",
        name="汽轮机轴瓦检查",
        ticket_type="工作票",
        task_template="{device}轴瓦检查：检查瓦面磨损/乌金脱落/间隙，必要时更换",
        hazard_template="机械伤害、高温蒸汽烫伤、起重伤害、高处作业",
        safety_template="停机挂闸、蒸汽隔离泄压、吊装专人指挥、高处作业防护",
        procedures_template="1.办理工作票 2.停机、蒸汽隔离泄压 3.揭盖检查轴瓦 4.处理/更换 5.回装验收、试运",
        required_attachments=["高处作业票", "起重作业方案"],
    ),
    CommonTicketTemplate(
        template_id="TPL-05",
        name="电气电机检修",
        ticket_type="工作票",
        task_template="{device}电机检修：检查绕组绝缘/轴承/接线，必要时更换",
        hazard_template="触电、机械伤害、误送电",
        safety_template="停电、验电、接地、挂牌、双重编号确认",
        procedures_template="1.办理工作票 2.停电验电接地挂牌 3.检修作业 4.绝缘测试 5.送电试运确认",
        required_attachments=["停电操作票", "验电记录"],
    ),
]

TEMPLATE_INDEX: dict[str, CommonTicketTemplate] = {
    t.name: t for t in COMMON_TEMPLATES
}


def match_template(keyword: str) -> CommonTicketTemplate | None:
    """按设备/作业关键词匹配常见票型（命中失败返回 None，人工手动填写）。"""
    for t in COMMON_TEMPLATES:
        if keyword in t.name or keyword in t.task_template:
            return t
    return None


def generate_draft(template: CommonTicketTemplate, device: str, location: str = "") -> dict:
    """按模板生成草稿（变量预填，人工确认）。"""
    return {
        "ticket_type": template.ticket_type,
        "device": device,
        "location": location,
        "task": template.task_template.format(device=device),
        "hazard_analysis": template.hazard_template,
        "safety_measures": template.safety_template,
        "procedures": template.procedures_template,
        "attachments": list(template.required_attachments),
        "template_ref": template.name,
    }
