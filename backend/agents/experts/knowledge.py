"""
知识库加载与班组分库。

数据源：
- FAQ 知识库：knowledge_base/faq/faq_knowledge.json（3200 条五级结构）
- 规程文档块：knowledge_base/index/chunks.json（360 块）

班组映射（id 前缀 → 五班组）：
- 6炉/5炉 → 锅炉；汽机 → 汽机；电气 → 电气；燃除 → 燃除
- 公用系统 → 通用库（所有班组专家共享）
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
FAQ_PATH = os.path.join(PROJECT_ROOT, "knowledge_base", "faq", "faq_knowledge.json")
CHUNKS_PATH = os.path.join(PROJECT_ROOT, "knowledge_base", "index", "chunks.json")
CASES_PATH = os.path.join(PROJECT_ROOT, "data", "cases", "cases.json")

TEAMS = ["锅炉", "汽机", "电气", "燃除", "热控"]

_PREFIX_TEAM: dict[str, str] = {
    "6炉": "锅炉", "5炉": "锅炉",
    "汽机": "汽机",
    "电气": "电气",
    "燃除": "燃除",
    "公用系统": "__shared__",
}


def _faq_team(faq_id: str) -> str:
    """按 id 前缀映射 FAQ 到班组（公用系统 → __shared__）。"""
    m = re.match(r"FAQ-([^-]+)", faq_id)
    prefix = m.group(1) if m else ""
    return _PREFIX_TEAM.get(prefix, "__shared__")


def _chunk_team(source: str) -> str:
    """规程文件 → 班组。当前规程均为锅炉侧；通用规程归 __shared__。"""
    if "锅炉" in source:
        return "锅炉"
    return "__shared__"


@dataclass
class KbDoc:
    """知识库文档条目（FAQ 或规程块统一结构）。"""
    doc_id: str
    team: str                 # 班组（__shared__ = 通用）
    text: str                 # 检索文本（FAQ: fault_name+cause+steps；规程: 块文本）
    source: str               # 来源（FAQ id / 规程文件名）
    meta: dict = field(default_factory=dict)


def load_kb() -> list[KbDoc]:
    """加载全部知识库文档。"""
    docs: list[KbDoc] = []

    # FAQ
    with open(FAQ_PATH, encoding="utf-8") as f:
        faq = json.load(f)
    for item in faq:
        fid = item.get("id", "")
        team = _faq_team(fid)
        text = "；".join(filter(None, [
            item.get("fault_name", ""),
            item.get("cause", ""),
            "步骤：" + "；".join(item.get("steps", [])),
            "工器具：" + "、".join(item.get("tools", [])),
        ]))
        docs.append(KbDoc(
            doc_id=fid, team=team, text=text, source="faq",
            meta={"fault_name": item.get("fault_name", ""), "risk_level": item.get("risk_level", "")},
        ))

    # 规程块
    if os.path.exists(CHUNKS_PATH):
        with open(CHUNKS_PATH, encoding="utf-8") as f:
            chunks = json.load(f)
        for i, c in enumerate(chunks):
            src = c.get("source", "")
            docs.append(KbDoc(
                doc_id=f"CHUNK-{i}", team=_chunk_team(src),
                text=c.get("text", ""), source=src,
            ))

    # 已入库复盘案例（知识闭环：案例成为可检索知识；保证与向量缓存 meta 长度一致）
    if os.path.exists(CASES_PATH):
        try:
            with open(CASES_PATH, encoding="utf-8") as f:
                cases = json.load(f)
            for c in cases:
                team = c.get("teams", [""])[0] if c.get("teams") else ""
                if team not in TEAMS:
                    team = "__shared__"
                text = "；".join(filter(None, [
                    f"案例设备：{c.get('device','')}",
                    f"故障原因：{c.get('final_cause','')}",
                    f"处置方案：{c.get('solution','')}",
                    f"执行结果：{c.get('result','')}",
                    f"补充经验：{c.get('extra_notes','')}",
                ]))
                docs.append(KbDoc(
                    doc_id=c.get("case_id", f"CASE-{len(docs)}"),
                    team=team, text=text, source="case",
                    meta={"event_id": c.get("event_id", "")},
                ))
        except Exception:  # noqa: BLE001
            pass

    return docs


def build_team_index(docs: list[KbDoc]) -> dict[str, list[KbDoc]]:
    """按班组分库（__shared__ 并入每个班组）。"""
    base: dict[str, list[KbDoc]] = {t: [] for t in TEAMS}
    shared: list[KbDoc] = []
    for d in docs:
        if d.team == "__shared__":
            shared.append(d)
        elif d.team in base:
            base[d.team].append(d)
        else:
            shared.append(d)  # 未知班组归通用
    for t in TEAMS:
        base[t].extend(shared)
    return base


if __name__ == "__main__":
    docs = load_kb()
    idx = build_team_index(docs)
    print(f"总文档数: {len(docs)}")
    for t in TEAMS:
        print(f"  {t}班: {len(idx[t])} 条")
