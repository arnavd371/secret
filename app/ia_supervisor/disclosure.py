"""
Renders a real AI-use disclosure statement from a project's logged
DisclosureEntry history (spec §11) — the actual text a student can
include with their IA/EE submission to document AI tool use, per IB's
academic integrity policy on AI assistance. Deterministic string
formatting only, no model call: every line comes directly from a real
logged entry, nothing paraphrased or summarized by an LLM (a disclosure
statement is exactly the kind of document that must not be generative).
"""

from __future__ import annotations

from app.ia_supervisor.models import DisclosureAssistanceType, DisclosureEntry

_ASSISTANCE_LABEL = {
    DisclosureAssistanceType.COACHING: "AI coaching feedback provided (guiding questions and structural feedback only; no content written by the AI).",
    DisclosureAssistanceType.GHOSTWRITING_REQUEST_REFUSED: "A request for the AI to write content directly was declined, per academic integrity policy.",
    DisclosureAssistanceType.PROJECT_ALREADY_COMPLETE: "No assistance given: this project was already marked complete.",
}


def render_disclosure_statement(student_id: str, project_id: str, entries: list[DisclosureEntry]) -> str:
    lines = [
        f"AI Assistance Disclosure for IA/EE Project: {project_id}",
        f"Student: {student_id}",
        "",
        "The following AI-assisted interactions occurred during this project's development:",
        "",
    ]

    if not entries:
        lines.append("(No AI-assisted interactions were logged for this project.)")
        return "\n".join(lines)

    for entry in entries:
        date = entry.timestamp.date().isoformat()
        label = _ASSISTANCE_LABEL.get(entry.assistance_type, entry.assistance_type.value)
        lines.append(f"- {date} [{entry.stage.value}] {label}")
        if entry.summary:
            lines.append(f"  {entry.summary}")

    return "\n".join(lines)
