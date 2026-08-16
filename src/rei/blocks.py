"""Block Kit rendering.

Design goal: the channel should be scannable in five seconds. The current report
puts several hundred UUIDs inline, so the actual signal — which failures happened
and how often — is buried. Here the top-level message carries labels, counts and
blast radius, and the raw quote IDs move into a threaded reply that is one click
away and stays there for anyone who needs to look one up.

Block Kit limits enforced here: 50 blocks per message, 3000 characters per
section, 150 per header, 10 elements per context block.
"""

from __future__ import annotations

from typing import Any

from .models import DailyReport

MAX_BLOCKS = 50
MAX_SECTION_CHARS = 2900
MAX_HEADER_CHARS = 150

_SEVERITY_MARK = {
    "blocking": "🔴",
    "degraded": "🟠",
    "cosmetic": "⚪",
    "unknown": "🔵",
}


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _section(text: str) -> dict[str, Any]:
    return {
        "type": "section",
        "text": {"type": "mrkdwn", "text": _truncate(text, MAX_SECTION_CHARS)},
    }


def _context(text: str) -> dict[str, Any]:
    return {
        "type": "context",
        "elements": [{"type": "mrkdwn", "text": _truncate(text, MAX_SECTION_CHARS)}],
    }


def build_report_blocks(
    report: DailyReport, mention_line: str, max_quotes_inline: int, max_clusters: int
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": _truncate(f"Renewal errors — {report.run_date}", MAX_HEADER_CHARS),
                "emoji": True,
            },
        },
        _context(
            f"*{report.events_extracted}* errors  ·  "
            f"*{len(report.clusters)}* distinct issues  ·  "
            f"*{report.total_unique_quotes}* quotes affected  ·  "
            f"{report.messages_scanned} messages scanned"
        ),
    ]

    if mention_line:
        blocks.append(_section(f"{mention_line} daily renewal error summary"))

    new_clusters = report.new_clusters
    if new_clusters:
        names = ", ".join(f"“{c.cluster.label}”" for c in new_clusters[:3])
        extra = f" and {len(new_clusters) - 3} more" if len(new_clusters) > 3 else ""
        blocks.append(
            _section(f":sparkles: *{len(new_clusters)} new issue type(s) today:* {names}{extra}")
        )

    blocks.append({"type": "divider"})

    if not report.clusters:
        blocks.append(_section("No renewal errors found in this window. :tada:"))
        return blocks[:MAX_BLOCKS]

    shown = report.clusters[:max_clusters]
    for entry in shown:
        mark = _SEVERITY_MARK.get(entry.cluster.severity, "🔵")
        badge = "  `NEW`" if entry.is_new else ""
        trend = ""
        if entry.delta_pct is not None and abs(entry.delta_pct) >= 25:
            arrow = "▲" if entry.delta_pct > 0 else "▼"
            trend = f"  {arrow} {abs(entry.delta_pct):.0f}% vs 7-day avg"

        blocks.append(
            _section(
                f"{mark}  *{entry.event_count}×*  ({entry.share_pct:.0f}%)  "
                f"{entry.cluster.label}{badge}"
            )
        )

        meta = [
            f"`{entry.cluster.category}`",
            f"{entry.unique_quote_count} quote(s)",
        ]
        if trend:
            meta.append(trend.strip())
        if entry.cluster.cluster_id:
            meta.append(f"cluster #{entry.cluster.cluster_id}")
        blocks.append(_context("  ·  ".join(meta)))

        if entry.sample_messages:
            sample = _truncate(entry.sample_messages[0].replace("\n", " "), 300)
            blocks.append(_context(f"```{sample}```"))

        if entry.cluster.root_cause:
            blocks.append(_context(f":mag: {_truncate(entry.cluster.root_cause, 600)}"))
        if entry.cluster.suggested_action:
            blocks.append(_context(f":wrench: {_truncate(entry.cluster.suggested_action, 400)}"))

        if max_quotes_inline and entry.unique_quotes:
            preview = ", ".join(f"`{q}`" for q in entry.unique_quotes[:max_quotes_inline])
            more = (
                f" +{entry.unique_quote_count - max_quotes_inline} more in thread"
                if entry.unique_quote_count > max_quotes_inline
                else ""
            )
            blocks.append(_context(f"{preview}{more}"))

        if len(blocks) >= MAX_BLOCKS - 4:
            break

    hidden = len(report.clusters) - len(shown)
    if hidden > 0:
        blocks.append(
            _context(f"_{hidden} lower-frequency issue type(s) omitted; full list in thread._")
        )

    blocks.append({"type": "divider"})
    blocks.append(
        _context(
            f"{report.unique_templates} distinct error templates  ·  "
            f"window {report.window_start:%Y-%m-%d %H:%M} → {report.window_end:%Y-%m-%d %H:%M} UTC  ·  "
            "full quote IDs in thread"
        )
    )
    return blocks[:MAX_BLOCKS]


def build_thread_text(report: DailyReport) -> str:
    """The complete detail, kept out of the channel view."""
    lines = ["*Full detail — every issue type and affected quote ID*", ""]
    for entry in report.clusters:
        lines.append(
            f"*{entry.event_count}× ({entry.share_pct:.0f}% of today) {entry.cluster.label}* "
            f"(cluster #{entry.cluster.cluster_id}, {entry.cluster.category}, "
            f"{entry.unique_quote_count} quotes)"
        )
        for template in entry.templates[:3]:
            lines.append(f"  • template: `{template[:220]}`")
        if entry.unique_quotes:
            lines.append("  • quotes: " + ", ".join(entry.unique_quotes))
        else:
            lines.append("  • quotes: _none could be attributed to a specific quote_")
        lines.append("")
    return "\n".join(lines)


def build_fallback_text(report: DailyReport, mention_line: str) -> str:
    """Notification text and the accessible fallback for the blocks."""
    head = f"{mention_line} Renewal errors {report.run_date}: " if mention_line else ""
    return (
        f"{head}{report.events_extracted} errors across {len(report.clusters)} issue types, "
        f"{report.total_unique_quotes} quotes affected."
    )


def chunk_text(text: str, limit: int = 3800) -> list[str]:
    """Split a long thread post into Slack-sized chunks on line boundaries."""
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for line in text.split("\n"):
        if size + len(line) + 1 > limit and current:
            chunks.append("\n".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks or [""]
