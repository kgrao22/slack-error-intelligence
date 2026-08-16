"""Consolidated historical report over everything in the corpus.

The daily Slack post answers "what happened yesterday". This answers the
different and more useful question: "what is actually wrong with renewals, and
which of it is worth fixing first". It reads the same database and adds no
model calls — every number is SQL.

The split that matters most is engineering-actionable versus customer-caused. A
card declined for insufficient funds is not a bug and no amount of engineering
removes it; an occupation ID missing from the reference data is a defect that
recurs every single day until someone fixes it. Reporting them in one
undifferentiated list is what makes an error channel feel like weather rather
than a work queue.
"""

from __future__ import annotations

import html
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations

from .store import Store

# Failures the renewal service causes, versus failures customers cause.
#
# Ownership is NOT read from the cluster category. Category is assigned by a
# language model and is legitimately unstable for ambiguous clusters: across two
# runs on identical data, "Multiple Stripe customers exist with the given email"
# moved from `data` to `payment`, which flipped it from an engineering defect to
# a customer problem and moved the headline share by five points. Nothing about
# the data had changed. A non-deterministic label must not drive a headline
# number, so ownership is derived from a closed, deterministic rule instead.
#
# Customer-caused means exactly one thing: the customer's payment instrument or
# their action failed. Everything else — including payment-*adjacent* data
# defects like duplicate customer records — is ours.
_CUSTOMER_SIGNALS = re.compile(
    r"card declined|card was declined|insufficient funds|card has expired|expired_card|"
    r"card number is incorrect|incorrect_number|does not support this type|"
    r"transaction_not_allowed|generic_decline|do not honou?r|"
    r"3d ?secure|authentication required|"
    r"no default payment method|could not find a default payment method",
    re.I,
)
ENGINEERING_CATEGORIES = {"data", "integration", "validation", "config", "auth", "timeout"}
CUSTOMER_CATEGORIES = {"payment"}

# Categorical slots in fixed order, never cycled (dataviz palette, validated
# light and dark). `unknown` is deliberately not a series colour: it is the
# absence of a classification, not an identity, so it takes muted ink.
CATEGORY_ORDER = ["data", "payment", "integration", "validation", "auth", "timeout", "config"]
LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9"]
MUTED = "#898781"

SEVERITY_RANK = {"blocking": 0, "degraded": 1, "cosmetic": 2, "unknown": 3}


@dataclass
class Issue:
    cluster_id: int
    label: str
    category: str
    severity: str
    events: int
    quotes: int
    templates: int
    first_seen: str
    last_seen: str
    active_days: int
    weekly: dict[str, int] = field(default_factory=dict)
    template_text: str = ""
    corpus_total: int = 0

    @property
    def share_pct(self) -> float:
        """Share of all errors in the window. A count alone reads as alarming or
        trivial depending on the reader's guess at the denominator."""
        return self.events / self.corpus_total * 100 if self.corpus_total else 0.0

    @property
    def owner(self) -> str:
        """Who has to act. Deterministic — never the model's category choice.

        Matched against the label and the cluster's masked templates, both of
        which are stable text, rather than the category, which is not.
        """
        if _CUSTOMER_SIGNALS.search(f"{self.label} {self.template_text}"):
            return "customer"
        if self.category in ENGINEERING_CATEGORIES or self.category in CUSTOMER_CATEGORIES:
            return "engineering"
        return "unclassified"

    @property
    def per_day(self) -> float:
        return self.events / max(self.active_days, 1)

    @property
    def is_engineering(self) -> bool:
        return self.owner == "engineering"

    @property
    def quick_win_reasons(self) -> list[str]:
        """Why this looks cheap to fix. Computed, and shown so it can be argued with."""
        reasons: list[str] = []
        if self.templates == 1:
            reasons.append("one exact error string, so one code path")
        if self.quotes and self.events / self.quotes >= 3:
            reasons.append(
                f"{self.events / self.quotes:.0f} failures per affected quote — retrying, not resolving"
            )
        if self.category in {"data", "config"} or "not found" in self.label.lower():
            reasons.append("reference-data or configuration gap rather than logic")
        if self.active_days >= 30 and self.per_day >= 1:
            reasons.append(f"present on {self.active_days} days and still recurring")
        return reasons


@dataclass
class Link:
    """Two clusters that keep appearing under the same correlation ID.

    One failure often produces several log lines — a duplicate-email lookup
    fails, then account creation fails as a consequence — and each line becomes
    its own cluster. Counted separately they inflate the issue list and hide the
    fact that fixing one removes both. The correlation ID the service already
    stamps on every message is the join that proves it, rather than eyeballing
    two clusters with suspiciously similar date ranges.
    """

    a_id: int
    a_label: str
    a_events: int
    a_category: str
    b_id: int
    b_label: str
    b_events: int
    b_category: str
    shared: int
    containment: float
    a_first_pct: float

    # Every consumer reads the pair in temporal order, so orient it once here.
    # Deriving the display order in one place and the verdict in another let
    # them disagree: a card could read "A then B" above a verdict saying "the
    # second appears to cause the first", where "second" meant B in cluster-id
    # order rather than B as displayed.

    @property
    def _a_is_first(self) -> bool:
        return self.a_first_pct >= 0.5

    @property
    def first_label(self) -> str:
        return self.a_label if self._a_is_first else self.b_label

    @property
    def second_label(self) -> str:
        return self.b_label if self._a_is_first else self.a_label

    @property
    def first_category(self) -> str:
        return self.a_category if self._a_is_first else self.b_category

    @property
    def order_pct(self) -> float:
        """How often the displayed order actually held, 0.5-1.0."""
        return max(self.a_first_pct, 1 - self.a_first_pct)

    @property
    def verdict(self) -> str:
        if self.order_pct >= 0.8:
            return (
                f"consistent order in {self.order_pct * 100:.0f}% of traces — "
                "the first plausibly causes the second"
            )
        return (
            f"order varies ({self.order_pct * 100:.0f}%) — "
            "likely emitted together rather than one causing the other"
        )


@dataclass
class Digest:
    generated_at: str
    window_start: str
    window_end: str
    total_events: int
    total_quotes: int
    issues: list[Issue]
    weeks: list[str]
    links: list[Link] = field(default_factory=list)
    correlated_events: int = 0

    @property
    def distinct_incidents(self) -> int:
        """Issues, after collapsing every linked group into one.

        Union-find over the links, so a chain A-B-C counts once rather than
        twice."""
        parent: dict[int, int] = {i.cluster_id: i.cluster_id for i in self.issues}

        def find(x: int) -> int:
            while parent.get(x, x) != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for link in self.links:
            ra, rb = find(link.a_id), find(link.b_id)
            if ra != rb:
                parent[ra] = rb
        return len({find(i.cluster_id) for i in self.issues})

    @property
    def engineering_events(self) -> int:
        return sum(i.events for i in self.issues if i.is_engineering)

    @property
    def customer_events(self) -> int:
        return sum(i.events for i in self.issues if i.owner == "customer")

    @property
    def engineering_share(self) -> float:
        return self.engineering_events / self.total_events * 100 if self.total_events else 0.0

    @property
    def quick_wins(self) -> list[Issue]:
        """Engineering-owned, recurring, and with at least two reasons to think
        it is cheap. Ranked by volume, capped at six so the section stays a
        shortlist rather than a second copy of the table."""
        candidates = [
            i
            for i in self.issues
            if i.is_engineering and i.events >= 10 and len(i.quick_win_reasons) >= 2
        ]
        return sorted(candidates, key=lambda i: -i.events)[:6]


def _iso_week(value: str) -> str:
    try:
        return datetime.fromisoformat(value).strftime("%G-W%V")
    except ValueError:
        return "unknown"


def build_digest(store: Store, days: int | None = None) -> Digest:
    conn = store.conn
    where, params = "", []
    if days:
        where = "WHERE occurred_at >= datetime('now', ?)"
        params = [f"-{days} days"]

    bounds = conn.execute(
        f"SELECT MIN(occurred_at) lo, MAX(occurred_at) hi, COUNT(*) n, "
        f"COUNT(DISTINCT quote_id) q FROM events {where}",
        params,
    ).fetchone()
    if not bounds or not bounds["n"]:
        raise SystemExit("No events in the corpus. Run `rei backfill` first.")

    rows = conn.execute(
        f"""
        SELECT e.cluster_id,
               c.label, c.category, c.severity, c.template_count,
               COUNT(*) AS events,
               COUNT(DISTINCT e.quote_id) AS quotes,
               MIN(e.occurred_at) AS first_seen,
               MAX(e.occurred_at) AS last_seen,
               COUNT(DISTINCT date(e.occurred_at)) AS active_days
        FROM events e JOIN clusters c ON c.cluster_id = e.cluster_id
        {where}
        GROUP BY e.cluster_id
        ORDER BY events DESC
        """,
        params,
    ).fetchall()

    # Inputs the ownership rule needs, so it never has to consult the category.
    template_text = {
        r["cluster_id"]: r["texts"]
        for r in conn.execute(
            "SELECT cluster_id, GROUP_CONCAT(text, ' ') AS texts FROM templates "
            "WHERE cluster_id IS NOT NULL GROUP BY cluster_id"
        )
    }
    weekly = conn.execute(f"SELECT cluster_id, occurred_at FROM events {where}", params).fetchall()
    by_cluster: dict[int, dict[str, int]] = {}
    weeks: set[str] = set()
    for row in weekly:
        week = _iso_week(row["occurred_at"])
        weeks.add(week)
        by_cluster.setdefault(row["cluster_id"], {}).setdefault(week, 0)
        by_cluster[row["cluster_id"]][week] += 1

    issues = [
        Issue(
            cluster_id=r["cluster_id"],
            label=r["label"],
            category=r["category"] or "unknown",
            severity=r["severity"] or "unknown",
            events=r["events"],
            quotes=r["quotes"],
            templates=r["template_count"],
            first_seen=r["first_seen"],
            last_seen=r["last_seen"],
            active_days=r["active_days"],
            weekly=by_cluster.get(r["cluster_id"], {}),
            template_text=template_text.get(r["cluster_id"], ""),
            corpus_total=bounds["n"],
        )
        for r in rows
    ]

    links, correlated = _build_links(conn, where, params, {i.cluster_id: i for i in issues})

    return Digest(
        links=links,
        correlated_events=correlated,
        generated_at=datetime.now().strftime("%d %b %Y"),
        window_start=(bounds["lo"] or "")[:10],
        window_end=(bounds["hi"] or "")[:10],
        total_events=bounds["n"],
        total_quotes=bounds["q"],
        issues=issues,
        weeks=sorted(w for w in weeks if w != "unknown"),
    )


MIN_SHARED = 5
MIN_CONTAINMENT = 0.6


def _build_links(
    conn: sqlite3.Connection, where: str, params: list, issues: dict[int, Issue]
) -> tuple[list[Link], int]:
    """Find cluster pairs that co-occur on the same correlation ID.

    Containment rather than Jaccard: the question is not "are these two the same
    size" but "does one essentially always come with the other". A 477-event
    cluster fully contained in a 4,000-event cluster is still one incident with
    two log lines, and Jaccard would score that pair near 0.1 and miss it.
    """
    clause = "WHERE correlation_id IS NOT NULL"
    if where:
        clause = where + " AND correlation_id IS NOT NULL"

    rows = conn.execute(
        f"SELECT cluster_id, correlation_id, MIN(occurred_at) AS t FROM events "
        f"{clause} GROUP BY cluster_id, correlation_id",
        params,
    ).fetchall()
    if not rows:
        return [], 0

    per_cluster: dict[int, int] = {}
    by_correlation: dict[str, dict[int, str]] = {}
    for row in rows:
        per_cluster[row["cluster_id"]] = per_cluster.get(row["cluster_id"], 0) + 1
        by_correlation.setdefault(row["correlation_id"], {})[row["cluster_id"]] = row["t"]

    shared: dict[tuple[int, int], int] = {}
    a_first: dict[tuple[int, int], int] = {}
    for members in by_correlation.values():
        if len(members) < 2:
            continue
        for a, b in combinations(sorted(members), 2):
            key = (a, b)
            shared[key] = shared.get(key, 0) + 1
            if members[a] < members[b]:
                a_first[key] = a_first.get(key, 0) + 1

    links: list[Link] = []
    for (a, b), count in shared.items():
        if count < MIN_SHARED or a not in issues or b not in issues:
            continue
        containment = count / max(min(per_cluster.get(a, 0), per_cluster.get(b, 0)), 1)
        if containment < MIN_CONTAINMENT:
            continue
        links.append(
            Link(
                a_id=a,
                a_label=issues[a].label,
                a_events=issues[a].events,
                a_category=issues[a].category,
                b_id=b,
                b_label=issues[b].label,
                b_events=issues[b].events,
                b_category=issues[b].category,
                shared=count,
                containment=min(containment, 1.0),
                a_first_pct=a_first.get((a, b), 0) / count,
            )
        )
    links.sort(key=lambda link: -link.shared)
    return links, len(by_correlation)


# --- rendering -------------------------------------------------------------


def _colour_vars() -> str:
    light = "\n".join(f"      --cat-{name}: {LIGHT[i]};" for i, name in enumerate(CATEGORY_ORDER))
    dark = "\n".join(f"      --cat-{name}: {DARK[i]};" for i, name in enumerate(CATEGORY_ORDER))
    return light, dark


def _esc(text: str) -> str:
    return html.escape(str(text), quote=True)


def _bar_chart(issues: list[Issue], limit: int = 15) -> str:
    """Ranked horizontal bars. Magnitude is the job, so length carries it and
    colour only carries category identity. Every bar is directly labelled, which
    is also the relief the light-mode contrast warning requires."""
    top = issues[:limit]
    if not top:
        return "<p class='empty'>No issues.</p>"
    peak = max(i.events for i in top)
    rows = []
    for issue in top:
        width = issue.events / peak * 100
        cat = issue.category if issue.category in CATEGORY_ORDER else "unknown"
        fill = f"var(--cat-{cat})" if cat in CATEGORY_ORDER else MUTED
        rows.append(
            f"""      <div class="bar-row" tabindex="0"
           aria-label="{_esc(issue.label)}: {issue.events} errors, {issue.share_pct:.1f}% of total, {issue.category}">
        <div class="bar-label" title="{_esc(issue.label)}">{_esc(issue.label)}</div>
        <div class="bar-track">
          <div class="bar-fill" style="width:{width:.2f}%;background:{fill}"></div>
          <span class="bar-value">{issue.events:,}</span>
          <span class="bar-share">{issue.share_pct:.1f}%</span>
        </div>
        <div class="bar-meta">{issue.quotes:,} quotes</div>
      </div>"""
        )
    return "\n".join(rows)


def _weekly_chart(digest: Digest) -> str:
    """Weekly volume, stacked by category. Adjacent stack segments are separated
    by a 2px surface gap so the boundary never relies on hue alone."""
    if len(digest.weeks) < 2:
        return "<p class='empty'>Not enough history yet for a weekly trend.</p>"

    totals: dict[str, dict[str, int]] = {w: {} for w in digest.weeks}
    for issue in digest.issues:
        cat = issue.category if issue.category in CATEGORY_ORDER else "unknown"
        for week, count in issue.weekly.items():
            if week in totals:
                totals[week][cat] = totals[week].get(cat, 0) + count

    peak = max((sum(v.values()) for v in totals.values()), default=1) or 1
    columns = []
    for week in digest.weeks:
        segments = []
        for cat in [*CATEGORY_ORDER, "unknown"]:
            count = totals[week].get(cat, 0)
            if not count:
                continue
            fill = f"var(--cat-{cat})" if cat in CATEGORY_ORDER else MUTED
            segments.append(
                f'<div class="seg" style="height:{count / peak * 100:.2f}%;background:{fill}"'
                f' title="{_esc(cat)}: {count:,}"></div>'
            )
        total = sum(totals[week].values())
        columns.append(
            f"""        <div class="col" tabindex="0" aria-label="Week {week}: {total} errors">
          <div class="stack">{"".join(reversed(segments))}</div>
          <div class="col-total">{total:,}</div>
          <div class="col-label">{_esc(week.split("-")[1])}</div>
        </div>"""
        )
    return "\n".join(columns)


def _legend(digest: Digest) -> str:
    """Only categories actually present. A legend entry for a series that is not
    on the chart is noise, and invites the reader to hunt for a colour that is
    not there."""
    seen = {i.category if i.category in CATEGORY_ORDER else "unknown" for i in digest.issues}
    present = [c for c in [*CATEGORY_ORDER, "unknown"] if c in seen]
    items = []
    for cat in present:
        fill = f"var(--cat-{cat})" if cat in CATEGORY_ORDER else MUTED
        items.append(f'<span class="key"><i style="background:{fill}"></i>{_esc(cat)}</span>')
    return "".join(items)


def _quick_wins(digest: Digest) -> str:
    wins = digest.quick_wins
    if not wins:
        return "<p class='empty'>Nothing meets the shortlist criteria in this window.</p>"
    cards = []
    for issue in wins:
        reasons = "".join(f"<li>{_esc(r)}</li>" for r in issue.quick_win_reasons)
        cards.append(
            f"""      <article class="win">
        <header><span class="dot" style="background:var(--cat-{issue.category})"></span>
          <h3>{_esc(issue.label)}</h3></header>
        <p class="win-stat"><strong>{issue.events:,}</strong> failures —
           <strong>{issue.share_pct:.1f}%</strong> of all errors ·
           {issue.quotes:,} quotes · {issue.active_days} active days</p>
        <ul>{reasons}</ul>
      </article>"""
        )
    return "\n".join(cards)


def _links(digest: Digest) -> str:
    if not digest.correlated_events:
        return (
            "<p class='empty'>No correlation IDs in this corpus yet — re-run "
            "<code>rei backfill</code> to capture them.</p>"
        )
    if not digest.links:
        return (
            "<p class='empty'>No two issues share correlation IDs often enough "
            f"to be the same incident (checked {digest.correlated_events:,} traces).</p>"
        )
    rows = []
    for link in digest.links:
        cat = link.first_category if link.first_category in CATEGORY_ORDER else None
        fill = f"var(--cat-{cat})" if cat else MUTED
        rows.append(
            f"""      <article class="win">
        <header><span class="dot" style="background:{fill}"></span>
          <h3>{_esc(link.first_label)}</h3></header>
        <p class="win-stat">then <strong>{_esc(link.second_label)}</strong></p>
        <ul>
          <li><strong>{link.shared:,}</strong> shared correlation IDs —
              {link.containment * 100:.0f}% of the smaller issue</li>
          <li>{_esc(link.verdict)}</li>
          <li>{link.a_events + link.b_events:,} errors across both, one incident</li>
        </ul>
      </article>"""
        )
    return "\n".join(rows)


def _table(digest: Digest) -> str:
    rows = []
    for issue in digest.issues:
        owner = issue.owner
        rows.append(
            f"""        <tr>
          <td class="lbl">{_esc(issue.label)}</td>
          <td>{_esc(issue.category)}</td>
          <td>{_esc(issue.severity)}</td>
          <td class="own {owner}">{owner}</td>
          <td class="num">{issue.events:,}</td>
          <td class="num">{issue.share_pct:.1f}%</td>
          <td class="num">{issue.quotes:,}</td>
          <td class="num">{issue.per_day:.1f}</td>
          <td class="num">{issue.active_days}</td>
          <td class="num">{_esc(issue.first_seen[:10])}</td>
          <td class="num">{_esc(issue.last_seen[:10])}</td>
        </tr>"""
        )
    return "\n".join(rows)


def render_html(digest: Digest) -> str:
    light_vars, dark_vars = _colour_vars()
    eng_pct = digest.engineering_share
    cust_pct = digest.customer_events / digest.total_events * 100 if digest.total_events else 0

    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Renewal errors — {_esc(digest.window_start)} to {_esc(digest.window_end)}</title>
<style>
  .viz-root {{
    color-scheme: light;
    --surface-1: #fcfcfb; --plane: #f9f9f7;
    --text-primary: #0b0b0b; --text-secondary: #52514e; --muted: #898781;
    --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
    --critical: #d03b3b; --good: #006300;
{light_vars}
  }}
  @media (prefers-color-scheme: dark) {{
    :root:where(:not([data-theme="light"])) .viz-root {{
      color-scheme: dark;
      --surface-1: #1a1a19; --plane: #0d0d0d;
      --text-primary: #ffffff; --text-secondary: #c3c2b7; --muted: #898781;
      --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
      --critical: #d03b3b; --good: #0ca30c;
{dark_vars}
    }}
  }}
  :root[data-theme="dark"] .viz-root {{
    color-scheme: dark;
    --surface-1: #1a1a19; --plane: #0d0d0d;
    --text-primary: #ffffff; --text-secondary: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    --critical: #d03b3b; --good: #0ca30c;
{dark_vars}
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; background: var(--plane); color: var(--text-primary);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif; font-size: 14px;
    line-height: 1.5; }}
  .wrap {{ max-width: 1100px; margin: 0 auto; padding: 32px 20px 64px; }}
  header.top {{ display:flex; justify-content:space-between; align-items:flex-start;
    gap:16px; flex-wrap:wrap; margin-bottom: 4px; }}
  h1 {{ font-size: 24px; margin: 0 0 4px; letter-spacing: -0.01em; }}
  .sub {{ color: var(--text-secondary); margin: 0; }}
  button.theme {{ background: var(--surface-1); color: var(--text-secondary);
    border: 1px solid var(--border); border-radius: 8px; padding: 6px 12px;
    cursor: pointer; font: inherit; }}
  section {{ background: var(--surface-1); border: 1px solid var(--border);
    border-radius: 12px; padding: 20px; margin-top: 20px; }}
  h2 {{ font-size: 15px; margin: 0 0 4px; }}
  .hint {{ color: var(--muted); margin: 0 0 16px; font-size: 13px; }}
  .tiles {{ display:grid; grid-template-columns: repeat(auto-fit,minmax(180px,1fr));
    gap: 12px; margin-top: 20px; }}
  .tile {{ background: var(--surface-1); border:1px solid var(--border);
    border-radius:12px; padding:16px; }}
  .tile .v {{ font-size: 30px; font-weight: 600; letter-spacing:-0.02em; }}
  .tile .k {{ color: var(--text-secondary); font-size: 13px; }}
  .split {{ display:flex; height: 34px; border-radius: 8px; overflow: hidden; gap: 2px; }}
  .split div {{ display:flex; align-items:center; justify-content:center; color:#fff;
    font-size: 12px; font-weight: 600; }}
  .split-key {{ display:flex; gap:20px; margin-top:10px; color: var(--text-secondary);
    font-size: 13px; flex-wrap: wrap; }}
  .bar-row {{ display:grid; grid-template-columns: minmax(180px,340px) 1fr 90px;
    gap:12px; align-items:center; padding: 5px 0; border-radius: 6px; }}
  .bar-row:hover, .bar-row:focus-visible {{ background: var(--plane); outline: none; }}
  .bar-label {{ overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
  .bar-track {{ display:flex; align-items:center; gap:8px; }}
  .bar-fill {{ height: 14px; border-radius: 0 4px 4px 0; min-width: 2px; }}
  .bar-value {{ font-variant-numeric: tabular-nums; color: var(--text-secondary); }}
  .bar-share {{ font-variant-numeric: tabular-nums; color: var(--muted); font-size: 12px;
    min-width: 46px; text-align: right; }}
  .bar-meta {{ color: var(--muted); text-align:right; font-variant-numeric: tabular-nums; }}
  .weekly {{ display:flex; align-items:flex-end; gap:6px; height: 200px;
    border-bottom: 1px solid var(--axis); padding-bottom: 0; overflow-x:auto; }}
  .col {{ flex:1 1 0; min-width:26px; display:flex; flex-direction:column;
    justify-content:flex-end; height:100%; }}
  .col:hover .col-total {{ color: var(--text-primary); }}
  .stack {{ display:flex; flex-direction:column; justify-content:flex-end;
    height:100%; gap:2px; }}
  .seg {{ border-radius: 2px; min-height:2px; }}
  .col-total {{ font-size:11px; color: var(--muted); text-align:center; padding-top:4px;
    font-variant-numeric: tabular-nums; }}
  .col-label {{ font-size:11px; color: var(--muted); text-align:center; }}
  .legend {{ display:flex; gap:14px; flex-wrap:wrap; margin-top:14px;
    color: var(--text-secondary); font-size:13px; }}
  .key {{ display:inline-flex; align-items:center; gap:6px; }}
  .key i {{ width:10px; height:10px; border-radius:3px; display:inline-block; }}
  .wins {{ display:grid; grid-template-columns: repeat(auto-fit,minmax(300px,1fr)); gap:12px; }}
  .win {{ border:1px solid var(--border); border-radius:10px; padding:14px; }}
  .win header {{ display:flex; align-items:center; gap:8px; }}
  .win h3 {{ font-size:14px; margin:0; }}
  .dot {{ width:10px; height:10px; border-radius:3px; flex:0 0 auto; }}
  .win-stat {{ color: var(--text-secondary); margin:8px 0; font-size:13px; }}
  .win ul {{ margin:0; padding-left:18px; color: var(--text-secondary); font-size:13px; }}
  table {{ width:100%; border-collapse: collapse; font-size:13px; }}
  th {{ text-align:left; color: var(--muted); font-weight:500; padding:8px 10px;
    border-bottom:1px solid var(--axis); white-space:nowrap; }}
  td {{ padding:8px 10px; border-bottom:1px solid var(--grid); }}
  td.num {{ text-align:right; font-variant-numeric: tabular-nums; white-space:nowrap; }}
  td.lbl {{ max-width: 360px; }}
  td.own {{ font-size:12px; }}
  td.own.engineering {{ color: var(--critical); }}
  td.own.customer {{ color: var(--text-secondary); }}
  .empty {{ color: var(--muted); }}
  footer {{ color: var(--muted); font-size:12px; margin-top:24px; }}
  @media (max-width: 720px) {{
    .bar-row {{ grid-template-columns: 1fr; gap: 2px; }}
    .bar-meta {{ text-align:left; }}
  }}
</style></head>
<body class="viz-root"><div class="wrap">

<header class="top">
  <div>
    <h1>Renewal errors</h1>
    <p class="sub">{_esc(digest.window_start)} → {_esc(digest.window_end)} ·
      generated {_esc(digest.generated_at)}</p>
  </div>
  <button class="theme" onclick="var r=document.documentElement;
    r.dataset.theme = r.dataset.theme === 'dark' ? 'light' : 'dark';">Toggle theme</button>
</header>

<div class="tiles">
  <div class="tile"><div class="v">{digest.total_events:,}</div><div class="k">errors</div></div>
  <div class="tile"><div class="v">{digest.distinct_incidents}</div>
    <div class="k">distinct incidents<br><span style="color:var(--muted)">from {len(digest.issues)} issue lines</span></div></div>
  <div class="tile"><div class="v">{digest.total_quotes:,}</div><div class="k">quotes affected</div></div>
  <div class="tile"><div class="v">{eng_pct:.0f}%</div>
    <div class="k">engineering-actionable</div></div>
</div>

<section>
  <h2>Who owns the failures</h2>
  <p class="hint">Customer payment declines are not defects — no amount of engineering
     removes them. Everything else recurs until someone changes code or data.</p>
  <div class="split">
    <div style="width:{max(eng_pct, 4):.1f}%;background:var(--cat-data)">{eng_pct:.0f}%</div>
    <div style="width:{max(cust_pct, 4):.1f}%;background:var(--cat-payment)">{cust_pct:.0f}%</div>
    <div style="width:{max(100 - eng_pct - cust_pct, 0):.1f}%;background:{MUTED}"></div>
  </div>
  <div class="split-key">
    <span class="key"><i style="background:var(--cat-data)"></i>
      engineering-actionable — {digest.engineering_events:,}</span>
    <span class="key"><i style="background:var(--cat-payment)"></i>
      customer payment — {digest.customer_events:,}</span>
    <span class="key"><i style="background:{MUTED}"></i>unclassified</span>
  </div>
</section>

<section>
  <h2>Worth fixing first</h2>
  <p class="hint">Engineering-owned, at least 10 occurrences, and meeting two or more
     cheap-to-fix signals. The signals are listed so you can disagree with them.</p>
  <div class="wins">
{_quick_wins(digest)}
  </div>
</section>

<section>
  <h2>Same incident, logged twice</h2>
  <p class="hint">Issues that keep appearing under the same <code>correlationId</code>.
     One failure often emits several log lines, and counting them separately hides
     that fixing one removes both. Ordering is from the timestamps in each trace.</p>
  <div class="wins">
{_links(digest)}
  </div>
</section>

<section>
  <h2>Every issue by volume</h2>
  <p class="hint">Top {min(15, len(digest.issues))} of {len(digest.issues)}. Colour is category; length is count.</p>
{_bar_chart(digest.issues)}
  <div class="legend">{_legend(digest)}</div>
</section>

<section>
  <h2>Weekly volume by category</h2>
  <p class="hint">ISO week. Hover a segment for its category and count.</p>
  <div class="weekly">
{_weekly_chart(digest)}
  </div>
  <div class="legend">{_legend(digest)}</div>
</section>

<section>
  <h2>Full breakdown</h2>
  <p class="hint">Every cluster in the window.</p>
  <div style="overflow-x:auto">
  <table>
    <thead><tr>
      <th>Issue</th><th>Category</th><th>Severity</th><th>Owner</th>
      <th style="text-align:right">Errors</th><th style="text-align:right">% of total</th>
      <th style="text-align:right">Quotes</th>
      <th style="text-align:right">Per day</th><th style="text-align:right">Active days</th>
      <th style="text-align:right">First</th><th style="text-align:right">Last</th>
    </tr></thead>
    <tbody>
{_table(digest)}
    </tbody>
  </table>
  </div>
</section>

<footer>
  Every figure is computed in SQL from the events table. Categories come from the
  renewal service's own <code>error.type</code> where it publishes one, otherwise from
  the labelling step. The engineering/customer split is a triage aid, not a verdict —
  a spike in declines can still mean a broken payment integration.
</footer>

</div></body></html>
"""


def render_slack(digest: Digest) -> str:
    """A short text version for posting alongside the link."""
    lines = [
        f"*Renewal errors — {digest.window_start} to {digest.window_end}*",
        f"{digest.total_events:,} errors · {len(digest.issues)} issue lines "
        f"({digest.distinct_incidents} distinct incidents) · "
        f"{digest.total_quotes:,} quotes affected",
        f"*{digest.engineering_share:.0f}%* are engineering-actionable "
        f"({digest.engineering_events:,}); {digest.customer_events:,} are customer payment declines.",
        "",
        "*Worth fixing first*",
    ]
    for issue in digest.quick_wins:
        lines.append(
            f"• *{issue.events:,}×* ({issue.share_pct:.1f}% of all errors) {issue.label} "
            f"— {issue.quotes:,} quotes, {issue.active_days} active days"
        )
        for reason in issue.quick_win_reasons[:2]:
            lines.append(f"    ◦ {reason}")
    if digest.links:
        lines += ["", "*Same incident, logged twice*"]
        for link in digest.links[:4]:
            lines.append(
                f"• {link.first_label} → {link.second_label} — {link.shared:,} shared "
                f"correlation IDs ({link.containment * 100:.0f}% of the smaller issue)"
            )
    lines += ["", "*Highest volume*"]
    for issue in digest.issues[:8]:
        owner = "eng" if issue.is_engineering else "customer"
        lines.append(
            f"• {issue.events:,}× ({issue.share_pct:.1f}%) {issue.label}  ({issue.category}, {owner})"
        )
    return "\n".join(lines)


def _row_factory(conn: sqlite3.Connection) -> None:  # pragma: no cover - convenience
    conn.row_factory = sqlite3.Row
