"""Command line interface."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import click

from .blocks import build_fallback_text, build_report_blocks, build_thread_text, chunk_text
from .config import load_settings
from .models import DailyReport
from .pipeline import Pipeline
from .slack import SlackClient, harvest_resolutions, window_bounds
from .store import Store


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Debug logging.")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """Renewals error intelligence — semantic clustering of renewal-flow errors."""
    _setup_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


def _print_report(report: DailyReport) -> None:
    click.echo(f"\n=== Renewal errors — {report.run_date} ===")
    click.echo(
        f"{report.events_extracted} errors · {len(report.clusters)} issue types · "
        f"{report.total_unique_quotes} quotes · {report.unique_templates} templates "
        f"· {report.messages_scanned} messages scanned\n"
    )
    for entry in report.clusters:
        flag = " [NEW]" if entry.is_new else ""
        click.echo(
            f"{entry.event_count:5d} x {entry.share_pct:5.1f}%  {entry.cluster.label}{flag}"
            f"   ({entry.cluster.category}/{entry.cluster.severity}, "
            f"{entry.unique_quote_count} quotes, cluster #{entry.cluster.cluster_id})"
        )
        if entry.cluster.root_cause:
            click.echo(f"        cause: {entry.cluster.root_cause}")
        for template in entry.templates[:2]:
            click.echo(f"        tmpl: {template[:150]}")
    click.echo()


def _post(settings, report: DailyReport) -> None:
    blocks = build_report_blocks(
        report,
        settings.mention_line,
        settings.max_quotes_inline,
        settings.max_clusters_in_message,
    )
    fallback = build_fallback_text(report, settings.mention_line)

    if settings.dry_run or not settings.slack_bot_token.get_secret_value():
        click.echo("--- dry run: Block Kit payload ---")
        click.echo(json.dumps({"text": fallback, "blocks": blocks}, indent=2))
        click.echo("--- dry run: thread reply ---")
        click.echo(build_thread_text(report))
        return

    with SlackClient(settings.slack_bot_token.get_secret_value()) as slack:
        parent_ts = slack.post_message(settings.dest_channel_id, fallback, blocks)
        for chunk in chunk_text(build_thread_text(report)):
            slack.post_message(settings.dest_channel_id, chunk, thread_ts=parent_ts)
    click.echo(f"posted to {settings.dest_channel_id}")


@cli.command()
@click.option("--days", type=int, default=None, help="Lookback window in days.")
@click.option(
    "--from-file", type=click.Path(exists=True), help="Read messages from JSON instead of Slack."
)
@click.option("--dry-run", is_flag=True, help="Print the payload instead of posting.")
@click.option("--offline", is_flag=True, help="No network at all: local encoder and rule labeller.")
@click.option("--db", type=str, default=None, help="Override the database path.")
def run(
    days: int | None, from_file: str | None, dry_run: bool, offline: bool, db: str | None
) -> None:
    """Fetch the last window, cluster it, and post the report."""
    overrides = {}
    if dry_run:
        overrides["dry_run"] = True
    if offline:
        overrides["offline"] = True
    if db:
        overrides["db_path"] = db
    settings = load_settings(**overrides)
    lookback = days or settings.lookback_days
    start, end = window_bounds(lookback)

    if from_file:
        payload = json.loads(Path(from_file).read_text())
        messages = payload["messages"] if isinstance(payload, dict) else payload
    else:
        with SlackClient(settings.slack_bot_token.get_secret_value()) as slack:
            messages = slack.fetch_history(settings.source_channel_id, oldest=start.timestamp())

    with Store(settings.db_path) as store:
        report = Pipeline(settings, store).run(messages, start, end)
        _print_report(report)
        _post(settings, report)


@cli.command()
@click.option(
    "--days", type=int, default=None, help="Limit the backfill window; default is everything."
)
@click.option("--from-file", type=click.Path(exists=True), help="Backfill from a JSON dump.")
@click.option("--offline", is_flag=True)
@click.option("--db", type=str, default=None)
@click.option(
    "--chunk",
    type=int,
    default=5000,
    show_default=True,
    help="Messages per committed chunk. Smaller loses less work if interrupted.",
)
def backfill(
    days: int | None, from_file: str | None, offline: bool, db: str | None, chunk: int
) -> None:
    """Build the historical corpus. Never posts to Slack.

    Run this once before the first scheduled run so that day one already knows
    what is normal and what is genuinely new.
    """
    overrides = {"dry_run": True}
    if offline:
        overrides["offline"] = True
    if db:
        overrides["db_path"] = db
    settings = load_settings(**overrides)

    if from_file:
        payload = json.loads(Path(from_file).read_text())
        messages = payload["messages"] if isinstance(payload, dict) else payload
    else:
        oldest = window_bounds(days)[0].timestamp() if days else None
        with SlackClient(settings.slack_bot_token.get_secret_value()) as slack:
            click.echo("fetching channel history (this can take a while)…")
            messages = slack.fetch_history(settings.source_channel_id, oldest=oldest)

    click.echo(f"{len(messages)} messages to process")
    if not messages:
        return

    # Oldest first, so clusters form in the order the failures actually happened
    # and the earliest occurrence is what sets each cluster's first_seen.
    messages.sort(key=lambda m: float(m.get("ts", 0)))

    # Processed in chunks, each committed before the next begins. A backfill of
    # a full channel history embeds hundreds of templates and can take many
    # minutes; without chunking, one rate-limit or network blip at minute nine
    # throws away every event extracted so far. Re-running skips what is already
    # stored, so an interrupted backfill resumes rather than restarting.
    total = len(messages)
    done = 0
    with Store(settings.db_path) as store:
        pipeline = Pipeline(settings, store)
        for offset in range(0, total, chunk):
            batch = messages[offset : offset + chunk]
            stamps = [float(m["ts"]) for m in batch if m.get("ts")]
            if not stamps:
                continue
            start = datetime.fromtimestamp(min(stamps), tz=UTC)
            end = datetime.fromtimestamp(max(stamps), tz=UTC)
            report = pipeline.run(batch, start, end)
            done += len(batch)
            click.echo(
                f"  {done}/{total} messages  ·  {report.events_extracted} errors this chunk  "
                f"·  {start:%Y-%m-%d} → {end:%Y-%m-%d}"
            )
        _print_report(report)
    click.echo(f"corpus written to {settings.db_path}")


@cli.command()
@click.option("--days", type=int, default=30, help="How far back to scan the report channel.")
@click.option("--db", type=str, default=None)
def harvest(days: int, db: str | None) -> None:
    """Pull human thread replies from the report channel into the RAG corpus."""
    overrides = {"db_path": db} if db else {}
    settings = load_settings(**overrides)
    with SlackClient(settings.slack_bot_token.get_secret_value()) as slack:
        replies = harvest_resolutions(slack, settings.dest_channel_id, since_days=days)

    if not replies:
        click.echo("no human replies found")
        return

    with Store(settings.db_path) as store:
        saved = 0
        for _parent_ts, text, author, reply_ts in replies:
            # Attach the note to the cluster whose label the reply mentions, if
            # any; otherwise store it unattached for a human to triage.
            rows = store.conn.execute("SELECT cluster_id, label FROM clusters").fetchall()
            match = next(
                (
                    r["cluster_id"]
                    for r in rows
                    if r["label"] and r["label"].lower()[:40] in text.lower()
                ),
                None,
            )
            store.add_resolution(match, text, author, reply_ts)
            saved += 1
    click.echo(f"stored {saved} resolution note(s)")


@cli.command("dump-slack")
@click.option("--days", type=int, default=7)
@click.option("--out", type=str, default="fixtures/slack_dump.json")
@click.option("--channel", type=str, default=None)
def dump_slack(days: int, out: str, channel: str | None) -> None:
    """Save raw Slack messages to a file, for offline development."""
    settings = load_settings()
    target = channel or settings.source_channel_id
    with SlackClient(settings.slack_bot_token.get_secret_value()) as slack:
        messages = slack.fetch_history(target, oldest=window_bounds(days)[0].timestamp())
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"messages": messages}, indent=2))
    click.echo(f"wrote {len(messages)} messages to {path}")


@cli.command()
@click.option("--days", type=int, default=None, help="Window in days; default is everything.")
@click.option("--out", type=str, default="renewal-error-report.html", show_default=True)
@click.option("--slack", is_flag=True, help="Also post the summary to the report channel.")
@click.option("--db", type=str, default=None)
def digest(days: int | None, out: str, slack: bool, db: str | None) -> None:
    """Consolidated historical report over the whole corpus.

    The daily post answers "what happened yesterday". This answers "what is
    actually wrong with renewals, and which of it is worth fixing first" — with
    engineering-owned defects separated from customer payment declines, because
    only one of those is a work queue.
    """
    from .digest import build_digest, render_html, render_slack

    settings = load_settings(**({"db_path": db} if db else {}))
    with Store(settings.db_path) as store:
        report = build_digest(store, days)

    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(report))
    click.echo(
        f"{report.total_events:,} errors · {len(report.issues)} issues · "
        f"{report.engineering_share:.0f}% engineering-actionable"
    )
    click.echo(f"wrote {path}")

    if slack:
        if not settings.slack_bot_token.get_secret_value():
            raise SystemExit("REI_SLACK_BOT_TOKEN is not set")
        text = render_slack(report)
        with SlackClient(settings.slack_bot_token.get_secret_value()) as client:
            parent = client.post_message(settings.dest_channel_id, text)
            for extra in chunk_text(text)[1:]:
                client.post_message(settings.dest_channel_id, extra, thread_ts=parent)
        click.echo(f"posted summary to {settings.dest_channel_id}")
    else:
        click.echo("(add --slack to post the summary; the HTML is local only)")


@cli.command()
@click.option("--db", type=str, default=None)
def stats(db: str | None) -> None:
    """Summarise what is in the corpus."""
    settings = load_settings(**({"db_path": db} if db else {}))
    with Store(settings.db_path) as store:
        conn = store.conn
        totals = conn.execute(
            "SELECT (SELECT COUNT(*) FROM events) AS events, "
            "(SELECT COUNT(*) FROM templates) AS templates, "
            "(SELECT COUNT(*) FROM clusters) AS clusters, "
            "(SELECT COUNT(*) FROM embeddings) AS embeddings, "
            "(SELECT COUNT(*) FROM resolutions) AS resolutions, "
            "(SELECT COUNT(*) FROM runs) AS runs"
        ).fetchone()
        click.echo(
            f"events={totals['events']} templates={totals['templates']} "
            f"clusters={totals['clusters']} embeddings={totals['embeddings']} "
            f"resolutions={totals['resolutions']} runs={totals['runs']}"
        )
        click.echo("\nTop clusters all-time:")
        for row in conn.execute(
            "SELECT cluster_id, label, category, total_events, template_count "
            "FROM clusters ORDER BY total_events DESC LIMIT 15"
        ):
            click.echo(
                f"  #{row['cluster_id']:<4} {row['total_events']:>6} events  "
                f"{row['template_count']:>3} tmpl  [{row['category']}]  {row['label'][:70]}"
            )


@cli.command("tune-threshold")
@click.option(
    "--offline/--online",
    default=True,
    help="--online uses the real Voyage encoder and needs REI_VOYAGE_API_KEY.",
)
def tune_threshold(offline: bool) -> None:
    """Sweep the clustering threshold against the labelled fixture set.

    Pick a value in the middle of the widest high-V-measure plateau rather than
    the single best point — a threshold sitting on a cliff edge will not survive
    contact with new wording.
    """
    from evals.run_eval import sweep

    sweep(offline=offline)


@cli.command("eval")
@click.option("--threshold", type=float, default=None)
def eval_cmd(threshold: float | None) -> None:
    """Score the pipeline against the baseline on the labelled fixture set."""
    from evals.run_eval import run as run_eval

    run_eval(threshold)


if __name__ == "__main__":
    cli()
