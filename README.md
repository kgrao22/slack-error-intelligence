# Slack Error Intelligence

Turns the raw error firehose in `#error-alerts` into a daily report of *distinct
failures with honest frequencies*, posted to `#error-digest`.

Replaces a 176-line script that grouped errors by exact string match, ran by hand
from a laptop, and had a hardcoded Slack token.

```
Slack #error-alerts
      │
      ├─ extract ······· structured ErrorEvent, quote ID attributed by key path
      ├─ mask ·········· volatile tokens → <UUID> <EMAIL> <STRIPE_ID> <NUM>
      ├─ enrich ········ generic transport errors get service + deeper message
      ├─ embed ········· Voyage voyage-3-lite, content-hash cached
      ├─ cluster ······· cosine search vs persistent centroids, τ = 0.76
      ├─ label ········· Claude names new clusters, with RAG over past clusters
      │                  and human resolutions. Never touches a number.
      └─ report ········ Block Kit summary + threaded detail
                              │
                     SQLite on the `data` branch
                     (history · embedding cache · centroids · resolutions)
```

Runs daily on GitHub Actions. Marginal cost is a fraction of a cent per day.

---

## Why the old grouping was wrong

Three failures, all observed on a real production error channel.

**1. Identifiers inside the message text fragmented every group.**

```
1 x Renewal Flow: Failed to get quote docs for: 8afd54b3-e236-4dcf-806e-3024680a3c9b
```

The quote ID is part of the string, so every occurrence hashes differently and is
reported as its own `1 x` line. A failure hitting fourteen quotes shows up
fourteen times, each looking negligible.

**2. One generic string hid several unrelated failures.**

```
12 x Request failed with status code 400
```

Every upstream in the renewal path emits that string. The line says twelve
somethings went wrong and identifies none of them.

**3. Counts and quote IDs measured different things.**

```
4 x Likely validation error || response-validation
   -> Quotes: [30 UUIDs]
```

`4 x` counted messages. The quote list was the union of *every UUID anywhere in
those payloads* — correlation IDs, policy IDs, document IDs — presented as
affected quotes. The "Likely validation error" label was invented by a rule that
relabelled any combined message under 20 characters, so unrelated validators from
different services all landed in one bucket.

---

## What replaces it

**Masking before matching.** Volatile tokens become placeholders, so the fourteen
quote-docs failures collapse into one template with a count of 14. Numbers are
handled asymmetrically on purpose: integers of one to three digits survive, so
`status code 400` and `status code 500` never merge, while amounts and long IDs
are masked.

**Enrichment of generic errors.** A template that is pure transport noise —
`Request failed with status code 400`, `Bad Request`, `invalid` — is enriched with
the emitting service and the next-most-specific message in the payload before it
is embedded. That single line of 12 splits into the seven quote-docs failures and
the five pricing-engine failures that were actually happening.

**Attributed quote IDs.** The quote is read from a named key (`quoteId`,
`quote_id`, …). Everything else is recorded as `related_ids` and never counted as
an affected quote. When the quote genuinely cannot be determined, the report says
so instead of guessing.

**Semantic clustering with stable identity.** Templates are embedded and assigned
to persistent clusters by cosine similarity against stored centroids. Cluster IDs
survive across days, which is what makes "is this new?" and week-over-week
comparison mean anything. Two different wordings of the same SQL timeout, emitted
by two different services, land in one cluster.

**An LLM that interprets but never counts.** Claude names and categorises
*new* clusters only, with the nearest historical clusters and any recorded human
resolutions retrieved as context. Every number in the report is computed in SQL
from the events table, after clustering has settled. A report whose figures
cannot be trusted is worse than no report.

---

## Does it actually work

`make eval` scores the pipeline against a faithful port of the original script on
196 labelled messages — 156 errors across 12 known failure types, plus 40
endorsement documents that share the channel and must be filtered rather than
counted. Each row adds one stage, so the gain is attributable rather than
hand-waved at "we added AI".

```
variant                       homog. complet.       V     ARI   #pred   #true
------------------------------------------------------------------------------
1 baseline (exact string)      0.903    0.897   0.900   0.880      22      12
2  + token masking             0.903    0.966   0.934   0.891      11      12
3  + context enrichment        1.000    0.970   0.985   0.979      13      12
4  + lexical vectors           1.000    1.000   1.000   1.000      12      12

variant                      precision  recall      F1   claimed  correct
------------------------------------------------------------------------
1 baseline (exact string)        0.500   1.000   0.667       312      156
2  + token masking               1.000   1.000   1.000       156      156
```

Read that as: the old script reported **22 groups for 12 real failures** and
claimed **312 affected quotes when there were 156**. Most of the clustering gain
comes from masking and enrichment, which involve no model at all; vectors close
the last gap by merging the two timeout wordings. Being able to say which stage
earned what is the point of running the ablation.

Measured against a real seven-day dump before this rewrite, the original script
was in worse shape than the table suggests: it parsed **20% of messages**,
attributed **1 quote out of 125 errors**, and swept 40% of all errors into two
meaningless buckets ("Request failed with status code 400" and
"response-validation").

Metrics are homogeneity (are unrelated failures merged?), completeness (is one
failure split across lines?), V-measure, and adjusted Rand index, implemented
dependency-free in `evals/metrics.py` and property-checked against scikit-learn in
the test suite.

**Caveat, stated plainly.** The fixture set is synthetic, but it is generated in
the real wire format — Slack rich-text `label ``` value` pairs with `methodName`,
`correlationId` and a structured `error` block — and every failure shape in it was
observed in an actual seven-day dump of the channel. Arms 1–4 need no API key, so CI can
gate on them. Arm 5 scores real Voyage embeddings and is skipped unless
`REI_VOYAGE_API_KEY` is set. To score real traffic:

```bash
rei dump-slack --days 30 --out fixtures/real.json   # never commit this file
```

Label a sample by hand and re-point the eval at it.

The clustering threshold was picked from a sweep, not guessed:

```
tau    homog.  complet.       V  #clusters
0.68    0.951     1.000   0.975         11   ← unrelated failures merging
0.70    1.000     1.000   1.000         12
0.76    1.000     1.000   1.000         12   ← default: middle of the plateau
0.82    1.000     1.000   1.000         12
0.84    1.000     0.970   0.985         13   ← one failure splitting in two
```

`rei tune-threshold --online` re-runs this against Voyage. A different encoder has
a different similarity distribution, so **re-tune before trusting the default in
production**. Pick the middle of the widest plateau, not the single best point — a
threshold on a cliff edge will not survive new wording.

---

## Setup

```bash
git clone <repo> && cd slack-error-intelligence
make install
cp .env.example .env      # fill in tokens
make demo                 # offline dry run, no keys, no network
```

Real run against Slack, printing instead of posting:

```bash
rei run --days 1 --dry-run
```

Build the historical corpus once before the first scheduled run, so day one
already knows what is normal:

```bash
rei backfill              # everything the channel retains; never posts
rei stats
```

Then go live:

```bash
rei run --days 1
```

### GitHub Actions

Repository **secrets**: `REI_SLACK_BOT_TOKEN`, `REI_ANTHROPIC_API_KEY`,
`REI_VOYAGE_API_KEY`.
Repository **variables**: `REI_SOURCE_CHANNEL_ID`, `REI_DEST_CHANNEL_ID`,
`REI_MENTION_USER_IDS`.

`.github/workflows/daily.yml` runs at 01:00 UTC (06:30 IST) and force-pushes the
SQLite file to an orphan `data` branch as a single squashed commit. The database
is a binary blob that changes daily; keeping its git history would grow the repo
without bound, and the history that matters lives in the database's own tables.
Push the seeded database once after backfilling:

```bash
git checkout --orphan data && git rm -rf . && cp data/rei.db rei.db
git add rei.db && git commit -m "state: seed" && git push -u origin data
git checkout main
```

`workflow_dispatch` accepts a lookback window and a dry-run toggle, so the whole
thing can be exercised from the Actions tab before the first scheduled fire.

---

## Commands

| Command | What it does |
| --- | --- |
| `rei run` | Fetch the window, cluster, post the report |
| `rei backfill` | Build the corpus from channel history. Never posts |
| `rei harvest` | Pull human thread replies into the RAG corpus |
| `rei dump-slack` | Save raw messages to JSON for offline work |
| `rei stats` | Summarise events, templates, clusters, resolutions |
| `rei eval` | Score against the old script |
| `rei tune-threshold` | Sweep the similarity threshold |

Every command takes `--db` to point at an alternate database. `run` and
`backfill` take `--offline` (no network at all) and `--dry-run` (print, never
post).

---

## Cost

| | per run | per year |
| --- | --- | --- |
| Voyage embeddings | ~0 (cache hit) to $0.0001 | < $0.05 |
| Claude labelling | $0 on a quiet day, ~$0.002 with new clusters | < $1 |
| GitHub Actions | free tier | $0 |

Templates are embedded once ever and cached by content hash, so a steady-state day
embeds nothing. Claude is called only for clusters created that day. The initial
backfill is the only meaningful spend, and it is cents.

---

## Design notes

**Why incremental clustering rather than k-means or HDBSCAN.** Batch clustering
produces better-shaped clusters on a fixed corpus and no identity continuity at
all. If today's cluster 3 is not yesterday's cluster 3, "new", "trending" and the
resolution corpus all become meaningless. Continuity is worth more here than
cluster shape.

**Why exact vector search and not a vector index.** The corpus is a few thousand
distinct templates across all of history. A `numpy` matmul over the centroid
matrix is sub-millisecond and exact; an ANN index would add a dependency, a build
step and approximation error to save nothing. Search is isolated behind
`Store.search_clusters`, so swapping in `sqlite-vec` when the corpus reaches
~10⁵ vectors is a one-function change.

**Why raw HTTP instead of three SDKs.** Slack, Voyage and Anthropic are all JSON
over HTTPS and fail the same three ways. `src/rei/http.py` gives all three one
retry, backoff, jitter and `Retry-After` policy, written once and tested once.
Total runtime dependencies: httpx, pydantic, numpy, click.

**Slack rate limits.** Since May 2025, `conversations.history` is capped at 1
request per minute and 15 objects per page for non-Marketplace *distributed*
apps. Internal apps built inside your own workspace — which this is — keep Tier 3
and 1,000 objects per page. The client does not assume either: it requests a large
page, observes what it gets back, and paces accordingly, so a full backfill
completes under either regime.

**Degradation.** No Voyage key falls back to a local hashed n-gram encoder. No
Anthropic key falls back to rule-based labels. A Claude API failure mid-run is
caught and degrades to rules rather than losing the day's report. The pipeline
always produces something.

---

## Security

The script this replaces had a live `xoxb-` bot token hardcoded on line 8. **That
token is compromised and must be rotated** — assume anything it could read has
been read. Rotate it at api.slack.com → your app → OAuth & Permissions → Reload,
then put the new value in `.env` locally and in GitHub repository secrets. If the
old script was ever committed, the token is in git history and rotation is the
only fix; removing the file does not help.

Raw Slack dumps contain customer emails and Stripe customer IDs and are
git-ignored. The database stores masked templates alongside raw excerpts, so treat
`rei.db` as containing customer data and keep the `data` branch in a private repo.

---

## Layout

```
src/rei/
  config.py       env-driven settings, no secrets on disk
  models.py       ErrorEvent, Cluster, ClusterReport, DailyReport
  http.py         shared retry / backoff / Retry-After for all three providers
  slack.py        paged history, thread harvesting, posting
  extract.py      raw message → structured ErrorEvent
  masking.py      volatile-token masking, generic-message detection
  embeddings.py   Voyage client + content-hash cache + offline encoder
  store.py        SQLite schema, embedding cache, centroids, resolutions
  clustering.py   incremental assignment against persistent centroids
  rag.py          retrieval over historical clusters and human resolutions
  llm.py          Claude labelling via forced tool use, rule-based fallback
  pipeline.py     orchestration; all counting happens here
  blocks.py       Block Kit rendering within Slack's limits
  baseline.py     the original algorithm, kept only for the eval
  cli.py          click entry points
scripts/
  generate_fixtures.py   labelled fixtures in the real wire format
  inspect_payloads.py    masked diagnostic report over a real dump
evals/            metrics + ablation harness
tests/            48 tests, no network required
```
