"""Environment-driven configuration.

Nothing secret is ever written to disk or committed. Locally these come from a
.env file (git-ignored); in CI they come from GitHub Actions secrets.
"""

from __future__ import annotations

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="REI_",
        extra="ignore",
    )

    # --- Slack -------------------------------------------------------------
    slack_bot_token: SecretStr = Field(default=SecretStr(""))
    # No defaults: set these for your own workspace in .env or as CI variables.
    source_channel_id: str = Field(default="")  # channel to read, e.g. C0123456789
    dest_channel_id: str = Field(default="")  # channel to post to
    mention_user_ids: list[str] = Field(default_factory=list)  # who to @ on the report

    # --- Model providers ---------------------------------------------------
    anthropic_api_key: SecretStr = Field(default=SecretStr(""))
    # claude-sonnet-5 is the current balanced model ($2/$10 per MTok). Labelling
    # a day's new clusters is a few hundred tokens, so this costs fractions of a
    # cent; swap to claude-haiku-4-5 if you want it cheaper still.
    anthropic_model: str = Field(default="claude-sonnet-5")
    voyage_api_key: SecretStr = Field(default=SecretStr(""))
    voyage_model: str = Field(default="voyage-3-lite")

    # --- Clustering --------------------------------------------------------
    # Cosine similarity above which a new template joins an existing cluster.
    #
    # Two values, because the two encoders have different similarity
    # distributions and a single number cannot serve both. voyage-3-lite packs
    # unrelated production errors into roughly 0.75-0.88, so 0.76 merges things
    # that should stay apart — "Failed to update quote" with "Parent quote found
    # as cancelled", or an insufficient-funds decline with a generic one. The
    # hashed lexical encoder used offline spreads much wider, and 0.76 is the
    # midpoint of its measured [0.70, 0.82] plateau.
    #
    # Re-tune against your own traffic before trusting either:
    #     rei tune-threshold --online
    cluster_threshold: float = Field(default=0.90, ge=0.0, le=1.0)
    offline_cluster_threshold: float = Field(default=0.76, ge=0.0, le=1.0)
    # Templates seen fewer than this many times in all of history are still
    # clustered, but are not allowed to *create* a cluster on their own until
    # they recur. Prevents one-off garbage from fragmenting the taxonomy.
    min_events_for_new_cluster: int = Field(default=1, ge=1)
    embedding_batch_size: int = Field(default=96, ge=1, le=128)

    # --- Behaviour ---------------------------------------------------------
    lookback_days: int = Field(default=1, ge=1)
    # Root-cause / suggested-fix generation. Off by default (v1 scope is
    # clustering + frequency); flip to true to light up the diagnosis section.
    enable_diagnosis: bool = Field(default=False)
    max_quotes_inline: int = Field(default=5, ge=0)
    max_clusters_in_message: int = Field(default=12, ge=1)
    db_path: str = Field(default="data/rei.db")

    # --- Safety ------------------------------------------------------------
    # When true, nothing is posted to Slack; the rendered payload is printed.
    dry_run: bool = Field(default=False)
    # When true, no network calls are made at all. Embeddings fall back to a
    # deterministic local hash encoder and the LLM to a rule-based labeller,
    # so the whole pipeline is testable offline and in CI without secrets.
    offline: bool = Field(default=False)

    @property
    def mention_line(self) -> str:
        return " ".join(f"<@{uid.strip('<>@')}>" for uid in self.mention_user_ids)


def load_settings(**overrides: object) -> Settings:
    return Settings(**overrides)  # type: ignore[arg-type]
