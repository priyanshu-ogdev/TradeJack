"""
Sentiment Source — free RSS ingestion + VADER scoring, feeding sentiment_oracle.py.

sentiment_oracle.py is a correctly-built Qdrant storage/query layer, but it had zero
producers — no scrapers, feeds, or APIs. This module is the fix, and it's deliberately
modest about what it is:

  - Source: free public RSS feeds (no API key, no auth, no rate-limit tier to pay for).
    This will not match a paid Twitter/X firehose or a licensed news API in volume or
    latency — that trade-off is real, not hidden.
  - Scoring: VADER (vaderSentiment), a free local lexicon-based scorer. No model
    download, no network call at score time, fully reproducible. It is a weaker signal
    than a transformer embedding (SentimentOracle's BGE-Large path is still the
    "real embedding" upgrade path once one is actually wired in) — VADER is an honest,
    working interim signal, not a placeholder pretending to be the final design.

Output: a compact per-(symbol, time-bucket) series — headline_count, sentiment_mean,
sentiment_std — cheap enough to join directly onto price/microstructure features
without needing the vector DB. Raw headline embeddings can still go into
SentimentOracle.ingest_embeddings() separately for similarity search; this module
does not replace that, it just finally gives it (and the RL feature pipeline) real
input instead of nothing.
"""

import os
import logging
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional

try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

from data_forge.config import config
from data_forge.schema import SentimentSchema

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (SentimentSource) %(message)s")
logger = logging.getLogger("SentimentSource")

# Free, no-key, publicly-published RSS feeds. Deliberately general market feeds rather
# than symbol-specific ones — filtering to a symbol happens via keyword match on the
# headline (see _matches_symbol), because none of these feeds offer a per-symbol tier.
DEFAULT_RSS_FEEDS = {
    "crypto": [
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://cointelegraph.com/rss",
    ],
    "forex": [
        "https://www.investing.com/rss/news_1.rss",
    ],
    "macro": [
        "https://feeds.reuters.com/reuters/businessNews",
    ],
}

# Rough symbol -> keyword mapping for filtering general-market headlines. Extend this
# per-symbol as coverage needs grow; it's intentionally simple substring matching rather
# than NER, to keep this module dependency-light.
_SYMBOL_KEYWORDS = {
    "BTC-USDT": ["bitcoin", "btc"],
    "ETH-USDT": ["ethereum", "eth"],
    "SOL-USDT": ["solana", "sol"],
    "EURUSD": ["euro", "eur/usd", "eurusd", "ecb"],
    "GBPUSD": ["pound", "sterling", "gbp/usd", "gbpusd", "boe"],
    "USDJPY": ["yen", "usd/jpy", "usdjpy", "boj"],
}


def _matches_symbol(headline: str, symbol: str) -> bool:
    keywords = _SYMBOL_KEYWORDS.get(symbol.upper())
    if not keywords:
        return True  # no keyword map for this symbol: keep everything (macro-general fallback)
    lowered = headline.lower()
    return any(kw in lowered for kw in keywords)


class SentimentSource:
    """Fetches free RSS headlines, scores them with VADER, and aggregates into
    per-symbol time buckets suitable for joining onto price features."""

    def __init__(self, feeds: Optional[Dict[str, List[str]]] = None):
        self.feeds = feeds or DEFAULT_RSS_FEEDS
        self._analyzer = SentimentIntensityAnalyzer() if VADER_AVAILABLE else None

    def fetch_headlines(self, category: str) -> List[dict]:
        """Fetches and parses one RSS category. Returns [] on any feed/parse failure
        rather than raising — one dead feed should not take down ingestion for others."""
        if not FEEDPARSER_AVAILABLE:
            logger.error("feedparser not installed. Cannot fetch RSS feeds.")
            return []

        headlines = []
        for url in self.feeds.get(category, []):
            try:
                parsed = feedparser.parse(url)
                if parsed.bozo and not parsed.entries:
                    logger.warning(f"Feed parse issue for {url}: {parsed.bozo_exception}")
                    continue
                for entry in parsed.entries:
                    title = getattr(entry, "title", "").strip()
                    if not title:
                        continue
                    published = getattr(entry, "published_parsed", None)
                    ts = (
                        datetime(*published[:6], tzinfo=timezone.utc)
                        if published
                        else datetime.now(timezone.utc)
                    )
                    headlines.append({"timestamp": ts, "headline": title, "source": url})
            except Exception as e:
                logger.warning(f"Failed to fetch/parse {url}: {e}")
        return headlines

    def score_headline(self, headline: str) -> float:
        """Returns VADER's compound score in [-1, 1]. 0.0 (neutral) if VADER unavailable —
        callers should treat a 0.0 with zero headline_count as 'no data', not 'neutral news'."""
        if not self._analyzer:
            return 0.0
        clean = re.sub(r"\s+", " ", headline).strip()
        return self._analyzer.polarity_scores(clean)["compound"]

    def build_symbol_sentiment(self, symbol: str, category: str, bucket_minutes: int = 60) -> "pl.DataFrame":
        """Fetches, filters to `symbol`, scores, and aggregates into time buckets.
        Returns an empty DataFrame (not an error) if no matching headlines were found —
        that's the honest "no signal right now" state, distinct from a fetch failure."""
        if not POLARS_AVAILABLE:
            logger.error("Polars required to build sentiment features.")
            return None

        raw = self.fetch_headlines(category)
        matched = [h for h in raw if _matches_symbol(h["headline"], symbol)]

        if not matched:
            logger.info(f"No matching headlines for {symbol} in category '{category}'.")
            return pl.DataFrame(schema={
                "timestamp": pl.Datetime, "symbol": pl.Utf8, "headline_count": pl.Int64,
                "sentiment_mean": pl.Float64, "sentiment_std": pl.Float64,
            })

        scored = [
            {"timestamp": h["timestamp"], "score": self.score_headline(h["headline"])}
            for h in matched
        ]
        df = pl.DataFrame(scored)
        bucketed = (
            df.sort("timestamp")
            .group_by_dynamic("timestamp", every=f"{bucket_minutes}m")
            .agg(
                pl.count().alias("headline_count"),
                pl.col("score").mean().alias("sentiment_mean"),
                pl.col("score").std().fill_null(0.0).alias("sentiment_std"),
            )
            .with_columns(pl.lit(symbol).alias("symbol"))
            .select(["timestamp", "symbol", "headline_count", "sentiment_mean", "sentiment_std"])
        )

        try:
            SentimentSchema.validate(bucketed)
        except Exception as e:
            logger.error(f"Sentiment features failed schema validation for {symbol}: {e}")
            return None

        return bucketed

    def write_symbol_sentiment(self, symbol: str, category: str, bucket_minutes: int = 60) -> str:
        df = self.build_symbol_sentiment(symbol, category, bucket_minutes)
        if df is None or df.is_empty():
            return ""

        out_dir = os.path.join(config.data_store_dir, "processed", symbol, "sentiment")
        os.makedirs(out_dir, exist_ok=True)
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        out_path = os.path.join(out_dir, f"{symbol}-sentiment-{date_str}.parquet")

        df.write_parquet(
            out_path,
            compression=config.compression_codec,
            compression_level=config.compression_level,
            row_group_size=config.row_group_size,
        )
        logger.info(f"Sentiment features -> {out_path} ({len(df)} buckets, {int(df['headline_count'].sum())} headlines)")
        return out_path


if __name__ == "__main__":
    source = SentimentSource()
    source.write_symbol_sentiment("BTC-USDT", category="crypto")
