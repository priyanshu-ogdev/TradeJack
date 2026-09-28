"""
Pandera Schema Validation for Trade-Flow Physics, L2 Order Book Depth, and Raw Trade Data.
Ensures that data fed to PyTorch / KvikIO strictly conforms to expected tensor shapes and types.
Protects the GPU VRAM from fragmentation due to corrupted data.

Gracefully degrades if pandera is not installed — validation is skipped with a warning.
"""

import logging

logger = logging.getLogger("TradeJackSchema")

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

try:
    import pandera.polars as pa
    PANDERA_AVAILABLE = True
except ImportError:
    PANDERA_AVAILABLE = False
    logger.info("pandera not installed. Schema validation will be skipped (degraded mode).")


if PANDERA_AVAILABLE and POLARS_AVAILABLE:

    class TradeJackPhysicsSchema(pa.DataFrameModel):
        """Schema for processed Trade-Flow Physics Parquet files (feature_engineering output)."""
        timestamp: pl.Datetime
        open_price: pl.Float64 = pa.Field(nullable=False)
        close_price: pl.Float64 = pa.Field(nullable=False)
        volume: pl.Float64 = pa.Field(nullable=False)
        ofi: pl.Float64 = pa.Field(nullable=False)
        # VPIN must be between 0 and 1
        vpin_50: pl.Float64 = pa.Field(ge=0.0, le=1.0, nullable=False)
        kyles_lambda: pl.Float64 = pa.Field(nullable=False)

        class Config:
            strict = False  # Allow intermediate columns during Volume-Gated MAD processing
            coerce = True   # Cast types where possible without losing precision

    class LOBDepthSchema(pa.DataFrameModel):
        """Schema for L2 Order Book depth snapshots (8-tier bid/ask). Currently only produced
        by lob_collector.py (live Binance WebSocket depth). There is no free bulk historical
        L2 source wired up — see bybit_ingest.py docstring for why Bybit can't supply this."""
        timestamp: pl.Float64 = pa.Field(nullable=False)
        bid_px_0: pl.Float64 = pa.Field(nullable=False)
        bid_sz_0: pl.Float64 = pa.Field(nullable=False)
        ask_px_0: pl.Float64 = pa.Field(nullable=False)
        ask_sz_0: pl.Float64 = pa.Field(nullable=False)
        bid_px_1: pl.Float64 = pa.Field(nullable=True)
        bid_sz_1: pl.Float64 = pa.Field(nullable=True)
        ask_px_1: pl.Float64 = pa.Field(nullable=True)
        ask_sz_1: pl.Float64 = pa.Field(nullable=True)

        class Config:
            strict = False  # Allow additional depth tiers (levels 2-7) beyond the required 2
            coerce = True

    class RawTradeSchema(pa.DataFrameModel):
        """Schema for raw aggTrades data before physics processing."""
        agg_trade_id: pl.Int64 = pa.Field(nullable=False)
        price: pl.Float64 = pa.Field(gt=0.0, nullable=False)
        quantity: pl.Float64 = pa.Field(gt=0.0, nullable=False)
        first_trade_id: pl.Int64 = pa.Field(nullable=False)
        last_trade_id: pl.Int64 = pa.Field(nullable=False)
        transact_time: pl.Int64 = pa.Field(nullable=False)
        is_buyer_maker: pl.Boolean = pa.Field(nullable=False)

        class Config:
            strict = False
            coerce = True

    class BybitTradeSchema(pa.DataFrameModel):
        """Schema for Bybit's real public trade dump (public.bybit.com/trading/{SYMBOL}/).
        These are executed trade ticks, not order book depth."""
        timestamp: pl.Float64 = pa.Field(nullable=False)
        symbol: pl.Utf8 = pa.Field(nullable=False)
        side: pl.Utf8 = pa.Field(nullable=False)
        size: pl.Float64 = pa.Field(gt=0.0, nullable=False)
        price: pl.Float64 = pa.Field(gt=0.0, nullable=False)

        class Config:
            strict = False
            coerce = True

    class ForexTickSchema(pa.DataFrameModel):
        """Schema for Dukascopy free historical FX tick data (bid/ask quotes, no volume-in-base)."""
        timestamp: pl.Datetime = pa.Field(nullable=False)
        bid: pl.Float64 = pa.Field(gt=0.0, nullable=False)
        ask: pl.Float64 = pa.Field(gt=0.0, nullable=False)
        bid_volume: pl.Float64 = pa.Field(ge=0.0, nullable=False)
        ask_volume: pl.Float64 = pa.Field(ge=0.0, nullable=False)

        class Config:
            strict = False
            coerce = True

    class ForexPhysicsSchema(pa.DataFrameModel):
        """Schema for processed FX microstructure features (forex_feature_engineering.py output).
        Quote-stream analogs of TradeJackPhysicsSchema's trade-flow features."""
        timestamp: pl.Datetime = pa.Field(nullable=False)
        mid_price: pl.Float64 = pa.Field(gt=0.0, nullable=False)
        spread: pl.Float64 = pa.Field(ge=0.0, nullable=False)
        relative_spread: pl.Float64 = pa.Field(ge=0.0, nullable=False)
        quote_imbalance: pl.Float64 = pa.Field(ge=-1.0, le=1.0, nullable=False)
        quote_intensity: pl.Float64 = pa.Field(ge=0.0, nullable=False)
        log_return: pl.Float64 = pa.Field(nullable=True)
        is_gap: pl.Boolean = pa.Field(nullable=False)

        class Config:
            strict = False
            coerce = True

    class SentimentSchema(pa.DataFrameModel):
        """Schema for sentiment_source.py output: aggregated RSS/VADER sentiment per bucket."""
        timestamp: pl.Datetime = pa.Field(nullable=False)
        symbol: pl.Utf8 = pa.Field(nullable=False)
        headline_count: pl.Int64 = pa.Field(ge=0, nullable=False)
        sentiment_mean: pl.Float64 = pa.Field(ge=-1.0, le=1.0, nullable=False)
        sentiment_std: pl.Float64 = pa.Field(ge=0.0, nullable=False)

        class Config:
            strict = False
            coerce = True

    class ForexToxicitySchema(pa.DataFrameModel):
        """Schema for forex_toxicity_engineering.py output: BVC-based VPIN/Kyle's Lambda
        analogs, volume-bucketed (not time-bucketed — see docs/DATA_FORGE_FX_TOXICITY_PLAN.md)."""
        timestamp: pl.Datetime = pa.Field(nullable=False)
        volume: pl.Float64 = pa.Field(ge=0.0, nullable=False)
        buy_volume: pl.Float64 = pa.Field(ge=0.0, nullable=False)
        sell_volume: pl.Float64 = pa.Field(ge=0.0, nullable=False)
        price_change: pl.Float64 = pa.Field(nullable=True)
        bvc_vpin: pl.Float64 = pa.Field(ge=0.0, nullable=True)
        kyles_lambda: pl.Float64 = pa.Field(nullable=True)
        amihud_illiquidity: pl.Float64 = pa.Field(ge=0.0, nullable=True)

        class Config:
            strict = False
            coerce = True

else:
    # Graceful fallback: create no-op schema classes that pass validation unconditionally
    class _NoOpSchema:
        """No-op schema placeholder when pandera is not installed."""
        @classmethod
        def validate(cls, df, *args, **kwargs):
            logger.debug("Schema validation skipped (pandera not installed).")
            return df

    TradeJackPhysicsSchema = _NoOpSchema
    LOBDepthSchema = _NoOpSchema
    RawTradeSchema = _NoOpSchema
    BybitTradeSchema = _NoOpSchema
    ForexTickSchema = _NoOpSchema
    ForexPhysicsSchema = _NoOpSchema
    SentimentSchema = _NoOpSchema
    ForexToxicitySchema = _NoOpSchema
