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
        """Schema for L2 Order Book depth snapshots (8-tier bid/ask from lob_collector or bybit_ingest)."""
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
