"""Internal types, enums, and constants."""

from enum import Enum


class Unit(Enum):
    BPS = "bps"
    PRICE = "price"


class HorizonType(Enum):
    WALL = "wall"
    TRADE = "trade"
    TICK = "tick"


# Reserved column names on the trades DataFrame
REQUIRED_TRADE_COLS = frozenset({"timestamp", "side", "price", "mid"})
OPTIONAL_TRADE_COLS = frozenset({"size"})
RESERVED_TRADE_COLS = REQUIRED_TRADE_COLS | OPTIONAL_TRADE_COLS

# Reserved column names on the quotes DataFrame
REQUIRED_QUOTE_COLS = frozenset({"timestamp", "mid"})

# Output column names added by compute()
OUTPUT_COLS = frozenset({"horizon_type", "horizon_value", "markout", "future_mid"})
