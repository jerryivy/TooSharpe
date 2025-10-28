"""
fund_pipeline: Data cleaning and integration toolkit for fund analytics.

Modules:
    - utils: Common helper functions (date parsing, safe file ops)
    - data_handling: Dataset cleaning and integration functions
    - intermediary_builder: Build intermediary datasets for analytics
    - analytics: Analytics functions for performance, liquidity, and attribution
"""

from . import utils
from . import data_handling
from . import intermediary_builder
from . import analytics

__version__ = "0.1.0"
__all__ = ["utils", "data_handling", "intermediary_builder", "analytics"]