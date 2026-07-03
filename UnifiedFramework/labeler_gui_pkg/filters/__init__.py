from labeler_gui_pkg.filters.base import FILTERS, FILTER_PARAMS, FilterResult, register

# Importing registers each filter into FILTERS via the @register decorator.
from labeler_gui_pkg.filters import crossover_filter  # noqa: F401
