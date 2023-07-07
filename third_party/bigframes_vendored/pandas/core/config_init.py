# Contains code from https://github.com/pandas-dev/pandas/blob/main/pandas/core/config_init.py
"""
This module is imported from the pandas package __init__.py file
in order to ensure that the core.config options registered here will
be available as soon as the user loads the package. if register_option
is invoked inside specific modules, they will not be registered until that
module is imported, which may or may not be a problem.

If you need to make sure options are available even before a certain
module is imported, register them here rather than in the module.

"""
from __future__ import annotations

display_options_doc = """
Encapsulates configuration for displaying objects.

Attributes:
    max_columns (int, default 20):
        If `max_columns` is exceeded, switch to truncate view.
    max_rows (int, default 25):
        If `max_rows` is exceeded, switch to truncate view.
    progress_bar (Optional(str), default "auto"):
        Determines if progress bars are shown during job runs.
        Valid values are `auto`, `notebook`, and `terminal`. Set
        to `None` to remove progress bars.
    repr_mode (Literal[`head`, `deferred`]):
        `head`:
            Execute, download, and display results (limited to head) from
            dataframe and series objects during repr.
        `deferred`:
            Prevent executions from repr statements in dataframe and series objects.
            Instead estimated bytes processed will be shown. Dataframe and Series
            objects can still be computed with methods that explicitly execute and
            download results.
"""
