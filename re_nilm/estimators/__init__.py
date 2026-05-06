"""Estimators package.

Registers legacy module aliases so older pickles saved when the disaggregator
classes lived in ``model/ac_disaggregation.py`` and
``hp_model/disaggregation_functions.py`` still unpickle without those legacy
files on disk. New pickles save the class with its current
``re_nilm.estimators._ac_disagg_v1`` / ``_hp_disagg_v1`` qualifier and are not
affected by these aliases.
"""

from __future__ import annotations

import sys

from re_nilm.estimators import _ac_disagg_v1, _hp_disagg_v1

sys.modules.setdefault("ac_disaggregation", _ac_disagg_v1)
sys.modules.setdefault("disaggregation_functions", _hp_disagg_v1)
