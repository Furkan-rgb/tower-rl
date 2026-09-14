"""Compatibility exports for the domain contract.

New code should import from :mod:`tower_rl.domain`; this module remains for the
initial public API while downstream callers migrate.
"""

from tower_rl.domain import *  # noqa: F403
from tower_rl.domain import __all__ as _domain_all

__all__ = _domain_all
