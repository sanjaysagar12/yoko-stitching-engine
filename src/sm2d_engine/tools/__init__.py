"""Tool strategies. Importing this package registers every built-in tool in ``default_registry``."""

from . import curves as curves
from . import operations as operations
from . import points as points
from .registry import default_registry as default_registry
