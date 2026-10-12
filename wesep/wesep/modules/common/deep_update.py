# config_utils/deep_update.py

from copy import deepcopy
from typing import Dict, Any


class DeepDictUpdater:

    def __init__(self, inplace: bool = True):
        self.inplace = inplace

    def update(
        self,
        default: Dict[str, Any],
        user: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not self.inplace:
            default = deepcopy(default)
        return self._deep_update(default, user)

    def _deep_update(self, default, user):
        for k, v in user.items():
            if (k in default and isinstance(default[k], dict)
                    and isinstance(v, dict)):
                self._deep_update(default[k], v)
            else:
                default[k] = v
        return default


def deep_update(default, user, inplace: bool = True):
    """
    Functional wrapper for DeepDictUpdater
    """
    return DeepDictUpdater(inplace=inplace).update(default, user)
