import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import j2gauss  # noqa: E402,F401  (turns on jax_enable_x64)
