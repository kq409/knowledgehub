"""Suite-wide defaults.

Write tools ask for approval in production. Tests that are not about that
flow keep the previous behaviour so they do not hang waiting for a human.
"""

import os

os.environ.setdefault("ASK_APPROVAL_MODE", "off")
