"""Keep tests away from the developer's Sovara sign-in, settings and saved state."""

import atexit
import os
import shutil
import tempfile

# Set before anything imports sovara, which fixes its home folder at import time.
_HOME = tempfile.mkdtemp(prefix="sovara-langchain-tests-")
atexit.register(shutil.rmtree, _HOME, ignore_errors=True)
os.environ["HOME"] = os.environ["USERPROFILE"] = _HOME
for _name in [n for n in os.environ if n.startswith("SOVARA_") and n != "SOVARA_SOURCE_DIR"]:
    del os.environ[_name]
