import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
# diffusers >= 0.40 (MiniMax-H3 classes) for the release-parity tests, e.g. a `pip --target` dir.
if os.environ.get("H3_DIFFUSERS_PATH"):
    sys.path.insert(0, os.environ["H3_DIFFUSERS_PATH"])
