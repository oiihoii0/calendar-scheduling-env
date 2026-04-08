"""
Upload project files to HuggingFace Space using the API.
This avoids all git binary file issues.
Run: python upload_to_hf.py
"""
import os
from huggingface_hub import HfApi

# ── FILL THESE IN ────────────────────────────────
HF_TOKEN   = os.getenv("HF_TOKEN", "")  # Set via environment variable or paste temporarily
SPACE_REPO = "oiihoii0/calendar-scheduling-env"   # already correct
# ────────────────────────────────────────────────

# Files/folders to skip (binaries, cache, etc.)
IGNORE = [
    "*.png", "*.jpg", "*.zip",
    "trained_models/*",
    "charts/*",
    "__pycache__",
    "*.pyc",
    ".git",
    ".pytest_cache",
    "*.egg-info",
    "upload_to_hf.py",   # don't upload this script itself
]

api = HfApi()

print(f"Uploading to: {SPACE_REPO}")
print("This may take a minute...")

api.upload_folder(
    folder_path=".",
    repo_id=SPACE_REPO,
    repo_type="space",
    token=HF_TOKEN,
    ignore_patterns=IGNORE,
)

print("\nDone! Go to:")
print(f"  https://huggingface.co/spaces/{SPACE_REPO}")
print("Watch the 'Building' badge — it should turn green in ~3-5 minutes.")
