"""Download model weights into ./models (Gemma is gated: accept its license on HF and `hf auth login` first)."""
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent
for repo, name in [("Qwen/Qwen3-VL-4B-Instruct", "Qwen3-VL-4B-Instruct"), ("google/gemma-3-4b-it", "gemma-3-4b-it"),
                   ("Qwen/Qwen3-8B", "Qwen3-8B")]:  # Qwen3-8B: rule extraction from documents only
    snapshot_download(repo, local_dir=ROOT / "models" / name)
from ultralytics import YOLO
YOLO("yolo11m.pt")  # cached by ultralytics on first use
