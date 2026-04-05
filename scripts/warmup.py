"""Pre-warm both GPU models before the API server starts accepting traffic.

Without this, the first ingest request triggers two expensive cold-start operations
simultaneously:
  1. PP-DocLayoutV3: loads weights + NVRTC JIT-compiles CUDA reduction kernels (~90s)
  2. GLM-OCR (Ollama): loads 2.2 GiB F16 model + 4 GiB KV cache into GPU VRAM (~20s)

Both compete for the same GPU, causing the glmocr SDK's 30s connection timeout to
fire 2–3 times before Ollama can respond — adding ~2 minutes of retries to the first
request.

This script runs once at startup, sequentially:
  - PP-DocLayoutV3 first  → NVRTC kernels compiled and cached in ~/.cache/torch/
  - GLM-OCR second        → model loaded into VRAM; stays there via OLLAMA_KEEP_ALIVE=-1

Subsequent requests find both models hot, eliminating the cold-start latency entirely.
Only runs when PARSER_BACKEND=ollama (no-op for cloud mode).
"""
from __future__ import annotations

import base64
import io
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import httpx
import numpy as np
from loguru import logger
from PIL import Image

from doc_parser.config import get_settings


def _tiny_white_png_b64() -> str:
    """Return a 64×64 white PNG as a base64 string (minimal valid image for GLM-OCR)."""
    img = Image.fromarray(np.ones((64, 64, 3), dtype=np.uint8) * 255)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def warmup_pp_doclayout(config_yaml_path: str) -> None:
    """Load PP-DocLayoutV3 onto GPU and compile NVRTC CUDA kernels.

    PyTorch caches JIT-compiled kernels in ~/.cache/torch/ for the container
    lifetime, so subsequent loads within the same container run skip recompilation.
    """
    try:
        import warnings
        warnings.filterwarnings("ignore")

        from glmocr.config import load_config
        from glmocr.layout.layout_detector import PPDocLayoutDetector

        logger.info("[warmup] PP-DocLayoutV3: loading weights onto GPU...")
        t0 = time.monotonic()

        cfg = load_config(config_yaml_path)
        det = PPDocLayoutDetector(cfg.pipeline.layout)
        det.start()
        logger.info(f"[warmup] PP-DocLayoutV3: weights loaded on {det._device}")

        # One dummy inference — triggers NVRTC JIT compilation of CUDA reduction
        # kernels (reduction_prod_kernel etc.). Kernels are cached after this call.
        dummy = Image.fromarray(np.zeros((640, 480, 3), dtype=np.uint8))
        det.process([dummy])
        det.stop()

        logger.info(
            f"[warmup] PP-DocLayoutV3: ready in {time.monotonic() - t0:.1f}s "
            "(NVRTC kernels compiled and cached)"
        )
    except Exception as exc:
        logger.warning(f"[warmup] PP-DocLayoutV3 skipped: {exc}")


def warmup_ollama_glmocr(host: str, port: int, model: str) -> None:
    """Send a minimal generate request to Ollama to load glm-ocr into GPU VRAM.

    The model stays resident in VRAM indefinitely due to OLLAMA_KEEP_ALIVE=-1,
    so all subsequent OCR requests get an already-loaded model.
    """
    url = f"http://{host}:{port}/api/generate"
    logger.info(f"[warmup] GLM-OCR: loading {model} into Ollama VRAM via {url} ...")
    t0 = time.monotonic()

    payload = {
        "model": model,
        "prompt": "warmup",
        "images": [_tiny_white_png_b64()],
        "stream": False,
        "options": {"num_predict": 1},  # Generate 1 token — enough to force model load
    }

    try:
        with httpx.Client(timeout=300.0) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
        logger.info(
            f"[warmup] GLM-OCR: model loaded into GPU VRAM in {time.monotonic() - t0:.1f}s"
        )
    except Exception as exc:
        logger.warning(f"[warmup] GLM-OCR Ollama warmup skipped: {exc}")


def warmup_vllm(base_url: str, timeout_seconds: int = 300) -> None:
    """Poll vLLM /health until ready or timeout.

    Only runs when OPENAI_BASE_URL is set and does not contain 'runpod.ai'.
    On first container start, model download can take 3–10 minutes, hence the
    generous default timeout.
    """
    if "runpod.ai" in base_url:
        logger.info("[warmup] vLLM: RunPod URL detected — skipping local warmup")
        return

    # Strip /v1 suffix to get the vLLM server root, then append /health
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    health_url = f"{root}/health"

    logger.info(f"[warmup] vLLM: polling {health_url} (timeout={timeout_seconds}s)...")
    t0 = time.monotonic()
    last_log = t0

    while True:
        elapsed = time.monotonic() - t0
        if elapsed >= timeout_seconds:
            logger.warning(
                f"[warmup] vLLM: timed out after {timeout_seconds}s — "
                "model may still be loading; continuing anyway"
            )
            return
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(health_url)
                if resp.status_code == 200:
                    logger.info(f"[warmup] vLLM: ready in {elapsed:.1f}s")
                    return
        except Exception:
            pass

        now = time.monotonic()
        if now - last_log >= 30:
            logger.info(f"[warmup] vLLM: still waiting... ({int(now - t0)}s elapsed)")
            last_log = now
        time.sleep(5)


def main() -> None:
    settings = get_settings()

    # vLLM readiness — runs whenever a non-RunPod OPENAI_BASE_URL is configured,
    # independent of PARSER_BACKEND.
    if settings.openai_base_url:
        warmup_vllm(settings.openai_base_url)

    if settings.parser_backend != "ollama":
        logger.info("[warmup] PARSER_BACKEND != ollama — skipping GPU warmup")
        return

    try:
        from glmocr.config import load_config
        cfg = load_config(settings.config_yaml_path)
        ocr = cfg.pipeline.ocr_api
        ollama_host = ocr.api_host
        ollama_port = ocr.api_port
        ollama_model = ocr.model
    except Exception as exc:
        logger.warning(f"[warmup] Could not read Ollama config, using defaults: {exc}")
        ollama_host, ollama_port, ollama_model = "ollama", 11434, "glm-ocr:latest"

    logger.info("[warmup] Starting model pre-warm sequence...")

    # PP-DocLayoutV3 first — gets NVRTC kernels compiled before Ollama
    # competes for GPU, avoiding the load-timeout race condition.
    warmup_pp_doclayout(settings.config_yaml_path)

    # GLM-OCR second — GPU is now free, model loads cleanly in one attempt.
    warmup_ollama_glmocr(ollama_host, ollama_port, ollama_model)

    logger.info("[warmup] All models hot — handing off to API server")


if __name__ == "__main__":
    main()
