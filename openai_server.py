"""
Qwen Image OpenAI-Compatible API Server
========================================
FastAPI server che espone un'interfaccia OpenAI Chat Completions per i modelli Qwen Image.

Task supportati (selezionati tramite il campo 'model' nella richiesta):
  qwen-t2i   : Text-to-Image   (QwenImagePipeline,         Qwen/Qwen-Image-2512)
  qwen-i2i   : Image Editing   (QwenImageEditPlusPipeline,  Qwen/Qwen-Image-Edit-2511)

Formato input (OpenAI chat completions):
  {
    "model": "qwen-t2i",
    "messages": [{"role": "user", "content": [
      {"type": "text", "text": "A beautiful landscape"},
      // solo per qwen-i2i:
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
    ]}],
    // parametri opzionali
    "seed": 42,
    "width": 1664,
    "height": 928,
    "num_inference_steps": 50,
    "true_cfg_scale": 4.0,
    "guidance_scale": 1.0,
    "negative_prompt": "low quality, ...",
    "aspect_ratio": "16:9"
  }

Formato output (OpenAI-compatibile):
  {
    "id": "gen-...",
    "object": "chat.completion",
    "created": 1234567890,
    "model": "qwen-t2i",
    "choices": [{
      "index": 0,
      "message": {
        "role": "assistant",
        "content": null,
        "images": [{"image_url": {"url": "data:image/png;base64,..."}}]
      },
      "finish_reason": "stop"
    }]
  }

Avvio:
  python openai_server.py

  Opzioni principali:
    --port 8000           Porta (default 8000)
    --gpu 0               GPU da usare (default 0)
    --preload             Carica i modelli subito invece che al primo request
    --model-t2i ID        HuggingFace model ID per t2i (default: Qwen/Qwen-Image-2512)
    --model-i2i ID        HuggingFace model ID per i2i (default: Qwen/Qwen-Image-Edit-2511)
"""
from __future__ import annotations

import argparse
import base64
import mimetypes
import random
import shutil
import threading
import time
import traceback
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from diffusers import QwenImageEditPlusPipeline, QwenImagePipeline
from PIL import Image

# ── FastAPI ──────────────────────────────────────────────────────────────────
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, model_validator

# ─────────────────────────────────────────────────────────────────────────────
# Costanti
# ─────────────────────────────────────────────────────────────────────────────

TASK_T2I = "t2i"
TASK_I2I = "i2i"

DEFAULT_MODEL_T2I = "Qwen/Qwen-Image-2512"
DEFAULT_MODEL_I2I = "Qwen/Qwen-Image-Edit-2511"

DEFAULT_NUM_INFERENCE_STEPS_T2I = 50
DEFAULT_NUM_INFERENCE_STEPS_I2I = 40
DEFAULT_TRUE_CFG_SCALE = 4.0
DEFAULT_GUIDANCE_SCALE = 1.0
DEFAULT_NEGATIVE_PROMPT = (
    "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，"
    "过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。"
)

# Aspect ratio → (width, height) per QwenImagePipeline (segue convenzione W:H)
ASPECT_RATIOS: Dict[str, tuple] = {
    "1:1":  (1328, 1328),
    "16:9": (1664,  928),
    "9:16": ( 928, 1664),
    "4:3":  (1472, 1104),
    "3:4":  (1104, 1472),
    "3:2":  (1584, 1056),
    "2:3":  (1056, 1584),
    "21:9": (1904,  816),
    "1:2":  ( 928, 1856),
    "2:1":  (1856,  928),
}

# ─────────────────────────────────────────────────────────────────────────────
# Pydantic models per la request/response
# ─────────────────────────────────────────────────────────────────────────────


class ImageUrl(BaseModel):
    url: str


class ContentPart(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[ImageUrl] = None


class Message(BaseModel):
    role: str = "user"
    content: Union[str, List[ContentPart]] = ""


class ChatCompletionRequest(BaseModel):
    model: str = "qwen-t2i"
    # Supporta sia "messages" (OpenAI standard) sia "input" (Responses API)
    messages: Optional[List[Message]] = None
    input: Optional[List[Message]] = None

    # Parametri di generazione opzionali
    seed: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    aspect_ratio: Optional[str] = None          # es. "16:9" – override width/height
    num_inference_steps: Optional[int] = None
    true_cfg_scale: Optional[float] = None
    guidance_scale: Optional[float] = None      # solo i2i
    negative_prompt: Optional[str] = None

    # Campi OpenAI standard accettati per compatibilità ma non utilizzati
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stream: Optional[bool] = False

    model_config = {"extra": "allow"}

    @model_validator(mode="after")
    def require_messages_or_input(self) -> "ChatCompletionRequest":
        if self.messages is None and self.input is None:
            raise ValueError("Almeno uno tra 'messages' e 'input' è obbligatorio.")
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Utility: gestione media (base64 / URL)
# ─────────────────────────────────────────────────────────────────────────────


def _download_url(url: str) -> bytes:
    """Scarica un URL e restituisce i byte grezzi."""
    import urllib.request
    with urllib.request.urlopen(url) as resp:  # noqa: S310
        return resp.read()


def resolve_image(url_or_b64: str) -> Image.Image:
    """
    Risolve un URL, una stringa base64 o un percorso locale in un'immagine PIL.
    """
    if url_or_b64.startswith("data:"):
        _, b64data = url_or_b64.split(",", 1)
        return Image.open(BytesIO(base64.b64decode(b64data))).convert("RGB")

    if url_or_b64.startswith("http://") or url_or_b64.startswith("https://"):
        return Image.open(BytesIO(_download_url(url_or_b64))).convert("RGB")

    local = Path(url_or_b64)
    if local.exists():
        return Image.open(local).convert("RGB")

    raise ValueError(f"Impossibile risolvere l'immagine: {url_or_b64[:80]!r}")


def pil_to_data_url(img: Image.Image) -> str:
    """Converte un'immagine PIL in un data URI base64 PNG."""
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


# ─────────────────────────────────────────────────────────────────────────────
# Task detection
# ─────────────────────────────────────────────────────────────────────────────

_MODEL_TO_TASK: Dict[str, str] = {
    "qwen-t2i":       TASK_T2I,
    "qwen-i2i":       TASK_I2I,
    "qwen-image":     TASK_T2I,
    "qwen-edit":      TASK_I2I,
    "qwen-image-edit": TASK_I2I,
}


def detect_task(model_name: str, has_image: bool) -> str:
    """Determina il task dal nome del modello e dalla presenza di un'immagine."""
    model_lower = model_name.strip().lower()
    if model_lower in _MODEL_TO_TASK:
        return _MODEL_TO_TASK[model_lower]
    # Fallback automatico: se c'è un'immagine → i2i, altrimenti t2i
    return TASK_I2I if has_image else TASK_T2I


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Qwen
# ─────────────────────────────────────────────────────────────────────────────


class QwenT2IPipeline:
    """Wrapper thread-safe attorno a QwenImagePipeline."""

    def __init__(self, model_id: str, device_id: int) -> None:
        self._lock = threading.Lock()
        self.model_id = model_id
        self.device_id = device_id
        self.pipe: Optional[QwenImagePipeline] = None
        self.initialized = False

    def initialize(self) -> None:
        with self._lock:
            if self.initialized:
                return
            torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            device = f"cuda:{self.device_id}" if torch.cuda.is_available() else "cpu"
            print(f"[qwen_server] Carico QwenImagePipeline ({self.model_id}) su {device}...", flush=True)
            t0 = time.perf_counter()
            self.pipe = QwenImagePipeline.from_pretrained(
                self.model_id, torch_dtype=torch_dtype
            ).to(device)
            print(f"[qwen_server] QwenImagePipeline pronta in {time.perf_counter()-t0:.1f}s", flush=True)
            self.initialized = True

    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        num_inference_steps: int,
        true_cfg_scale: float,
        seed: int,
    ) -> Image.Image:
        self.initialize()
        assert self.pipe is not None
        device = f"cuda:{self.device_id}" if torch.cuda.is_available() else "cpu"
        with self._lock:
            generator = torch.Generator(device=device).manual_seed(seed)
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                true_cfg_scale=true_cfg_scale,
                generator=generator,
            )
        return result.images[0]


class QwenI2IPipeline:
    """Wrapper thread-safe attorno a QwenImageEditPlusPipeline."""

    def __init__(self, model_id: str, device_id: int) -> None:
        self._lock = threading.Lock()
        self.model_id = model_id
        self.device_id = device_id
        self.pipe: Optional[QwenImageEditPlusPipeline] = None
        self.initialized = False

    def initialize(self) -> None:
        with self._lock:
            if self.initialized:
                return
            torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            device = f"cuda:{self.device_id}" if torch.cuda.is_available() else "cpu"
            print(f"[qwen_server] Carico QwenImageEditPlusPipeline ({self.model_id}) su {device}...", flush=True)
            t0 = time.perf_counter()
            self.pipe = QwenImageEditPlusPipeline.from_pretrained(
                self.model_id, torch_dtype=torch_dtype
            ).to(device)
            print(f"[qwen_server] QwenImageEditPlusPipeline pronta in {time.perf_counter()-t0:.1f}s", flush=True)
            self.initialized = True

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: str,
        num_inference_steps: int,
        true_cfg_scale: float,
        guidance_scale: float,
        seed: int,
    ) -> Image.Image:
        self.initialize()
        assert self.pipe is not None
        with self._lock:
            generator = torch.manual_seed(seed)
            with torch.inference_mode():
                result = self.pipe(
                    image=[image],
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    true_cfg_scale=true_cfg_scale,
                    guidance_scale=guidance_scale,
                    num_images_per_prompt=1,
                    generator=generator,
                )
        return result.images[0]


# ─────────────────────────────────────────────────────────────────────────────
# Applicazione FastAPI
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Qwen Image OpenAI-Compatible API",
    description="API REST compatibile OpenAI per i modelli Qwen Image (t2i, i2i)",
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    error_trace = traceback.format_exc()
    print(f"ERRORE NON GESTITO:\n{error_trace}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "traceback": error_trace},
    )


# Istanze globali delle pipeline
_t2i_pipeline: Optional[QwenT2IPipeline] = None
_i2i_pipeline: Optional[QwenI2IPipeline] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "t2i_pipeline": _t2i_pipeline.initialized if _t2i_pipeline else False,
        "i2i_pipeline": _i2i_pipeline.initialized if _i2i_pipeline else False,
    }


@app.get("/v1/models")
async def list_models():
    available = []
    if _t2i_pipeline is not None:
        available.append({"id": "qwen-t2i", "object": "model", "owned_by": "qwen"})
    if _i2i_pipeline is not None:
        available.append({"id": "qwen-i2i", "object": "model", "owned_by": "qwen"})
    return {"object": "list", "data": available}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    Endpoint principale compatibile OpenAI Chat Completions.
    Supporta qwen-t2i (text-to-image) e qwen-i2i (image editing).
    """
    try:
        body = await request.json()
    except Exception:
        body = None

    try:
        req = ChatCompletionRequest.model_validate(body or {})
    except Exception as exc:
        print(f"[qwen_server][error] Validazione request fallita: {exc}", flush=True)
        raise HTTPException(status_code=400, detail=f"Request validation error: {exc}")

    # ── Normalizza messaggi ────────────────────────────────────────────────
    messages = req.messages or req.input or []
    content_parts: List[ContentPart] = []
    for msg in reversed(messages):
        if msg.role in ("user", "human") or len(messages) == 1:
            if isinstance(msg.content, str):
                content_parts = [ContentPart(type="text", text=msg.content)]
            else:
                content_parts = msg.content or []
            break

    # ── Estrai testo e immagini ────────────────────────────────────────────
    texts: List[str] = []
    image_urls: List[str] = []
    for part in content_parts:
        if part.type == "text" and part.text:
            texts.append(part.text)
        elif part.type == "image_url" and part.image_url:
            image_urls.append(part.image_url.url)

    prompt = " ".join(texts).strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Il campo 'text' (prompt) è obbligatorio.")

    # ── Rileva il task ─────────────────────────────────────────────────────
    task = detect_task(model_name=req.model, has_image=bool(image_urls))
    print(f"[qwen_server] model={req.model!r}  task={task!r}  has_image={bool(image_urls)}", flush=True)

    # ── Parametri di generazione ───────────────────────────────────────────
    seed = req.seed if req.seed is not None else random.randint(0, 2**31 - 1)
    negative_prompt = req.negative_prompt if req.negative_prompt is not None else DEFAULT_NEGATIVE_PROMPT
    true_cfg_scale = req.true_cfg_scale if req.true_cfg_scale is not None else DEFAULT_TRUE_CFG_SCALE
    guidance_scale = req.guidance_scale if req.guidance_scale is not None else DEFAULT_GUIDANCE_SCALE

    # ── Risoluzione / aspect ratio (solo t2i) ──────────────────────────────
    if task == TASK_T2I:
        if req.aspect_ratio is not None:
            if req.aspect_ratio not in ASPECT_RATIOS:
                raise HTTPException(
                    status_code=400,
                    detail=f"aspect_ratio '{req.aspect_ratio}' non supportato. "
                           f"Valori validi: {sorted(ASPECT_RATIOS)}",
                )
            w_default, h_default = ASPECT_RATIOS[req.aspect_ratio]
        else:
            w_default, h_default = ASPECT_RATIOS["16:9"]
        width = req.width if req.width is not None else w_default
        height = req.height if req.height is not None else h_default
        num_steps = req.num_inference_steps if req.num_inference_steps is not None else DEFAULT_NUM_INFERENCE_STEPS_T2I
    else:
        num_steps = req.num_inference_steps if req.num_inference_steps is not None else DEFAULT_NUM_INFERENCE_STEPS_I2I
        width = height = 0  # non usati per i2i

    # ── Esegui inferenza ───────────────────────────────────────────────────
    import asyncio
    loop = asyncio.get_event_loop()

    try:
        if task == TASK_T2I:
            if _t2i_pipeline is None:
                raise HTTPException(status_code=503, detail="Pipeline t2i non disponibile.")
            t0 = time.perf_counter()
            output_image: Image.Image = await loop.run_in_executor(
                None,
                lambda: _t2i_pipeline.generate(  # type: ignore[union-attr]
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_steps,
                    true_cfg_scale=true_cfg_scale,
                    seed=seed,
                ),
            )
            print(f"[qwen_server] t2i completato in {time.perf_counter()-t0:.2f}s", flush=True)

        else:  # TASK_I2I
            if _i2i_pipeline is None:
                raise HTTPException(status_code=503, detail="Pipeline i2i non disponibile.")
            if not image_urls:
                raise HTTPException(status_code=400, detail="qwen-i2i richiede un'immagine in input.")
            try:
                input_image = resolve_image(image_urls[0])
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Impossibile caricare l'immagine: {exc}") from exc
            t0 = time.perf_counter()
            output_image = await loop.run_in_executor(
                None,
                lambda: _i2i_pipeline.generate(  # type: ignore[union-attr]
                    image=input_image,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_steps,
                    true_cfg_scale=true_cfg_scale,
                    guidance_scale=guidance_scale,
                    seed=seed,
                ),
            )
            print(f"[qwen_server] i2i completato in {time.perf_counter()-t0:.2f}s", flush=True)

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Errore durante l'inferenza: {exc}") from exc

    # ── Codifica output ────────────────────────────────────────────────────
    image_data_url = pil_to_data_url(output_image)

    return {
        "id": f"gen-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "images": [
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data_url},
                            "imageUrl":  {"url": image_data_url},
                        }
                    ],
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen Image OpenAI-compatible API server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-t2i", type=str, default=DEFAULT_MODEL_T2I,
                        help="HuggingFace model ID per text-to-image.")
    parser.add_argument("--model-i2i", type=str, default=DEFAULT_MODEL_I2I,
                        help="HuggingFace model ID per image editing.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="ID GPU da usare per entrambe le pipeline.")
    parser.add_argument("--port", type=int, default=8000, help="Porta del server.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host del server.")
    parser.add_argument("--preload", action="store_true",
                        help="Carica i modelli subito all'avvio.")
    parser.add_argument("--disable-t2i", action="store_true",
                        help="Non caricare la pipeline text-to-image.")
    parser.add_argument("--disable-i2i", action="store_true",
                        help="Non caricare la pipeline image editing.")
    return parser.parse_args()


def main() -> None:
    import uvicorn

    args = _parse_args()

    global _t2i_pipeline, _i2i_pipeline

    if not args.disable_t2i:
        _t2i_pipeline = QwenT2IPipeline(model_id=args.model_t2i, device_id=args.gpu)
        print(f"[qwen_server] t2i pipeline: {args.model_t2i} @ GPU {args.gpu}")

    if not args.disable_i2i:
        _i2i_pipeline = QwenI2IPipeline(model_id=args.model_i2i, device_id=args.gpu)
        print(f"[qwen_server] i2i pipeline: {args.model_i2i} @ GPU {args.gpu}")

    if _t2i_pipeline is None and _i2i_pipeline is None:
        print("[qwen_server] ERRORE: Nessuna pipeline abilitata.")
        return

    if args.preload:
        if _t2i_pipeline is not None:
            print("[qwen_server] Pre-carico t2i pipeline...")
            _t2i_pipeline.initialize()
        if _i2i_pipeline is not None:
            print("[qwen_server] Pre-carico i2i pipeline...")
            _i2i_pipeline.initialize()

    print(f"\n[qwen_server] Server in ascolto su http://{args.host}:{args.port}")
    print("[qwen_server] Endpoint: POST /v1/chat/completions")
    print(f"[qwen_server] Docs:     http://localhost:{args.port}/docs\n")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        h11_max_incomplete_event_size=256 * 1024 * 1024,  # 256 MB per immagini base64
    )


if __name__ == "__main__":
    main()
