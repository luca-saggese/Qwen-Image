#!/usr/bin/env python3
"""
test_openai_server.py
---------------------
Testa tutti gli endpoint del server qwen OpenAI-compatibile.

Uso:
    python test_openai_server.py <port>

Esempio:
    python test_openai_server.py 8000
"""

import argparse
import base64
import json
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("Dipendenza mancante: installa 'requests'  →  pip install requests")

OUTPUT_DIR = Path("test_outputs")
INPUT_DIR = Path("test_input")
INPUT_IMAGE = INPUT_DIR / "t2i.png"
INPUT_VIDEO = INPUT_DIR / "t2v.mp4"
INPUT_GIRL   = INPUT_DIR / "girl.png"
INPUT_FLOWER = INPUT_DIR / "flower.png"
INPUT_HAT    = INPUT_DIR / "hat.jpeg"

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
SKIP = "\033[93m⚬\033[0m"


def _result(ok: bool, label: str, detail: str = "") -> None:
    icon = PASS if ok else FAIL
    msg = f"{icon}  {label}"
    if detail:
        msg += f"\n     {detail}"
    print(msg)


def _file_to_data_uri(path: Path, mime: str) -> str:
    """Legge un file da disco e lo restituisce come data URI base64."""
    raw = path.read_bytes()
    return f"data:{mime};base64," + base64.b64encode(raw).decode()


def _input_image_b64() -> str:
    return _file_to_data_uri(INPUT_IMAGE, "image/png")


def _input_video_b64() -> str:
    return _file_to_data_uri(INPUT_VIDEO, "video/mp4")


def _input_girl_b64() -> str:
    return _file_to_data_uri(INPUT_GIRL, "image/png")


def _input_flower_b64() -> str:
    return _file_to_data_uri(INPUT_FLOWER, "image/png")


def _input_hat_b64() -> str:
    return _file_to_data_uri(INPUT_HAT, "image/jpeg")


def _post(base_url: str, payload: dict, timeout: int) -> requests.Response:
    return requests.post(
        f"{base_url}/v1/chat/completions",
        json=payload,
        timeout=timeout,
    )


def _save_data_uri(data_uri: str, label: str, ext: str) -> Path:
    """Decodifica un data URI base64 e lo salva in OUTPUT_DIR."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _, b64 = data_uri.split(",", 1)
    raw = base64.b64decode(b64)
    out_path = OUTPUT_DIR / f"{label}{ext}"
    out_path.write_bytes(raw)
    return out_path


def _check_generation_response(
    resp: requests.Response, field: str, label: str
) -> tuple[bool, str]:
    """Valida la struttura di una response di generazione (image/video) e salva i file."""
    if resp.status_code != 200:
        return False, f"HTTP {resp.status_code}: {resp.text[:300]}"
    try:
        data = resp.json()
    except Exception:
        return False, "Risposta non è JSON valido"
    # Stampa sempre le info di debug se presenti
    debug_info = data.get("_debug")
    if debug_info:
        print(f"     [SERVER DEBUG] {json.dumps(debug_info, ensure_ascii=False)}")
    choices = data.get("choices", [])
    if not choices:
        return False, "Nessun elemento in 'choices'"
    msg = choices[0].get("message", {})
    items = msg.get(field, [])
    if not items:
        print (f"DEBUG: Risposta completa: {json.dumps(data)[:500]}")  # debug extra
        return False, f"Campo '{field}' assente o vuoto nel messaggio"
    url_field = "imageUrl" if field == "images" else "videoUrl"
    ext = ".png" if field == "images" else ".mp4"
    saved: list[Path] = []
    for i, item in enumerate(items):
        url = item.get(url_field, {}).get("url", "")
        if not url.startswith("data:"):
            return False, f"URL nel campo '{url_field}' non è un data URI valido"
        suffix = f"_{i}" if len(items) > 1 else ""
        saved.append(_save_data_uri(url, f"{label}{suffix}", ext))
    paths = ", ".join(str(p) for p in saved)
    return True, f"Salvat{'a' if len(saved) == 1 else 'e'}: {paths}"


def _check_text_response(resp: requests.Response) -> tuple[bool, str]:
    """Valida la struttura di una response di understanding (text)."""
    if resp.status_code != 200:
        return False, f"HTTP {resp.status_code}: {resp.text[:300]}"
    try:
        data = resp.json()
    except Exception:
        return False, "Risposta non è JSON valido"
    # Stampa sempre le info di debug se presenti
    debug_info = data.get("_debug")
    if debug_info:
        print(f"     [SERVER DEBUG] {json.dumps(debug_info, ensure_ascii=False)}")
    choices = data.get("choices", [])
    if not choices:
        return False, "Nessun elemento in 'choices'"
    content = choices[0].get("message", {}).get("content", None)
    if not content:
        return False, "Campo 'content' assente o vuoto"
    return True, f"Risposta: {str(content)[:120]}"


# ─────────────────────────────────────────────────────────────────────────────
# Singoli test
# ─────────────────────────────────────────────────────────────────────────────

def test_health(base_url: str, timeout: int) -> bool:
    try:
        resp = requests.get(f"{base_url}/health", timeout=timeout)
        ok = resp.status_code == 200 and resp.json().get("status") == "ok"
        _result(ok, "GET /health", json.dumps(resp.json()) if ok else resp.text[:200])
        return ok
    except Exception as exc:
        _result(False, "GET /health", str(exc))
        return False


def test_list_models(base_url: str, timeout: int) -> list[str]:
    try:
        resp = requests.get(f"{base_url}/v1/models", timeout=timeout)
        if resp.status_code != 200:
            _result(False, "GET /v1/models", f"HTTP {resp.status_code}")
            return []
        ids = [m["id"] for m in resp.json().get("data", [])]
        _result(True, "GET /v1/models", "Modelli: " + ", ".join(ids) if ids else "(nessun modello caricato)")
        return ids
    except Exception as exc:
        _result(False, "GET /v1/models", str(exc))
        return []


def test_t2i(base_url: str, timeout: int, seed: int) -> bool:
    payload = {
        "model": "qwen-t2i",
        "messages": [{"role": "user", "content": "A serene mountain lake at sunset"}],
        "seed": seed,
        "num_timesteps": 5,
    }
    try:
        resp = _post(base_url, payload, timeout)
        ok, detail = _check_generation_response(resp, "images", "t2i")
        _result(ok, "POST /v1/chat/completions  [qwen-t2i  – Text→Image]", detail)
        return ok
    except Exception as exc:
        _result(False, "POST /v1/chat/completions  [qwen-t2i  – Text→Image]", str(exc))
        return False


def test_t2v(base_url: str, timeout: int, seed: int) -> bool:
    payload = {
        "model": "qwen-t2v",
        "messages": [{"role": "user", "content": "A bird flying over the ocean"}],
        "seed": seed,
        "num_timesteps": 5,
        "num_frames": 25,
    }
    try:
        resp = _post(base_url, payload, timeout)
        ok, detail = _check_generation_response(resp, "videos", "t2v")
        _result(ok, "POST /v1/chat/completions  [qwen-t2v  – Text→Video]", detail)
        return ok
    except Exception as exc:
        _result(False, "POST /v1/chat/completions  [qwen-t2v  – Text→Video]", str(exc))
        return False


def test_i2i(base_url: str, timeout: int, seed: int) -> bool:
    if not INPUT_IMAGE.exists():
        _result(False, "POST /v1/chat/completions  [qwen-i2i  – Image→Image]", f"File non trovato: {INPUT_IMAGE}")
        return False
    payload = {
        "model": "qwen-i2i",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Turn the sunset sky into a dramatic stormy sky"},
                    {"type": "image_url", "image_url": {"url": _input_image_b64()}},
                ],
            }
        ],
        "seed": seed,
        "num_timesteps": 5,
    }
    try:
        resp = _post(base_url, payload, timeout)
        ok, detail = _check_generation_response(resp, "images", "i2i")
        _result(ok, "POST /v1/chat/completions  [qwen-i2i  – Image→Image]", detail)
        return ok
    except Exception as exc:
        _result(False, "POST /v1/chat/completions  [qwen-i2i  – Image→Image]", str(exc))
        return False


def test_i2t(base_url: str, timeout: int, seed: int) -> bool:
    if not INPUT_IMAGE.exists():
        _result(False, "POST /v1/chat/completions  [qwen-i2t  – Image→Text]", f"File non trovato: {INPUT_IMAGE}")
        return False
    payload = {
        "model": "qwen-i2t",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in detail"},
                    {"type": "image_url", "image_url": {"url": _input_image_b64()}},
                ],
            }
        ],
        "seed": seed,
    }
    try:
        resp = _post(base_url, payload, timeout)
        ok, detail = _check_text_response(resp)
        _result(ok, "POST /v1/chat/completions  [qwen-i2t  – Image→Text]", detail)
        return ok
    except Exception as exc:
        _result(False, "POST /v1/chat/completions  [qwen-i2t  – Image→Text]", str(exc))
        return False


def test_v2v(base_url: str, timeout: int, seed: int) -> bool:
    if not INPUT_VIDEO.exists():
        _result(False, "POST /v1/chat/completions  [qwen-v2v  – Video→Video]", f"File non trovato: {INPUT_VIDEO}")
        return False
    payload = {
        "model": "qwen-v2v",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Make the bird fly in slow motion over a golden ocean at sunset"},
                    {"type": "video_url", "video_url": {"url": _input_video_b64()}},
                ],
            }
        ],
        "seed": seed,
        "num_timesteps": 5,
        "num_frames": 25,
    }
    try:
        resp = _post(base_url, payload, timeout)
        ok, detail = _check_generation_response(resp, "videos", "v2v")
        _result(ok, "POST /v1/chat/completions  [qwen-v2v  – Video→Video]", detail)
        return ok
    except Exception as exc:
        _result(False, "POST /v1/chat/completions  [qwen-v2v  – Video→Video]", str(exc))
        return False


def test_v2t(base_url: str, timeout: int, seed: int) -> bool:
    if not INPUT_VIDEO.exists():
        _result(False, "POST /v1/chat/completions  [qwen-v2t  – Video→Text]", f"File non trovato: {INPUT_VIDEO}")
        return False
    payload = {
        "model": "qwen-v2t",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe what the bird is doing in this video"},
                    {"type": "video_url", "video_url": {"url": _input_video_b64()}},
                ],
            }
        ],
        "seed": seed,
    }
    try:
        resp = _post(base_url, payload, timeout)
        ok, detail = _check_text_response(resp)
        _result(ok, "POST /v1/chat/completions  [qwen-v2t  – Video→Text]", detail)
        return ok
    except Exception as exc:
        _result(False, "POST /v1/chat/completions  [qwen-v2t  – Video→Text]", str(exc))
        return False


def test_ti2v(base_url: str, timeout: int, seed: int) -> bool:
    """Image+Text → Video (ti2v / tiv2v_idip)."""
    if not INPUT_IMAGE.exists():
        _result(False, "POST /v1/chat/completions  [qwen-ti2v – Image+Text→Video]", f"File non trovato: {INPUT_IMAGE}")
        return False
    payload = {
        "model": "qwen-ti2v",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Animate this scene with gentle camera movement and soft lighting"},
                    {"type": "image_url", "image_url": {"url": _input_image_b64()}},
                ],
            }
        ],
        "seed": seed,
        "num_timesteps": 5,
        "num_frames": 25,
    }
    try:
        resp = _post(base_url, payload, timeout)
        ok, detail = _check_generation_response(resp, "videos", "ti2v")
        _result(ok, "POST /v1/chat/completions  [qwen-ti2v – Image+Text→Video]", detail)
        return ok
    except Exception as exc:
        _result(False, "POST /v1/chat/completions  [qwen-ti2v – Image+Text→Video]", str(exc))
        return False


def test_x2v(base_url: str, timeout: int, seed: int) -> bool:
    """Text + Image + Video → Video (x2v: qualsiasi mix di media → video)."""
    if not INPUT_IMAGE.exists():
        _result(False, "POST /v1/chat/completions  [qwen-x2v  – Any→Video]", f"File non trovato: {INPUT_IMAGE}")
        return False
    if not INPUT_VIDEO.exists():
        _result(False, "POST /v1/chat/completions  [qwen-x2v  – Any→Video]", f"File non trovato: {INPUT_VIDEO}")
        return False
    payload = {
        "model": "qwen-x2v",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Generate a video in the visual style of the image, with motion inspired by the reference video"},
                    {"type": "image_url", "image_url": {"url": _input_image_b64()}},
                    {"type": "video_url", "video_url": {"url": _input_video_b64()}},
                ],
            }
        ],
        "seed": seed,
        "num_timesteps": 5,
        "num_frames": 25,
    }
    try:
        resp = _post(base_url, payload, timeout)
        ok, detail = _check_generation_response(resp, "videos", "x2v")
        _result(ok, "POST /v1/chat/completions  [qwen-x2v  – Any→Video]", detail)
        return ok
    except Exception as exc:
        _result(False, "POST /v1/chat/completions  [qwen-x2v  – Any→Video]", str(exc))
        return False


def test_x2v_hat(base_url: str, timeout: int, seed: int) -> bool:
    """Girl + Flower + Hat → Video: la ragazza indossa il cappello con il fiore sopra (video)."""
    for path in (INPUT_GIRL, INPUT_FLOWER, INPUT_HAT):
        if not path.exists():
            _result(False, "POST /v1/chat/completions  [qwen-x2v  – Girl+Flower+Hat→Video]", f"File non trovato: {path}")
            return False
    payload = {
        "model": "qwen-x2v",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        "Three reference images are provided. "
                        "The first image is a hat. "
                        "The second image is a decorative flower. "
                        "The third image is a portrait of a girl. "
                        "Generate a short video portrait: the girl from the third image is wearing the hat from the first image on her head, "
                        "with the flower from the second image placed on top of the hat as a decoration. "
                        "Preserve the girl's face, skin tone and hair from the third image exactly."
                    )},
                    {"type": "image_url", "image_url": {"url": _input_hat_b64()}},
                    {"type": "image_url", "image_url": {"url": _input_flower_b64()}},
                    {"type": "image_url", "image_url": {"url": _input_girl_b64()}},
                ],
            }
        ],
        "seed": seed,
        "num_timesteps": 30,
        "num_frames": 25,
    }
    try:
        resp = _post(base_url, payload, timeout)
        ok, detail = _check_generation_response(resp, "videos", "x2v_hat")
        _result(ok, "POST /v1/chat/completions  [qwen-x2v  – Girl+Flower+Hat→Video]", detail)
        return ok
    except Exception as exc:
        _result(False, "POST /v1/chat/completions  [qwen-x2v  – Girl+Flower+Hat→Video]", str(exc))
        return False


def test_x2i_hat(base_url: str, timeout: int, seed: int) -> bool:
    """Girl + Flower + Hat → Image: metti il cappello alla ragazza con il fiore sopra."""
    for path in (INPUT_GIRL, INPUT_FLOWER, INPUT_HAT):
        if not path.exists():
            _result(False, "POST /v1/chat/completions  [qwen-x2i  – Girl+Flower+Hat→Image]", f"File non trovato: {path}")
            return False
    payload = {
        "model": "qwen-x2i",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        "Three reference images are provided. "
                        "The first image is a hat. "
                        "The second image is a decorative flower. "
                        "The third image is a portrait of a girl. "
                        "Generate a new portrait: the girl from the third image is wearing the hat from the first image on her head, "
                        "with the flower from the second image placed on top of the hat as a decoration. "
                        "Preserve the girl's face, skin tone, hair and body from the third image exactly."
                    )},
                    {"type": "image_url", "image_url": {"url": _input_hat_b64()}},
                    {"type": "image_url", "image_url": {"url": _input_flower_b64()}},
                    {"type": "image_url", "image_url": {"url": _input_girl_b64()}},
                ],
            }
        ],
        "seed": seed,
        "num_timesteps": 30,
    }
    try:
        resp = _post(base_url, payload, timeout)
        ok, detail = _check_generation_response(resp, "images", "x2i_hat")
        _result(ok, "POST /v1/chat/completions  [qwen-x2i  – Girl+Flower+Hat→Image]", detail)
        return ok
    except Exception as exc:
        _result(False, "POST /v1/chat/completions  [qwen-x2i  – Girl+Flower+Hat→Image]", str(exc))
        return False


def test_x2i(base_url: str, timeout: int, seed: int) -> bool:
    """Text + Video + Image → Image (x2i: qualsiasi mix di media → immagine)."""
    if not INPUT_VIDEO.exists():
        _result(False, "POST /v1/chat/completions  [qwen-x2i  – Any→Image]", f"File non trovato: {INPUT_VIDEO}")
        return False
    if not INPUT_IMAGE.exists():
        _result(False, "POST /v1/chat/completions  [qwen-x2i  – Any→Image]", f"File non trovato: {INPUT_IMAGE}")
        return False
    payload = {
        "model": "qwen-x2i",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Generate an image that combines the composition of this image with the mood of the video"},
                    {"type": "video_url", "video_url": {"url": _input_video_b64()}},
                    {"type": "image_url", "image_url": {"url": _input_image_b64()}},
                ],
            }
        ],
        "seed": seed,
        "num_timesteps": 5,
    }
    try:
        resp = _post(base_url, payload, timeout)
        ok, detail = _check_generation_response(resp, "images", "x2i")
        _result(ok, "POST /v1/chat/completions  [qwen-x2i  – Any→Image]", detail)
        return ok
    except Exception as exc:
        _result(False, "POST /v1/chat/completions  [qwen-x2i  – Any→Image]", str(exc))
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Registro dei test (nome → funzione, modello richiesto, descrizione)
# ─────────────────────────────────────────────────────────────────────────────

# Ogni entry: (funzione, modello_richiesto_su_server_o_None, descrizione)
_TESTS: list[tuple] = [
    ("health",      None,           "GET /health",                          lambda bu, to, se: test_health(bu, to)),
    ("list_models", None,           "GET /v1/models",                       None),  # gestito a parte
    ("t2i",         "qwen-t2i",    "Text → Image",                         lambda bu, to, se: test_t2i(bu, to, se)),
    ("i2i",         "qwen-i2i",    "Image → Image",                        lambda bu, to, se: test_i2i(bu, to, se)),
    ("i2t",         "qwen-i2t",    "Image → Text",                         lambda bu, to, se: test_i2t(bu, to, se)),
    ("t2v",         "qwen-t2v",    "Text → Video",                         lambda bu, to, se: test_t2v(bu, to, se)),
    ("v2v",         "qwen-v2v",    "Video → Video",                        lambda bu, to, se: test_v2v(bu, to, se)),
    ("v2t",         "qwen-v2t",    "Video → Text",                         lambda bu, to, se: test_v2t(bu, to, se)),
    ("ti2v",        "qwen-ti2v",   "Image + Text → Video",                 lambda bu, to, se: test_ti2v(bu, to, se)),
    ("x2v",         "qwen-x2v",    "Any (text/image/video) → Video",       lambda bu, to, se: test_x2v(bu, to, se)),
    ("x2i",         "qwen-x2i",    "Any (text/image/video) → Image",       lambda bu, to, se: test_x2i(bu, to, se)),
    ("x2v_hat",     "qwen-x2v",    "Girl + Flower + Hat → Video (cappello)", lambda bu, to, se: test_x2v_hat(bu, to, se)),
    ("x2i_hat",     "qwen-x2i",    "Girl + Flower + Hat → Image (cappello)", lambda bu, to, se: test_x2i_hat(bu, to, se)),
]

_RUNNABLE_NAMES = [name for name, model, *_ in _TESTS if model is not None]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    test_list_help = "  " + "\n  ".join(
        f"{name:<12}  {desc}"
        for name, model, desc, *_ in _TESTS
        if model is not None
    )
    parser = argparse.ArgumentParser(
        description="Test del server qwen OpenAI-compatibile",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Test disponibili:\n{test_list_help}",
    )
    parser.add_argument("port", type=int, help="Porta su cui gira il server (es. 8000)")
    parser.add_argument("--host", default="127.0.0.1", help="Host del server (default: 127.0.0.1)")
    parser.add_argument("--seed", type=int, default=42, help="Seed per la generazione (default: 42)")
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout in secondi per ogni richiesta (default: 300)",
    )
    parser.add_argument(
        "--test",
        metavar="NOME",
        nargs="+",
        choices=_RUNNABLE_NAMES,
        help=(
            "Esegui solo i test specificati (spazio-separati). "
            f"Valori validi: {', '.join(_RUNNABLE_NAMES)}"
        ),
    )
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    selected: set[str] = set(args.test) if args.test else set(_RUNNABLE_NAMES)

    print(f"\nServer: {base_url}  |  seed={args.seed}  |  timeout={args.timeout}s")
    if args.test:
        print(f"Test selezionati: {', '.join(args.test)}")
    print(f"Output:  {OUTPUT_DIR.resolve()}\n")
    print("=" * 65)

    results: dict[str, bool | None] = {}

    # ── Health & models list (sempre eseguiti) ────────────────────────────
    results["health"] = test_health(base_url, timeout=10)
    available_models = test_list_models(base_url, timeout=10)
    results["list_models"] = True

    print()

    # ── Test selezionati ──────────────────────────────────────────────────
    prev_group = None
    for name, model, desc, fn in _TESTS:
        if model is None:
            continue  # health/list_models già eseguiti sopra
        if name not in selected:
            continue

        # Separatore visivo tra gruppo immagine e gruppo video
        group = "image" if model in {"qwen-t2i", "qwen-i2i", "qwen-i2t", "qwen-x2i"} else "video"
        if prev_group is not None and group != prev_group:
            print()
        prev_group = group

        if model in available_models:
            results[name] = fn(base_url, args.timeout, args.seed)
        else:
            label = f"POST /v1/chat/completions  [{model}  – {desc}]"
            print(f"{SKIP}  {label}  (pipeline non caricata)")
            results[name] = None

    # ── Riepilogo ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    passed  = sum(1 for v in results.values() if v is True)
    failed  = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)
    print(f"Riepilogo: {passed} passati  |  {failed} falliti  |  {skipped} saltati\n")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
