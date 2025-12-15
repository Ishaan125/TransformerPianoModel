from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.concurrency import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator
from typing import AsyncIterator, AsyncIterator, List, Optional, Dict, Any
import io
import torch
from pathlib import Path
from load_model import load_model, autoregressive_generate
import pretty_midi
import time
import asyncio
import logging

logger = logging.getLogger("piano_api")
logging.basicConfig(level=logging.INFO)

class GenRequest(BaseModel):
    seed: List[int] = Field(..., description="Seed pitch tokens (ints)")
    gen_steps: int = Field(200, ge=1, le=2000)
    temperature: float = Field(1.0, gt=0.0)
    top_k: int = Field(8, ge=1)
    repeat_penalty: float = Field(1.0, ge=0.1)
    device: str = Field("cuda")  # preferred device; server may fall back to cpu
    checkpoint: Optional[str] = Field(None, description="Path to checkpoint. If omitted, the default cached model is used.")
    filename: Optional[str] = Field(None, description="Optional filename for the returned MIDI (basename only). If omitted a timestamped name will be used.")

    @field_validator("seed")
    def seed_non_empty(cls, v):
        if not isinstance(v, list) or len(v) == 0:
            raise ValueError("seed must be a non-empty list of integer tokens")
        return v


class LoadRequest(BaseModel):
    checkpoint: str
    device: Optional[str] = Field(None, description="Explicit device to load model on (e.g. 'cuda' or 'cpu'). If omitted, server chooses.)")


app = FastAPI(title="Piano AI Generation API")

# CORS (if you serve from browser/another host)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

# Model cache: mapping checkpoint_path -> { model, device, loaded_at }
MODEL_CACHE: Dict[str, Dict[str, Any]] = {}
# Protects cache manipulations and concurrent loads
CACHE_LOCK = asyncio.Lock()
# Default checkpoint path (optional). Set to None to avoid auto-loading.
DEFAULT_CHECKPOINT: Optional[str] = None


def _select_device(requested: Optional[str]) -> torch.device:
    if requested:
        req = requested.lower()
        if "cuda" in req and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_midi_bytes(generated, vel_norm=True, min_dur=0.02):
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    current_time = 0.0
    for pitch, vel, dur in generated:
        if pitch is None or pitch < 0:
            current_time += max(min_dur, float(dur) if dur is not None else min_dur)
            continue
        pitch = int(max(0, min(127, int(round(pitch)))) )
        if vel is None:
            vel_int = 64
        else:
            if vel_norm:
                vel_int = int(round(max(0, min(1.0, float(vel))) * 127))
            else:
                vel_int = int(round(max(0, min(127, float(vel)))))
        dur_val = float(dur) if dur is not None else min_dur
        dur_val = max(min_dur, dur_val)
        start = current_time
        end = current_time + dur_val
        inst.notes.append(pretty_midi.Note(velocity=vel_int, pitch=pitch, start=start, end=end))
        current_time = end
    pm.instruments.append(inst)
    bio = io.BytesIO()
    pm.write(bio)
    bio.seek(0)
    return bio


async def _load_checkpoint_to_cache(checkpoint: str, device: torch.device, warmup: bool = True):
    """Load a checkpoint into MODEL_CACHE. Blocking model load runs in a thread."""
    p = Path(checkpoint).expanduser()
    cp = str(p)
    async with CACHE_LOCK:
        if cp in MODEL_CACHE:
            return MODEL_CACHE[cp]

        model = await asyncio.to_thread(load_model, cp, device)
        entry = {"model": model, "device": device, "loaded_at": time.time()}
        MODEL_CACHE[cp] = entry

    if warmup:
        try:
            seed = [0] if hasattr(model, "hparams") else [60]
            await asyncio.to_thread(autoregressive_generate, model, seed, 1, 1.0, 1, device, 1.0)
        except Exception:
            logger.debug("Warmup generation failed for %s", cp, exc_info=True)
    return entry


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[dict]:
    """Optionally preload a default checkpoint on startup.
    To enable, set the `DEFAULT_CHECKPOINT` variable at top of file or export an environment variable and modify this file accordingly.
    """
    if DEFAULT_CHECKPOINT:
        device = _select_device(None)
        await _load_checkpoint_to_cache(DEFAULT_CHECKPOINT, device, warmup=True)
        logger.info("Preloaded default checkpoint %s on %s", DEFAULT_CHECKPOINT, device)


@app.get("/health")
async def health():
    return {"ok": True, "cuda_available": torch.cuda.is_available()}


@app.get("/status")
async def status():
    async with CACHE_LOCK:
        loaded = {k: {"device": str(v["device"]), "loaded_at": v["loaded_at"]} for k, v in MODEL_CACHE.items()}
    return {"cuda_available": torch.cuda.is_available(), "loaded_checkpoints": loaded}


@app.post("/load")
async def load_checkpoint(req: LoadRequest, background_tasks: BackgroundTasks):
    p = Path(req.checkpoint).expanduser()
    if not p.exists():
        raise HTTPException(status_code=400, detail=f"checkpoint file not found: {p}")
    cp = str(p)
    device = _select_device(req.device)
    await _load_checkpoint_to_cache(cp, device, warmup=True)
    return {"ok": True, "checkpoint": cp, "device": str(device)}


@app.post("/unload")
async def unload_checkpoint(checkpoint: str):
    cp = str(Path(checkpoint).expanduser())
    async with CACHE_LOCK:
        if cp in MODEL_CACHE:
            # attempt to free CUDA memory if on gpu
            entry = MODEL_CACHE.pop(cp)
            model = entry.get("model")
            if model is not None:
                del model
                torch.cuda.empty_cache()
            return {"ok": True, "unloaded": cp}
    raise HTTPException(status_code=404, detail="checkpoint not loaded")


@app.post("/generate")
async def generate(req: GenRequest):
    start_t = time.time()
    # determine checkpoint to use: explicit or the single-cached default
    async with CACHE_LOCK:
        if req.checkpoint:
            cp = str(Path(req.checkpoint).expanduser())
            entry = MODEL_CACHE.get(cp)
        else:
            # if only one model is cached, use it; otherwise require explicit checkpoint
            if len(MODEL_CACHE) == 1:
                cp, entry = next(iter(MODEL_CACHE.items()))
            else:
                entry = None
    if entry is None:
        # if provided a checkpoint, try to load on demand
        if req.checkpoint:
            device = _select_device(req.device)
            entry = await _load_checkpoint_to_cache(req.checkpoint, device, warmup=False)
        else:
            raise HTTPException(status_code=400, detail="No model cached. Provide `checkpoint` or preload one with /load.")

    model = entry["model"]
    device = entry["device"]

    # basic validation against model hparams if present
    seq_len = None
    if hasattr(model, "hparams") and isinstance(model.hparams, dict):
        seq_len = model.hparams.get("seq_len")
    if seq_len and len(req.seed) > seq_len:
        raise HTTPException(status_code=400, detail=f"seed length ({len(req.seed)}) exceeds model seq_len ({seq_len})")

    generated = await asyncio.to_thread(autoregressive_generate, model, req.seed, req.gen_steps,
        req.temperature,req.top_k,device,req.repeat_penalty,)

    midi_bytes = _to_midi_bytes(generated)
    elapsed = time.time() - start_t
    # Determine safe filename (basename only) for Content-Disposition
    if req.filename:
        safe_name = Path(req.filename).name
    else:
        safe_name = f"generated_{int(time.time())}.mid"

    headers = {"X-Gen-Time": f"{elapsed:.3f}", "Content-Disposition": f'attachment; filename="{safe_name}"'}
    return StreamingResponse(midi_bytes, media_type="audio/midi", headers=headers)


@app.get("/generate/info")
async def generate_info():
    """Return brief info on server capabilities and defaults."""
    return {
        "cuda_available": torch.cuda.is_available(),
        "default_checkpoint": DEFAULT_CHECKPOINT,
        "cached_count": len(MODEL_CACHE),
    }
