# AuroraIMG-6M image API. GET /generate?prompt=... with "Authorization: Bearer $AURORA_API_KEY" returns an image.
# Runs the ONNX export of https://huggingface.co/AuroraAI-Research/AuroraIMG-6M on CPU with onnxruntime.
import os, re, io, json, hmac, time, zlib, threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from pathlib import Path
import numpy as np, onnxruntime as ort
from PIL import Image
from tokenizers import Tokenizer

KEY = os.environ.get("AURORA_API_KEY", "").encode()
PORT = int(os.environ.get("PORT", 8080))
THREADS = int(os.environ.get("THREADS", 1))          # onnxruntime threads per generation
CONCURRENCY = int(os.environ.get("CONCURRENCY", 2))  # generations at once; the rest queue
MAX_QUEUE = int(os.environ.get("MAX_QUEUE", 32))     # queued requests beyond this get 503
FORMATS = {"webp": ("WEBP", "image/webp", {"quality": 90}), "png": ("PNG", "image/png", {}), "jpeg": ("JPEG", "image/jpeg", {"quality": 90})}

M = Path(__file__).parent / "model"
opts = ort.SessionOptions()
opts.intra_op_num_threads, opts.inter_op_num_threads = THREADS, 1
S = {n: ort.InferenceSession(M / f"{n}.onnx", opts, providers=["CPUExecutionProvider"]) for n in ("text_encoder", "denoiser", "decoder")}
run = lambda n, **x: S[n].run(None, x)[0]
tok = Tokenizer.from_file(str(M / "tokenizer.json"))

# Same cleanup and layout as the reference pipeline: <bos>=1, up to 62 tokens, <eos>=2, padded with 0 to 64
def encode(p):
    p = re.sub(r"https?://\S+|www\.\S+", " ", re.sub(r"<[^>]+>", " ", p))
    seq = [1] + tok.encode(re.sub(r"\s+", " ", p.lower()).strip(), add_special_tokens=False).ids[:62] + [2]
    t = np.array([seq + [0] * (64 - len(seq))], np.int64)
    return t, t != 0

NULL_TOK, NULL_MASK = encode("")
NULL_CTX = run("text_encoder", tok=NULL_TOK, mask=NULL_MASK)

def generate(prompt, steps, cfg, seed):
    t, m = encode(prompt)
    ctx = run("text_encoder", tok=t, mask=m)
    z = np.random.default_rng(seed).standard_normal((1, 8, 16, 16), dtype=np.float32)
    ts = np.linspace(1, 0, steps + 1, dtype=np.float32)
    for a, b in zip(ts[:-1], ts[1:]):
        if cfg > 1:
            v_c, v_u = run("denoiser", x=np.concatenate([z, z]), t=np.full(2, a * 1000, np.float32),
                           ctx=np.concatenate([ctx, NULL_CTX]), mask=np.concatenate([m, NULL_MASK]))
            v = v_u + cfg * (v_c - v_u)
            v *= np.linalg.norm(v_c) / max(np.linalg.norm(v), 1e-8)  # the pipeline's default cfg_rescale=1.0
        else:
            v = run("denoiser", x=z, t=np.full(1, a * 1000, np.float32), ctx=ctx, mask=m)
        z = z + (b - a) * v
    img = run("decoder", z=z)[0].transpose(1, 2, 0)
    return Image.fromarray((img * 255).round().astype(np.uint8))

def params(q):
    def num(k, cast, lo, hi, default):
        try: v = cast(q[k]) if k in q else default
        except ValueError: raise ValueError(f"{k} must be a number")
        if not lo <= v <= hi: raise ValueError(f"{k} must be {lo}-{hi}")
        return v
    prompt = q.get("prompt", "").strip()
    if not 1 <= len(prompt) <= 200: raise ValueError("prompt must be 1-200 characters")
    fmt = q.get("format", "webp").lower()
    if fmt not in FORMATS: raise ValueError(f"format must be one of {', '.join(FORMATS)}")
    return dict(prompt=prompt, steps=num("steps", int, 1, 50, 12), cfg=num("cfg", float, 0, 20, 6.0),
                seed=num("seed", int, 0, 2**32 - 1, zlib.crc32(prompt.lower().encode())), size=num("size", int, 64, 1024, 512), fmt=fmt)

gen_slots, queue_lock, queued = threading.BoundedSemaphore(CONCURRENCY), threading.Lock(), 0

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def reply(s, code, body, ctype="application/json", **headers):
        s.send_response(code)
        for k, v in {"Content-Type": ctype, "Content-Length": len(body), "Cache-Control": "no-store", **headers}.items(): s.send_header(k, str(v))
        s.end_headers(); s.wfile.write(body)

    err = lambda s, code, msg, **h: s.reply(code, json.dumps({"error": msg}).encode(), **h)

    def do_GET(s):
        global queued
        u = urlparse(s.path)
        if u.path == "/health": return s.reply(200, b"ok", "text/plain")
        if u.path != "/generate": return s.err(404, "not found")
        if not KEY or not hmac.compare_digest(s.headers.get("Authorization", "").encode(), b"Bearer " + KEY): return s.err(401, "unauthorized")
        try: p = params({k: v[-1] for k, v in parse_qs(u.query).items()})
        except ValueError as e: return s.err(400, str(e))
        with queue_lock:
            if queued >= CONCURRENCY + MAX_QUEUE: return s.err(503, "busy, retry shortly", **{"Retry-After": 2})
            queued += 1
        try:
            t0 = time.perf_counter()
            with gen_slots: img = generate(p["prompt"], p["steps"], p["cfg"], p["seed"])
            if p["size"] != 64: img = img.resize((p["size"],) * 2, Image.LANCZOS)
            fmt, ctype, save = FORMATS[p["fmt"]]
            buf = io.BytesIO(); img.save(buf, fmt, **save)
            s.reply(200, buf.getvalue(), ctype, **{"X-Seed": p["seed"], "X-Generation-Ms": round((time.perf_counter() - t0) * 1000)})
        finally:
            with queue_lock: queued -= 1

if __name__ == "__main__":
    if not KEY: print("warning: AURORA_API_KEY is not set, so /generate rejects every request", flush=True)
    generate("warmup", 2, 6.0, 0)
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    server.daemon_threads = True
    print(f"listening on :{PORT} (threads={THREADS}, concurrency={CONCURRENCY})", flush=True)
    server.serve_forever()
