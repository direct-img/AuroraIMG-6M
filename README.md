# AuroraIMG-6M API

A small, authenticated text-to-image API for direct-img. It runs [AuroraIMG-6M](https://huggingface.co/AuroraAI-Research/AuroraIMG-6M), a 6.8M-parameter model, on CPU with onnxruntime. An image takes about 0.4s and the container uses about 150-200 MB of RAM.

The output is 64×64 and upscaled, so images are soft and painterly: right colors, setting and mood, often not the exact object. That suits an abstract fallback when no real image exists.

## API

```
GET /generate?prompt=frog+wizard+casting+a+spell
Authorization: Bearer $AURORA_API_KEY
```

| Param | Default | Range |
|---|---|---|
| `prompt` | required | 1-200 characters |
| `size` | `512` | 64-1024 (output is square, upscaled from 64×64) |
| `format` | `webp` | `webp`, `png`, `jpeg` |
| `steps` | `12` | 1-50 (12 looks the same as 50, at a quarter of the time) |
| `cfg` | `6` | 0-20 (guidance; ≤1 disables it) |
| `seed` | CRC32 of the lowercased prompt | 0-4294967295 |

The same prompt and params always give the same image. Responses include `X-Seed` and `X-Generation-Ms`.

Errors are JSON `{"error": "..."}`: 400 bad params, 401 missing or wrong key, 503 when the queue is full (`Retry-After: 2`).

`GET /health` returns `ok` without auth.

## Config

| Env var | Default | |
|---|---|---|
| `AURORA_API_KEY` | none | Required. Without it `/generate` rejects every request |
| `PORT` | `8080` | |
| `THREADS` | `1` | onnxruntime threads per generation |
| `CONCURRENCY` | `2` | Generations at once; more requests queue |
| `MAX_QUEUE` | `32` | Queued requests beyond this get 503 |

## Deploy

Pushing to `main` builds the image with GitHub Actions, pushes it to `ghcr.io/direct-img/auroraimg-6m:main`, then calls the Dokploy deploy webhook (repo secret `DOKPLOY_WEBHOOK_URL`). It runs as the AuroraIMG-6M app in the direct-img Dokploy project.

Run locally:

```sh
pip install -r requirements.txt
AURORA_API_KEY=dev python app.py
curl -H "Authorization: Bearer dev" "localhost:8080/generate?prompt=orange+cat" -o cat.webp
```

## Model and license

`model/` holds an ONNX conversion of [AuroraAI-Research/AuroraIMG-6M](https://huggingface.co/AuroraAI-Research/AuroraIMG-6M) at revision `3d66743`, licensed under the Apache License 2.0 (copy in [`model/LICENSE`](model/LICENSE), attribution and changes in [`model/NOTICE`](model/NOTICE); both ship in the Docker image). The weights are unchanged; they were converted from the original safetensors to ONNX, and the latent un-normalization and output scaling were folded into `decoder.onnx`. Output matches the original pipeline to within 1/255 per pixel.

The model was trained on [CC12M](https://huggingface.co/datasets/pixparse/cc12m-wds), which is distributed as image URLs and captions; the underlying images carry their own licenses.

`scripts/export_onnx.py` regenerates `model/` from the original repo (needs torch, onnx, onnxscript, onnxruntime, safetensors).
