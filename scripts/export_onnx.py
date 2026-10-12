# Converts AuroraIMG-6M (https://huggingface.co/AuroraAI-Research/AuroraIMG-6M, Apache-2.0) to the ONNX files in model/.
# Only needed to regenerate them. Needs: torch onnx onnxscript onnxruntime safetensors tokenizers numpy
#   python scripts/export_onnx.py
# Weights are read from the .safetensors files only; the repo's pickle .pt files are never downloaded.
import sys, types, json, shutil, urllib.request, importlib.util
from pathlib import Path
import numpy as np, torch, onnxruntime as ort
from safetensors.torch import load_file
torch.set_grad_enabled(False)

REPO, REV = "AuroraAI-Research/AuroraIMG-6M", "3d66743c3139d034b832a670d067221f5d544bbb"
FILES = ["pipeline.py", "tokenizer/tokenizer.json", "vae/config.json"] + [f"{m}/diffusion_pytorch_model.safetensors" for m in ("text_encoder", "unet", "vae")]
ROOT = Path(__file__).resolve().parent.parent
HF, OUT = ROOT / ".hf", ROOT / "model"

for f in FILES:
    p = HF / f
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"https://huggingface.co/{REPO}/resolve/{REV}/{f}", p)

# pipeline.py imports diffusers/transformers only for base classes; these stubs let it load with plain torch
class ConfigMixin:
    def register_to_config(self, **kw): self.__dict__["config"] = types.SimpleNamespace(**kw)
stub = lambda name, **attrs: types.SimpleNamespace(__name__=name, **attrs)
sys.modules |= {"diffusers": stub("diffusers", DiffusionPipeline=object, ModelMixin=type("ModelMixin", (), {}), ConfigMixin=ConfigMixin),
                "diffusers.utils": stub("diffusers.utils", BaseOutput=object), "transformers": stub("transformers", PreTrainedTokenizerFast=object)}
spec = importlib.util.spec_from_file_location("aurora_pipeline", HF / "pipeline.py")
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)

def load(m, sub):
    m.load_state_dict(load_file(HF / sub / "diffusion_pytorch_model.safetensors"), strict=True); return m.eval()
vae_cfg = json.loads((HF / "vae/config.json").read_text())
te, den = load(P.TextEncoder(vocab=4096, d=128, layers=4), "text_encoder"), load(P.Denoiser(), "unet")
vae = load(P.VAE(lat_mu=vae_cfg["lat_mu"], lat_sd=vae_cfg["lat_sd"]), "vae")

class Decode(torch.nn.Module):
    """Un-normalizes latents and maps the VAE output to 0..1 RGB, as the original pipeline does after sampling."""
    def __init__(s):
        super().__init__(); s.vae = vae
        s.register_buffer("sd", torch.tensor(vae_cfg["lat_sd"]).view(1, 8, 1, 1)); s.register_buffer("mu", torch.tensor(vae_cfg["lat_mu"]).view(1, 8, 1, 1))
    def forward(s, z): return (s.vae.decode(z * s.sd + s.mu) / 2 + 0.5).clamp(0, 1)
dec = Decode().eval()

B = torch.export.Dim("batch", min=1, max=64)
tok = torch.randint(3, 4096, (2, 64)); tok[:, 10:] = 0; tok[:, 0] = 1; mask = tok != 0
ctx, x, t = te(tok, mask), torch.randn(2, 8, 16, 16), torch.tensor([500.0, 500.0])
OUT.mkdir(exist_ok=True)
graphs = {
    "text_encoder": (te, (tok, mask), ["tok", "mask"], ["ctx"]),
    "denoiser": (den, (x, t, ctx, mask), ["x", "t", "ctx", "mask"], ["v"]),
    "decoder": (dec, (x,), ["z"], ["image"]),
}
for name, (m, args, ins, outs) in graphs.items():
    shapes = tuple({0: B} for _ in args)
    torch.onnx.export(m, args, OUT / f"{name}.onnx", input_names=ins, output_names=outs, dynamic_shapes=shapes, dynamo=True, external_data=False)
    sess = ort.InferenceSession(OUT / f"{name}.onnx", providers=["CPUExecutionProvider"])
    want = m(*args).numpy()
    got = sess.run(None, {k: a.numpy() for k, a in zip(ins, args)})[0]
    err = float(np.abs(got - want).max())
    print(f"{name}.onnx: {(OUT / f'{name}.onnx').stat().st_size / 1e6:.1f} MB, max abs diff vs torch {err:.2e}")
    assert err < 1e-3, name
shutil.copyfile(HF / "tokenizer/tokenizer.json", OUT / "tokenizer.json")
