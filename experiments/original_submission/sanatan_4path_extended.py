"""
SANATAN 4-Path Power^4 — Extended 10B Token Training
=====================================================
Best ablation result: 4.86 at 1B tokens
6-path went 4.85 → 4.30 with 10B tokens
4-path power^4 could reach ~4.0 or below

Feature map: [1, r, r·cos, r·sin]  (4d features, 64 KB/head)
Sharpening: power^4
Gate: soft leaky (gradient flows through mismatched roots)

Usage: python sanatan_4path_10B.py
"""

import os, math, time, json, random, threading, queue, functools, subprocess, gc
import numpy as np
import jax
import jax.numpy as jnp
from jax import random as jrand
import flax.linen as nn
import flax
import optax
from flax.training import train_state
from tqdm.auto import tqdm

jax.config.update('jax_default_matmul_precision', 'bfloat16')
print(f"JAX {jax.__version__}, devices: {len(jax.devices())}")

GCS_BUCKET = "gs://anonymous-bucket/anonymous"
LOCAL_DIR  = "./ckpt_neurips_suite"

DIM       = 768
SL        = 2048
LR        = 6e-4
WARMUP    = 500
N_LAYERS  = 12
N_HEADS   = 12
STD_FFN   = 2048
TOKENS    = 10_000_000_000  # 10B

def gcs_cp(src, dst):
    try:
        subprocess.run(["gsutil", "-q", "cp", src, dst],
                       check=True, capture_output=True, timeout=300)
    except: pass

def gcs_exists(path):
    try:
        r = subprocess.run(["gsutil", "ls", path], capture_output=True, timeout=60)
        return r.returncode == 0
    except: return False

def save_checkpoint(tag, params, step, tokens, best, metrics, label="latest"):
    os.makedirs(LOCAL_DIR, exist_ok=True)
    meta = {"step": step, "tokens": tokens, "best": best, "met": metrics}
    meta_path = f"{LOCAL_DIR}/{tag}_{label}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    flat = jax.tree.leaves(params)
    npz_path = f"{LOCAL_DIR}/{tag}_{label}.npz"
    np.savez(npz_path, **{f"p_{i}": np.array(v) for i, v in enumerate(flat)})
    gcs_cp(meta_path, f"{GCS_BUCKET}/{tag}_{label}_meta.json")
    gcs_cp(npz_path, f"{GCS_BUCKET}/{tag}_{label}.npz")

def load_checkpoint(tag, state):
    os.makedirs(LOCAL_DIR, exist_ok=True)
    local_meta = f"{LOCAL_DIR}/{tag}_latest_meta.json"
    local_npz = f"{LOCAL_DIR}/{tag}_latest.npz"
    loaded = False
    if gcs_exists(f"{GCS_BUCKET}/{tag}_latest_meta.json"):
        gcs_cp(f"{GCS_BUCKET}/{tag}_latest_meta.json", local_meta)
        gcs_cp(f"{GCS_BUCKET}/{tag}_latest.npz", local_npz)
        loaded = True
    elif os.path.exists(local_meta) and os.path.exists(local_npz):
        loaded = True
    if loaded and os.path.exists(local_meta) and os.path.exists(local_npz):
        try:
            with open(local_meta) as f:
                meta = json.load(f)
            data = np.load(local_npz, allow_pickle=True)
            flat_params = [data[f"p_{i}"] for i in range(len(data.files))]
            tree_struct = jax.tree.structure(state.params)
            if len(flat_params) == len(jax.tree.leaves(state.params)):
                restored = jax.tree.unflatten(tree_struct, flat_params)
                state = state.replace(params=restored)
                step = meta["step"] + 1
                tokens = meta["tokens"]
                best = meta.get("best", float("inf"))
                metrics = meta.get("met", {"train": [], "val": []})
                print(f"  Resumed from step {meta['step']:,} ({tokens / 1e9:.2f}B tok)")
                del data, flat_params
                gc.collect()
                return state, step, tokens, best, metrics
        except Exception as e:
            print(f"  WARNING: checkpoint load failed: {e}")
    return state, 0, 0, float("inf"), {"train": [], "val": []}

class RMSNorm(nn.Module):
    d: int
    @nn.compact
    def __call__(self, x):
        w = self.param('w', nn.initializers.ones, (self.d,))
        var = jnp.mean(x.astype(jnp.float32) ** 2, axis=-1, keepdims=True)
        return x * jax.lax.rsqrt(var + 1e-6).astype(x.dtype) * w


class Sanatan4PathAttn(nn.Module):
    """4-path: [1, r, r·cos, r·sin] with power^4
    Score = 1 + r_q·r_k·(1 + cosΔθ)
    Soft gate: mismatched roots leak -cosΔθ (gradient flows)
    """
    d: int
    n_layers: int
    nh: int = 12

    @nn.compact
    def __call__(self, x):
        B, S, D = x.shape
        hd = D // self.nh
        scale = 1.0 / math.sqrt(hd)
        HP = jax.lax.Precision.HIGHEST

        W_q = self.param('W_q', nn.initializers.normal(stddev=0.02), (D, D))
        W_k = self.param('W_k', nn.initializers.normal(stddev=0.02), (D, D))
        W_v = self.param('W_v', nn.initializers.normal(stddev=0.02), (D, D))
        W_out = self.param('W_out', nn.initializers.normal(stddev=0.02), (D, D))

        omega = self.param('omega', nn.initializers.constant(math.pi / 4), (self.nh, hd))
        lambda_raw = self.param('lambda_raw', nn.initializers.zeros, (self.nh, hd))
        lambda_rot = jnp.tanh(lambda_raw) * 0.01

        q = (x @ W_q).astype(jnp.float32).reshape(B, S, self.nh, hd) * scale
        k = (x @ W_k).astype(jnp.float32).reshape(B, S, self.nh, hd) * scale
        v = (x @ W_v).astype(jnp.float32).reshape(B, S, self.nh, hd)

        theta_q = jnp.pi * (q / (1.0 + jnp.abs(q)))
        theta_k = jnp.pi * (k / (1.0 + jnp.abs(k)))

        root_q = jnp.tanh(10.0 * (jnp.cos(theta_q + omega) - jnp.sin(theta_q + omega)))
        root_k = jnp.tanh(10.0 * (jnp.cos(theta_k + omega) - jnp.sin(theta_k + omega)))

        pos = jnp.arange(S)[None, :, None, None]
        rot_q = theta_q + pos * lambda_rot
        rot_k = theta_k + pos * lambda_rot

        def get_phi(theta, root):
            return jnp.concatenate([
                jnp.ones_like(theta),
                root,
                root * jnp.cos(theta),
                root * jnp.sin(theta),
            ], axis=-1)

        phi_q = get_phi(rot_q, root_q)
        phi_k = get_phi(rot_k, root_k)

        pq = jnp.transpose(phi_q, (0, 2, 1, 3))
        pk = jnp.transpose(phi_k, (0, 2, 1, 3))
        vt = jnp.transpose(v, (0, 2, 1, 3))

        scores = jnp.matmul(pq, jnp.swapaxes(pk, -2, -1), precision=HP)

        hd_f = jnp.float32(hd)
        scores = (scores + hd_f) / (4.0 * hd_f + 1e-6)
        scores = jnp.power(jnp.maximum(scores, 0.0), 4.0)  # power^4

        causal = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
        scores = jnp.where(causal[None, None], scores, 0.0)
        attn = scores / (scores.sum(axis=-1, keepdims=True) + 1e-6)

        out = jnp.matmul(attn, vt, precision=HP)
        out = jnp.transpose(out, (0, 2, 1, 3)).reshape(B, S, D)
        return out @ W_out


class SwiGLUFFN(nn.Module):
    d: int
    h: int
    n_layers: int
    @nn.compact
    def __call__(self, x):
        init = nn.initializers.normal(stddev=0.02)
        init_d = nn.initializers.normal(stddev=0.02 / math.sqrt(2 * self.n_layers))
        gate = nn.Dense(self.h, use_bias=False, kernel_init=init)(x)
        up = nn.Dense(self.h, use_bias=False, kernel_init=init)(x)
        return nn.Dense(self.d, use_bias=False, kernel_init=init_d)(jax.nn.silu(gate) * up)


class Block(nn.Module):
    d: int
    ffn_h: int
    n_layers: int
    nh: int = 12
    @nn.compact
    def __call__(self, x):
        norm_x = RMSNorm(self.d)(x)
        h_attn = Sanatan4PathAttn(self.d, self.n_layers, self.nh)(norm_x)
        x = x + h_attn
        h_ffn = SwiGLUFFN(self.d, self.ffn_h, self.n_layers)(RMSNorm(self.d)(x))
        return x + h_ffn


class CausalLM(nn.Module):
    dim: int
    n_layers: int
    ffn_hidden: int
    seq_len: int
    vocab: int
    nh: int = 12
    @nn.compact
    def __call__(self, ids):
        init = nn.initializers.normal(stddev=0.02)
        tok_emb = self.param('tok_emb', init, (self.vocab, self.dim))
        pos_emb = self.param('pos_emb', nn.initializers.normal(stddev=0.02),
                             (self.seq_len, self.dim))
        x = tok_emb[ids] + pos_emb[:ids.shape[1]]
        for i in range(self.n_layers):
            block_fn = nn.remat(Block)(self.dim, self.ffn_hidden, self.n_layers, self.nh)
            x = block_fn(x)
        x = RMSNorm(self.dim)(x)
        logits = x @ tok_emb.T
        return logits


def causal_lm_loss(params, model, batch):
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    logits = model.apply(params, inputs)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    return jnp.mean(loss), loss.shape[0] * loss.shape[1]


DATASETS_CONFIG = {
    "fineweb":     ("HuggingFaceFW/fineweb",      None,            "text",    0.52),
    "fineweb_edu": ("HuggingFaceFW/fineweb-edu",   "sample-100BT", "text",    0.06),
    "openwebmath": ("open-web-math/open-web-math",  None,           "text",    0.015),
    "finemath":    ("HuggingFaceTB/finemath",       "finemath-3plus","text",   0.015),
    "culturax_zh": ("uonlp/CulturaX",              "zh",            "text",    0.11),
    "starcoder":   ("bigcode/starcoderdata",        "default",       "content", 0.28),
}
FALLBACKS = {
    "culturax_zh": ("wikimedia/wikipedia", "20231101.zh", "text", 0.11),
    "starcoder":   ("code_search_net",     "all",         "whole_func_string", 0.28),
}

class DL:
    def __init__(self, tok, sl, bs):
        from datasets import load_dataset
        self.tok, self.sl, self.bs = tok, sl, bs
        self.q = queue.Queue(maxsize=32)
        self.stop = threading.Event()
        self.src, self.wt = [], []
        print("  Datasets:")
        for nm, (p, sub, k, w) in DATASETS_CONFIG.items():
            try:
                kw = {"split": "train", "streaming": True}
                if sub: kw["name"] = sub
                ds = load_dataset(p, **kw)
                self.src.append({"ds": ds, "it": iter(ds), "k": k, "p": p, "s": sub})
                self.wt.append(w)
                print(f"    {nm:<15} {w:>5.1%}")
            except:
                if nm in FALLBACKS:
                    fp, fsub, fk, fw = FALLBACKS[nm]
                    try:
                        fkw = {"split": "train", "streaming": True}
                        if fsub: fkw["name"] = fsub
                        ds = load_dataset(fp, **fkw)
                        self.src.append({"ds": ds, "it": iter(ds), "k": fk, "p": fp, "s": fsub})
                        self.wt.append(fw)
                        print(f"    {nm:<15} {w:>5.1%}  (fallback: {fp})")
                    except:
                        print(f"    {nm}: FAILED")
                else:
                    print(f"    {nm}: FAILED")
        tw = sum(self.wt)
        self.wt = [w / tw for w in self.wt]
        print("  Val...", end=" ", flush=True)
        self.val = self._mkv(20)
        print(f"{len(self.val)} batches")
        threading.Thread(target=self._w, daemon=True).start()

    def _t(self, s):
        try: return next(s["it"]).get(s["k"], "")
        except StopIteration:
            from datasets import load_dataset
            kw = {"split": "train", "streaming": True}
            if s["s"]: kw["name"] = s["s"]
            s["ds"] = load_dataset(s["p"], **kw)
            s["it"] = iter(s["ds"])
            return next(s["it"]).get(s["k"], "")

    def _fill(self, buf, n):
        while len(buf) < n:
            idx = random.choices(range(len(self.src)), weights=self.wt, k=1)[0]
            txt = self._t(self.src[idx])
            if txt and len(txt) > 20:
                ids = self.tok.encode(txt)
                buf.extend(ids)

    def _mkb(self):
        buf = []
        self._fill(buf, self.bs * self.sl + self.sl)
        rows = []
        for i in range(self.bs):
            start = random.randint(0, len(buf) - self.sl)
            rows.append(buf[start:start + self.sl])
        return np.array(rows, dtype=np.int32)

    def _mkv(self, n):
        return [self._mkb() for _ in range(n)]

    def _w(self):
        while not self.stop.is_set():
            try:
                self.q.put(self._mkb(), timeout=1)
            except queue.Full:
                continue

    def get(self):
        return self.q.get()

    def shutdown(self):
        self.stop.set()


def train_model(model, tag, loader, steps, micro,
                save_every=500, val_every=200):
    os.makedirs(LOCAL_DIR, exist_ok=True)
    n_dev = len(jax.devices())
    def lr_fn(step):
        return LR * jnp.minimum(1.0, (step + 1) / WARMUP)
    rng = jrand.PRNGKey(42)
    rng, init_rng = jrand.split(rng)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=lr_fn, b1=0.9, b2=0.95, weight_decay=0.1)
    )
    dummy = jnp.zeros((2, SL), dtype=jnp.int32)
    params = model.init(init_rng, dummy)
    total_p = sum(x.size for x in jax.tree.leaves(params))
    emb_p = params['params']['tok_emb'].size + params['params']['pos_emb'].size
    print(f"  {tag}: {total_p / 1e6:.0f}M total, {(total_p - emb_p) / 1e6:.2f}M non-emb")

    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer
    )
    state, start_step, total_tokens, best_val, metrics = load_checkpoint(tag, state)
    tps = micro * SL
    bf16_params = jax.tree.map(
        lambda x: x.astype(jnp.bfloat16) if (x.dtype == jnp.float32 and x.size > 64) else x,
        state.params
    )
    state = state.replace(params=bf16_params)
    del bf16_params
    gc.collect()
    state = flax.jax_utils.replicate(state)

    @functools.partial(jax.pmap, axis_name='batch')
    def train_step(state, batch):
        def loss_fn(params):
            return causal_lm_loss(params, model, batch)
        (loss, nm), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        grads = jax.lax.pmean(grads, axis_name='batch')
        state = state.apply_gradients(grads=grads)
        return state, loss, nm

    @functools.partial(jax.pmap, axis_name='batch')
    def val_step(params, batch):
        return causal_lm_loss(params, model, batch)

    print(f"\n  Training {tag}: {steps:,} steps, {tps:,} tok/step")
    print(f"  {n_dev} TPU chips, micro={micro} ({micro // n_dev}/chip)")

    pbar = tqdm(range(start_step, steps), initial=start_step, total=steps, desc=tag)
    for step in pbar:
        t0 = time.time()
        batch_np = loader.get()
        batch = jnp.array(batch_np)
        batch = batch.reshape(n_dev, batch.shape[0] // n_dev, batch.shape[1])
        state, loss, nm = train_step(state, batch)
        avg_loss = float(loss[0])
        total_tokens += int(np.prod(batch_np.shape))
        elapsed = time.time() - t0
        pbar.set_postfix(loss=f"{avg_loss:.3f}", tps=f"{tps / elapsed / 1e3:.1f}k",
                         tok=f"{total_tokens / 1e9:.2f}B")
        if step % 25 == 0:
            metrics["train"].append((step, avg_loss))
        if step > 0 and step % 100 == 0:
            eta = (steps - step) * elapsed / 3600
            print(f"  [{step:>7,}/{steps:,}] loss={avg_loss:.3f} "
                  f"tps={tps / elapsed / 1e3:.1f}k tok={total_tokens / 1e9:.2f}B "
                  f"ETA={eta:.0f}h", flush=True)
        if step > 0 and step % save_every == 0:
            params_single = jax.tree.map(lambda x: x[0], state.params)
            save_checkpoint(tag, params_single, step, total_tokens, best_val, metrics, "latest")
            del params_single
            gc.collect()
        if step > 0 and step % val_every == 0:
            vs, vn = 0.0, 0
            for vb_np in loader.val:
                vb = jnp.array(vb_np)
                vb = vb.reshape(n_dev, vb.shape[0] // n_dev, vb.shape[1])
                vl, vnm = val_step(state.params, vb)
                vnm_val = int(vnm[0])
                if vnm_val > 0:
                    vs += float(vl[0]) * vnm_val
                    vn += vnm_val
            val_loss = vs / max(vn, 1)
            metrics["val"].append((step, val_loss))
            print(f"\n  Step {step}: val={val_loss:.4f} train={avg_loss:.3f} "
                  f"tok={total_tokens / 1e9:.2f}B")
            if val_loss < best_val:
                best_val = val_loss
                params_single = jax.tree.map(lambda x: x[0], state.params)
                save_checkpoint(tag, params_single, step, total_tokens, val_loss, metrics, "best")
                del params_single
                gc.collect()

    params_single = jax.tree.map(lambda x: x[0], state.params)
    save_checkpoint(tag, params_single, steps, total_tokens, best_val, metrics, "final")
    save_checkpoint(tag, params_single, steps, total_tokens, best_val, metrics, "latest")
    del params_single
    metrics_path = f"{LOCAL_DIR}/{tag}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)
    gcs_cp(metrics_path, f"{GCS_BUCKET}/{tag}_metrics.json")

    # Update results_neurips.json
    results_file = "results_neurips.json"
    all_results = {}
    if os.path.exists(results_file):
        with open(results_file) as f:
            all_results = json.load(f)
    all_results["sanatan_4path_p4_10B"] = {
        "val": best_val,
        "desc": "4-path power^4 extended 10B tokens (64 KB/head)",
        "tokens": TOKENS,
        "completed": True,
    }
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    gcs_cp(results_file, f"{GCS_BUCKET}/results_neurips.json")

    del state
    gc.collect()
    print(f"\n{'=' * 60}")
    print(f"  4-PATH POWER^4 10B RESULT: val = {best_val:.4f}")
    print(f"  Compare: 6-path power^2 10B = 4.30")
    print(f"  Compare: softmax 3B = 3.27")
    print(f"  State: 64 KB/head (vs 96 KB for 6-path)")
    print(f"{'=' * 60}")
    return best_val, metrics


def main():
    from transformers import AutoTokenizer
    tok = None
    for n in ["GSAI-ML/LLaDA-8B-Instruct", "GSAI-ML/LLaDA-8B-Base",
              "mistralai/Mistral-7B-v0.1", "gpt2"]:
        try:
            tok = AutoTokenizer.from_pretrained(n, trust_remote_code=True)
            print(f"Tokenizer: {n} (vocab={len(tok)})")
            break
        except Exception as e:
            print(f"  Tokenizer {n} failed: {e}")
    if tok is None:
        raise RuntimeError("No tokenizer!")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    VOCAB = len(tok)
    n_dev = len(jax.devices())
    MICRO = 32
    MICRO = MICRO - (MICRO % n_dev)
    STEPS = int(TOKENS / (MICRO * SL))

    print(f"\n{'=' * 60}")
    print(f"  SANATAN 4-PATH POWER^4 — 10B EXTENDED TRAINING")
    print(f"  Feature map: [1, r, r·cos, r·sin]")
    print(f"  State: 4d × d = 64 KB/head (33% less than 6-path)")
    print(f"  {n_dev} TPU chips, {STEPS:,} steps, 10B tokens")
    print(f"  ~55 hours estimated")
    print(f"{'=' * 60}")

    model = CausalLM(dim=DIM, n_layers=N_LAYERS, ffn_hidden=STD_FFN,
                      seq_len=SL, vocab=VOCAB, nh=N_HEADS)

    loader = DL(tok, SL, MICRO)
    best_val, metrics = train_model(
        model, "184M_sanatan_4path_p4_10B", loader, STEPS, MICRO,
        save_every=500, val_every=200
    )
    loader.shutdown()


if __name__ == "__main__":
    main()
