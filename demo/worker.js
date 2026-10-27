// VoxLock in a worker: onnxruntime-web (WASM), two graphs, the same chunk loop as stream_reference in export_onnx.py.
importScripts("ort/ort.wasm.min.js");
ort.env.wasm.wasmPaths = new URL("ort/", self.location.href).href;
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

const SR = 16000, HOP = 128, NFFT = 512, K = 100, NB = 32, HID = 256, LAYERS = 6;
let enrollSess, chunkSess;

async function fetchCached(url, onBytes) {
  let cache = null;
  try { cache = await caches.open("voxlock-models-v2"); } catch (e) {}
  const hit = cache && (await cache.match(url));
  if (hit) { const b = await hit.arrayBuffer(); onBytes(b.byteLength, b.byteLength); return b; }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const total = +res.headers.get("content-length") || 0;
  const reader = res.body.getReader(), parts = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value); got += value.length; onBytes(got, total);
  }
  const buf = new Uint8Array(got);
  let o = 0;
  for (const p of parts) { buf.set(p, o); o += p.length; }
  try { if (cache) await cache.put(url, new Response(buf.slice())); } catch (e) {}
  return buf.buffer;
}

function loudSample(mix, sec = 3, within = 15) {
  const seg = mix.subarray(0, Math.min(mix.length, within * SR));
  const H = 160, n = Math.floor(seg.length / H), w = Math.floor((sec * SR) / H);
  if (n <= w) return seg.slice();
  const e = new Float64Array(n);
  for (let i = 0; i < n; i++) { let s = 0; for (let j = 0; j < H; j++) { const v = seg[i * H + j]; s += v * v; } e[i] = s / H; }
  let cur = 0, best = -1, bi = 0;
  for (let i = 0; i < w; i++) cur += e[i];
  for (let i = 0; ; i++) {
    if (cur > best) { best = cur; bi = i; }
    if (i + w >= n) break;
    cur += e[i + w] - e[i];
  }
  return seg.slice(bi * H, bi * H + sec * SR);
}

function reflectPad(x, p) {
  const n = x.length, y = new Float32Array(n + 2 * p);
  y.set(x, p);
  for (let i = 0; i < p; i++) { y[p - 1 - i] = x[i + 1]; y[p + n + i] = x[n - 2 - i]; }
  return y;
}

async function isolate(mix, sample) {
  const t0 = performance.now();
  const enr = sample || loudSample(mix);
  const eo = await enrollSess.run({ enroll: new ort.Tensor("float32", enr, [1, enr.length]) });
  const voice = { enroll_n: eo.enroll_n, k: eo.k, v: eo.v };
  const x = reflectPad(mix, 256);
  const nfr = 1 + Math.floor((x.length - NFFT) / HOP);
  const out = new Float32Array((nfr - 1) * HOP + NFFT), env = new Float32Array(out.length);
  const w2 = new Float32Array(NFFT);
  for (let i = 0; i < NFFT; i++) { const h = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / NFFT); w2[i] = h * h; }
  let h = new ort.Tensor("float32", new Float32Array(LAYERS * NB * HID), [LAYERS, NB, HID]);
  let c = new ort.Tensor("float32", new Float32Array(LAYERS * NB * HID), [LAYERS, NB, HID]);
  for (let f0 = 0; f0 < nfr; f0 += K) {
    const k = Math.min(K, nfr - f0);
    const seg = x.slice(f0 * HOP, (f0 + k - 1) * HOP + NFFT);
    const r = await chunkSess.run({ wav: new ort.Tensor("float32", seg, [1, seg.length]), ...voice, h, c });
    const fr = r.frames.data;
    for (let j = 0; j < k; j++) {
      const s = (f0 + j) * HOP;
      for (let i = 0; i < NFFT; i++) { out[s + i] += fr[j * NFFT + i]; env[s + i] += w2[i]; }
    }
    h = r.h_out; c = r.c_out;
    postMessage({ type: "progress", done: Math.min(nfr, f0 + k), total: nfr });
  }
  const y = new Float32Array(mix.length);
  for (let i = 0; i < mix.length; i++) { const e = env[i + 256]; y[i] = e > 1e-11 ? out[i + 256] / e : 0; }
  const rms = (a) => { let e = 0; for (let i = 0; i < a.length; i++) e += a[i] * a[i]; return 10 * Math.log10(e / Math.max(1, a.length) + 1e-12); };
  let nans = 0; for (let i = 0; i < y.length; i++) if (!Number.isFinite(y[i])) { nans++; y[i] = 0; }
  const diag = { in_db: +rms(mix).toFixed(1), sample_db: +rms(enr).toFixed(1), out_db: +rms(y).toFixed(1), nans, threads: ort.env.wasm.numThreads };
  return { y, seconds: (performance.now() - t0) / 1000, diag };
}

onmessage = async ({ data }) => {
  try {
    if (data.type === "load") {
      const sizes = {};
      const prog = (name) => (got, total) => { sizes[name] = [got, total]; const g = Object.values(sizes).reduce((a, b) => a + b[0], 0);
        postMessage({ type: "download", got: g, total: data.totalBytes }); };
      // Hugging Face counts a download when config.yaml is fetched (its default query file), so a browser run counts
      // the way transformers.js's config.json fetch does for other models.
      if (data.base.includes("huggingface.co")) fetch(new URL("../../config.yaml", data.base)).catch(() => {});
      const [eb, cb] = await Promise.all([fetchCached(data.base + "enroll_fp16.onnx", prog("e")), fetchCached(data.base + "chunk_fp16.onnx", prog("c"))]);
      const opt = { executionProviders: ["wasm"], graphOptimizationLevel: "all" };
      enrollSess = await ort.InferenceSession.create(eb, opt);
      chunkSess = await ort.InferenceSession.create(cb, opt);
      postMessage({ type: "loaded", threads: ort.env.wasm.numThreads });
    } else if (data.type === "run") {
      const { y, seconds, diag } = await isolate(data.mix, data.sample);
      postMessage({ type: "result", y, seconds, diag }, [y.buffer]);
    }
  } catch (e) {
    postMessage({ type: "error", message: String((e && e.message) || e) });
  }
};
