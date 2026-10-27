// VoxLock demo widget: synced on/off playback of precomputed clips, and record -> isolate in a worker.
// Mounts into <div id="voxlock-demo">; every asset is loaded relative to this file.
const SR = 16000, MODEL_BYTES = 49841922, MAX_REC_S = 15, MAX_UPLOAD_S = 60;
const HF_BASE = "https://huggingface.co/arbazsiddiqui/voxlock/resolve/main/onnx/v2/";
const MODEL_BASE = new URLSearchParams(location.search).get("models") || HF_BASE;
const asset = (p) => new URL(p, import.meta.url).href;
function mount(root) {
root.classList.add("vl");
const recordOnly = root.dataset.only === "record";   // the Hugging Face Space shows only the record section
root.innerHTML = `
${recordOnly ? "" : `
  <div class="vl-section">
    <h3 class="vl-h">Listen</h3>
    <p class="vl-note">A caller with a second person talking in the room. Flip the switch while it plays: off is the call as recorded, on is VoxLock. Below each clip is what speech-to-text hears.</p>
    <div class="vl-layout"><ol class="vl-clips" data-clips></ol><div class="vl-player" data-player-a></div></div>
    <p class="vl-fine">Simulated calls: two LibriSpeech test-clean speakers, never used in training, in a simulated room. Transcripts by Whisper small.</p>
  </div>`}
  <div class="vl-section">
    <h3 class="vl-h">Try your own voice</h3>
    <p class="vl-note">Record up to 15 seconds or upload a file. Mix in a second talker, or play a video near the mic while you talk. VoxLock runs in this tab; nothing is uploaded.</p>
    <div class="vl-rec">
      <div class="vl-rec-row">
        <button class="vl-btn vl-primary" data-rec>Record</button>
        <label class="vl-btn vl-quiet">Upload audio<input data-file type="file" accept="audio/*" hidden></label>
        <label class="vl-check"><input type="checkbox" data-talker checked> Mix in a second talker</label>
        <select data-talker-level aria-label="Second talker level">
          <option value="6" selected>6 dB quieter than you</option>
          <option value="0">as loud as you</option>
          <option value="12">12 dB quieter than you</option>
        </select>
      </div>
      <div class="vl-status" data-status></div>
      <div class="vl-bar" data-bar hidden><div></div></div>
      <div class="vl-tip" data-diag></div>
    </div>
    <div class="vl-player" data-player-b hidden></div>
  </div>`;
let ctx;
const audioCtx = () => (ctx ||= new (window.AudioContext || window.webkitAudioContext)());
const $ = (s) => root.querySelector(s);
const fmt = (t) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;
const css = (name) => getComputedStyle(root).getPropertyValue(name).trim();

class Player {
  constructor(el) {
    this.el = el;
    this.on = true; this.playing = false; this.offset = 0;
    el.innerHTML = `
      <p class="vl-p-title"></p><p class="vl-p-meta"></p>
      <div class="vl-wave"><canvas></canvas></div>
      <div class="vl-controls">
        <button class="vl-btn vl-primary" data-play>Play</button>
        <button class="vl-switch" role="switch" aria-checked="true"><span class="track"></span><b>VoxLock on</b></button>
        <span class="vl-time">0:00 / 0:00</span>
      </div>
      <div class="vl-scores" hidden></div>`;
    this.canvas = el.querySelector("canvas");
    this.btn = el.querySelector("[data-play]");
    this.sw = el.querySelector(".vl-switch");
    this.time = el.querySelector(".vl-time");
    this.btn.onclick = () => (this.playing ? this.pause() : this.play());
    this.sw.onclick = () => this.setOn(!this.on);
    const wave = el.querySelector(".vl-wave");
    wave.onpointerdown = (e) => {
      if (!this.off) return;
      const r = wave.getBoundingClientRect();
      this.seek(Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)) * this.off.duration);
    };
    new ResizeObserver(() => this.draw()).observe(wave);
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => this.draw());
  }
  set({ title, meta, off, on, scores }) {
    this.stop();
    this.off = off; this.onBuf = on; this.offset = 0;
    this.el.querySelector(".vl-p-title").textContent = title;
    this.el.querySelector(".vl-p-meta").textContent = meta;
    const sc = this.el.querySelector(".vl-scores");
    sc.hidden = !scores; sc.innerHTML = scores || "";
    this.peaks = [this.envelope(off), this.envelope(on)];
    this.draw(); this.tick();
  }
  envelope(buf, n = 600) {
    const d = buf.getChannelData(0), step = Math.max(1, Math.floor(d.length / n)), out = new Float32Array(n);
    for (let i = 0; i < n; i++) { let m = 0; for (let j = i * step; j < Math.min(d.length, (i + 1) * step); j++) m = Math.max(m, Math.abs(d[j])); out[i] = m; }
    return out;
  }
  draw() {
    if (!this.peaks) return;
    const c = this.canvas, dpr = devicePixelRatio || 1, w = c.clientWidth, h = c.clientHeight;
    c.width = w * dpr; c.height = h * dpr;
    const g = c.getContext("2d"); g.scale(dpr, dpr); g.clearRect(0, 0, w, h);
    const top = Math.max(...this.peaks[0], ...this.peaks[1], 1e-6), FLOOR = 54;  // decibel scale, so quiet background voices show
    const bars = (p, color) => {
      g.fillStyle = color;
      const bw = w / p.length;
      for (let i = 0; i < p.length; i++) {
        const db = 20 * Math.log10(p[i] / top + 1e-9), a = Math.max(0.75, Math.max(0, 1 + db / FLOOR) * (h / 2 - 2));
        g.fillRect(i * bw, h / 2 - a, Math.max(1, bw - 0.6), 2 * a);
      }
    };
    bars(this.peaks[0], css("--wave-off"));
    bars(this.peaks[1], css("--wave-on"));
    if (this.off) {
      const x = (this.pos() / this.off.duration) * w;
      g.fillStyle = css("--fg"); g.fillRect(x - 0.75, 0, 1.5, h);
    }
  }
  pos() { return this.playing ? Math.min(this.off.duration, this.offset + audioCtx().currentTime - this.t0) : this.offset; }
  play() {
    if (!this.off) return;
    const a = audioCtx(); a.resume();
    if (this.offset >= this.off.duration - 0.05) this.offset = 0;
    this.gOff = a.createGain(); this.gOn = a.createGain();
    this.gOff.gain.value = this.on ? 0 : 1; this.gOn.gain.value = this.on ? 1 : 0;
    this.gOff.connect(a.destination); this.gOn.connect(a.destination);
    this.sOff = a.createBufferSource(); this.sOff.buffer = this.off; this.sOff.connect(this.gOff);
    this.sOn = a.createBufferSource(); this.sOn.buffer = this.onBuf; this.sOn.connect(this.gOn);
    const when = a.currentTime + 0.03;
    this.sOff.start(when, this.offset); this.sOn.start(when, this.offset);
    this.t0 = when; this.playing = true; this.btn.textContent = "Pause";
    this.sOn.onended = () => { if (this.playing && this.pos() >= this.off.duration - 0.05) { this.stop(); this.offset = 0; this.draw(); this.tick(); } };
    const loop = () => { if (!this.playing) return; this.draw(); this.tick(); requestAnimationFrame(loop); };
    loop();
  }
  stop() {
    if (this.playing) { try { this.sOff.stop(); this.sOn.stop(); } catch (e) {} this.sOn.onended = null; }
    this.playing = false; this.btn.textContent = "Play";
  }
  pause() { this.offset = this.pos(); this.stop(); this.draw(); this.tick(); }
  seek(t) { const was = this.playing; this.stop(); this.offset = t; if (was) this.play(); else { this.draw(); this.tick(); } }
  setOn(v) {
    this.on = v;
    this.sw.setAttribute("aria-checked", String(v));
    this.sw.querySelector("b").textContent = v ? "VoxLock on" : "VoxLock off";
    if (this.playing) {
      const t = audioCtx().currentTime;
      this.gOn.gain.setTargetAtTime(v ? 1 : 0, t, 0.012); this.gOff.gain.setTargetAtTime(v ? 0 : 1, t, 0.012);
    }
  }
  tick() { if (this.off) this.time.textContent = `${fmt(this.pos())} / ${fmt(this.off.duration)}`; }
}

async function decode(url) {
  const r = await fetch(url);
  return audioCtx().decodeAudioData(await r.arrayBuffer());
}

// ---------- precomputed clips ----------
const pa = recordOnly ? null : new Player($("[data-player-a]"));
let clips = [];
async function pick(i) {
  root.querySelectorAll(".vl-clip").forEach((b, j) => b.setAttribute("aria-current", String(j === i)));
  const c = clips[i];
  const [off, on] = await Promise.all([decode(asset(`audio/${c.id}_off.mp3`)), decode(asset(`audio/${c.id}_on.mp3`))]);
  const esc = (t) => t.replace(/&/g, "&amp;").replace(/</g, "&lt;");
  const scores = `<div class="vl-heard"><span>Speech-to-text hears, VoxLock off</span><p>${esc(c.heard_off)}</p></div><div class="vl-heard on"><span>VoxLock on</span><p>${esc(c.heard_on)}</p></div>`;
  pa.set({ title: c.title, meta: c.source, off, on, scores });
}
if (!recordOnly) fetch(asset("clips.json"), { cache: "no-cache" }).then((r) => r.json()).then((rows) => {
  clips = rows;
  $("[data-clips]").innerHTML = rows.map((c, i) => `<li><button class="vl-clip" aria-current="false"><span class="n">${String(i + 1).padStart(2, "0")}</span><b>${c.title}</b><small>${Math.round(c.dur_s)} s</small></button></li>`).join("");
  root.querySelectorAll(".vl-clip").forEach((b, i) => (b.onclick = () => pick(i)));
  pick(0);
});

// ---------- record and isolate ----------
const status = (t) => ($("[data-status]").textContent = t);
const bar = (f) => { $("[data-bar]").hidden = f == null; if (f != null) $("[data-bar]").firstElementChild.style.width = `${Math.round(f * 100)}%`; };
let worker, loaded = false, talker16k;

async function decode16k(bytes) {
  const b = await new OfflineAudioContext(1, SR, SR).decodeAudioData(bytes);  // decodes straight to 16 kHz
  const x = new Float32Array(b.length);
  for (let ch = 0; ch < b.numberOfChannels; ch++) { const d = b.getChannelData(ch); for (let i = 0; i < x.length; i++) x[i] += d[i] / b.numberOfChannels; }
  return x;
}
function frameDb(x, H = 160) {
  const n = Math.floor(x.length / H), f = new Float32Array(n);
  for (let i = 0; i < n; i++) { let s = 0; for (let j = 0; j < H; j++) s += x[i * H + j] ** 2; f[i] = 10 * Math.log10(s / H + 1e-12); }
  return f;
}
function level(x, mask) {
  const f = frameDb(x);
  if (!mask) { const q = [...f].sort((a, b) => a - b)[Math.floor(f.length * 0.8)]; mask = f.map((v) => v > q); }
  let s = 0, k = 0;
  for (let i = 0; i < f.length; i++) if (mask[i]) { s += 10 ** (f[i] / 10); k++; }
  return [10 * Math.log10(s / Math.max(1, k) + 1e-12), mask];
}
// The caller's first `sec` seconds of speech (10 ms frames at least 10 dB above the quietest frame of the last 2 s,
// digital silence aside), as VoxLock.live() collects them: the voice sample, and the point after which the second
// talker joins.
function firstSpeech(x, sec = 3) {
  const H = 160, f = frameDb(x, H), need = sec * SR / H, win = [], keep = [];
  for (let i = 0; i < f.length && keep.length < need; i++) {
    if (f[i] > -90) { win.push(f[i]); if (win.length > 200) win.shift(); }
    if (win.length >= 20 && f[i] >= Math.min(...win) + 10) keep.push(i);
  }
  if (keep.length < need / 2) return null;
  const sample = new Float32Array(keep.length * H);
  keep.forEach((i, k) => sample.set(x.subarray(i * H, i * H + H), k * H));
  return { sample, end: (keep[keep.length - 1] + 1) * H };
}
async function addTalker(x, snrDb, lag) {
  talker16k ||= await decode16k(await (await fetch(asset("talker.mp3"))).arrayBuffer());
  const [lx] = level(x), [lt] = level(talker16k);
  const g = 10 ** ((lx - snrDb - lt) / 20), y = new Float32Array(x.length);
  let peak = 0;
  for (let i = 0; i < x.length; i++) { y[i] = x[i] + (i >= lag ? g * talker16k[(i - lag) % talker16k.length] : 0); peak = Math.max(peak, Math.abs(y[i])); }
  if (peak > 0.97) for (let i = 0; i < y.length; i++) y[i] *= 0.97 / peak;
  return y;
}
function normalize(x, target = -20) {
  const [lx] = level(x);
  let g = 10 ** ((target - lx) / 20), peak = 0;
  for (let i = 0; i < x.length; i++) peak = Math.max(peak, Math.abs(x[i]) * g);
  if (peak > 0.97) g *= 0.97 / peak;
  const y = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) y[i] = x[i] * g;
  return y;
}
function wavBlob(x) {
  const b = new DataView(new ArrayBuffer(44 + x.length * 2)), w = (o, t) => [...t].forEach((c, i) => b.setUint8(o + i, c.charCodeAt(0)));
  w(0, "RIFF"); b.setUint32(4, 36 + x.length * 2, true); w(8, "WAVEfmt "); b.setUint32(16, 16, true); b.setUint16(20, 1, true); b.setUint16(22, 1, true);
  b.setUint32(24, SR, true); b.setUint32(28, SR * 2, true); b.setUint16(32, 2, true); b.setUint16(34, 16, true); w(36, "data"); b.setUint32(40, x.length * 2, true);
  for (let i = 0; i < x.length; i++) b.setInt16(44 + i * 2, Math.max(-1, Math.min(1, x[i])) * 32767, true);
  return new Blob([b], { type: "audio/wav" });
}
function bufferOf(x) { const b = audioCtx().createBuffer(1, x.length, SR); b.copyToChannel(x, 0); return b; }

function ensureWorker() {
  if (worker) return Promise.resolve();
  worker = new Worker(asset("worker.js?v=" + Date.now()));
  return new Promise((res, rej) => {
    worker.onmessage = ({ data }) => {
      if (data.type === "download") { status(`Downloading the model, once: ${(data.got / 1e6).toFixed(0)} of ${(data.total / 1e6).toFixed(0)} MB`); bar(data.got / data.total); }
      if (data.type === "loaded") { loaded = true; worker.threads = data.threads; res(); }
      if (data.type === "error") { worker = null; rej(new Error(data.message)); }
    };
    worker.onerror = (e) => { e.preventDefault(); worker = null; rej(new Error(e.message || "The model could not be loaded in this tab.")); };
    worker.postMessage({ type: "load", base: MODEL_BASE, totalBytes: MODEL_BYTES });
  });
}
function runWorker(mix, sample) {
  return new Promise((res, rej) => {
    worker.onmessage = ({ data }) => {
      if (data.type === "progress") { status(`Isolating: ${Math.round((100 * data.done) / data.total)}%`); bar(data.done / data.total); }
      if (data.type === "result") res(data);
      if (data.type === "error") rej(new Error(data.message));
    };
    worker.postMessage({ type: "run", mix: mix.slice(), sample: sample && sample.slice() });
  });
}

let busy = false;
async function process(x16, label) {  // x16: 16 kHz mono
  if (busy) return;
  busy = true; $("[data-rec]").disabled = true;
  try {
    let peak0 = 0;
    for (let i = 0; i < x16.length; i++) peak0 = Math.max(peak0, Math.abs(x16[i]));
    if (peak0 < 1e-3) { status("No sound reached the microphone. Check the input device in your system sound settings and try again."); return; }
    x16 = normalize(x16);  // the browser's gain control is off, so a laptop mic records very quietly
    let mix = x16;
    const first = firstSpeech(x16);   // the caller is heard alone first, as on a real call
    if ($("[data-talker]").checked) mix = await addTalker(x16, +$("[data-talker-level]").value, first ? first.end : 3 * SR);
    if (!loaded) { status("Loading the model..."); await ensureWorker(); }
    status("Isolating: 0%"); bar(0);
    const { y, seconds, diag } = await runWorker(mix, first && first.sample);
    console.log("[voxlock]", diag);
    const [lx, mask] = level(mix), [ly] = level(y, mask), g = -(10 ** ((lx - ly) / 20));   // the network returns the caller inverted
    let peak = 0; for (let i = 0; i < y.length; i++) { y[i] *= g; peak = Math.max(peak, Math.abs(y[i])); }
    if (peak > 0.97) for (let i = 0; i < y.length; i++) y[i] *= 0.97 / peak;
    bar(null);
    status(`Done: ${(mix.length / SR).toFixed(1)} s of audio in ${seconds.toFixed(1)} s, on ${worker.threads} CPU thread${worker.threads > 1 ? "s" : ""} in this tab.`);
    $("[data-diag]").innerHTML = `Input ${diag.in_db} dB · voice sample ${diag.sample_db} dB · output ${diag.out_db} dB${diag.nans ? ` · ${diag.nans} invalid samples` : ""} · <a href="${URL.createObjectURL(wavBlob(mix))}" download="voxlock-input.wav">save input</a> · <a href="${URL.createObjectURL(wavBlob(y))}" download="voxlock-output.wav">save output</a>`;
    $("[data-player-b]").hidden = false;
    pb.set({ title: label, meta: $("[data-talker]").checked ? "With a second talker mixed in" : "As recorded", off: bufferOf(mix), on: bufferOf(y) });
    pb.setOn(true);
  } catch (e) {
    bar(null); status(`Something went wrong: ${e.message}`);
  } finally {
    busy = false; $("[data-rec]").disabled = false;
  }
}
const pb = new Player($("[data-player-b]"));
window.__voxlock = { pa, pb };  // handle for debugging in the console

let rec = null, recTimer;
$("[data-rec]").onclick = async () => {
  if (rec) { rec.stop(); return; }
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 } });
  } catch (e) { status("Microphone access was blocked. Allow it in the address bar, or upload a file instead."); return; }
  audioCtx().resume();
  const chunks = [];
  rec = new MediaRecorder(stream);
  rec.ondataavailable = (e) => chunks.push(e.data);
  rec.onstop = async () => {
    clearInterval(recTimer); bar(null); src.disconnect(); stream.getTracks().forEach((t) => t.stop());
    $("[data-rec]").textContent = "Record"; $("[data-rec]").classList.remove("live"); rec = null;
    try {
      const x = await decode16k(await new Blob(chunks).arrayBuffer());
      if (x.length < 4 * SR) { status("That was under 4 seconds. Record a little longer: the model needs a few seconds of your voice."); return; }
      process(x, "Your recording");
    } catch (e) { status(`Could not read the recording: ${e.message}`); }
  };
  rec.start();
  const t0 = Date.now();
  const an = audioCtx().createAnalyser(); an.fftSize = 2048;
  const src = audioCtx().createMediaStreamSource(stream); src.connect(an);
  const buf = new Float32Array(an.fftSize);
  $("[data-rec]").textContent = "Stop"; $("[data-rec]").classList.add("live");
  status(`Recording... 0 / ${MAX_REC_S} s`); bar(0);
  recTimer = setInterval(() => {
    const s = (Date.now() - t0) / 1000;
    an.getFloatTimeDomainData(buf);
    let e = 0; for (const v of buf) e += v * v;
    const db = 10 * Math.log10(e / buf.length + 1e-12);
    bar(Math.max(0, Math.min(1, (db + 70) / 55)));  // live mic level
    status(`Recording... ${Math.floor(s)} / ${MAX_REC_S} s`);
    if (s >= MAX_REC_S && rec) rec.stop();
  }, 100);
};
$("[data-file]").onchange = async (e) => {
  const f = e.target.files[0]; e.target.value = "";
  if (!f) return;
  try {
    let x = await decode16k(await f.arrayBuffer());
    const long = x.length > MAX_UPLOAD_S * SR;
    if (long) x = x.slice(0, MAX_UPLOAD_S * SR);
    if (x.length < 4 * SR) { status("That file is under 4 seconds. Use a longer one."); return; }
    process(x, f.name + (long ? ` (first ${MAX_UPLOAD_S} s)` : ""));
  } catch (err) { status(`Could not read that file: ${err.message}`); }
};
document.addEventListener("astro:before-swap", () => { pa?.stop(); pb.stop(); if (rec) rec.stop(); }, { once: true });
}

// Mount on first load and again after client-side navigation (the site's router swaps pages without a reload).
const tryMount = () => {
  const el = document.getElementById("voxlock-demo");
  if (!el || el.dataset.mounted) return;
  // The page's isolation headers (needed for WASM threads) only apply on a full load, not after client-side
  // navigation, so reload once where the browser supports them.
  if (!self.crossOriginIsolated && location.hostname.endsWith("arbazsiddiqui.me") && "credentialless" in HTMLIFrameElement.prototype) {
    try {
      if (!sessionStorage.getItem("voxlock-reloaded")) { sessionStorage.setItem("voxlock-reloaded", "1"); location.reload(); return; }
    } catch (e) {}
  }
  el.dataset.mounted = "1"; mount(el);
};
tryMount();
document.addEventListener("astro:page-load", tryMount);
