"""Word timings for the AMI eval windows (caller vs everyone else), relative to each window start, for score_v2 leak.

    python ami_times.py ami_headset ami_headset2
"""
import json, sys, zipfile
from pathlib import Path
from ami_headset_set import ANNOT, fetch, words, channels

cache = Path("/tmp/ami_annot"); cache.mkdir(exist_ok=True)
fetch(ANNOT, cache / "annot.zip")
z = zipfile.ZipFile(cache / "annot.zip")
for s in sys.argv[1:]:
    meta = json.loads((Path(s) / "meta.json").read_text())
    out = {}
    for m in meta:
        meeting, spk = m["meeting"], m["caller_id"].split("-", 1)[1]
        w0, w1 = m["window"]
        span = lambda ws: [(round(a - w0, 3), round(b - w0, 3)) for a, b, _ in ws if b > w0 and a < w1]
        others = [x for sp in channels(z, meeting) if sp != spk for x in span(words(z, meeting, sp))]
        out[str(m["i"])] = {"dur": round(w1 - w0, 3), "caller": span(words(z, meeting, spk)), "others": others}
    (Path(s) / "ami_times.json").write_text(json.dumps(out))
    print(s, len(out), "windows timed", flush=True)
