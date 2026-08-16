"""The atlas viewer — a self-contained page for comparing J-space across models.

This is the recreation of what `eliebak.com/viz/jspace-open` shows. Its
frontend was never published, so this is built from the other direction: from
the data schema and the frozen analysis rules, both of which are public. It
renders anything :mod:`jlens.atlas` produces — a Pythia checkpoint sweep run
here, a clone of the open campaign's 77 lenses, or both in one page.

Three views, because there are three questions:

**Gallery** — every lens as a CKA thumbnail. This is the "comparisons across
models" view, and the point of it is that you can see at a glance whether two
unrelated families lay their structure out the same way.

**Detail** — one model, full size, on the relative-depth axis with its block
boundaries drawn. Depth is always plotted as ``rho = l/(L-1)`` rather than as a
layer index, which is the whole reason a 6-layer model can be put next to a
66-layer one.

**Training** — a metric against training. The open campaign had at most a dozen
checkpoints per model; a Pythia sweep has 154 per size on identical data order,
which turns a scatter of snapshots into a curve. The x-axis offers tokens and
compute as well as step, because step is a unit of optimizer rather than of
learning, and two sizes at the same step have bought very different amounts.

Everything is inlined — no CDN, no build step, no network at view time. The
heatmaps ship as base64 uint8: CKA lives in [0, 1], so a byte per cell is finer
than the display, and it keeps a 154-checkpoint atlas to a few megabytes rather
than tens.

The instrument is part of the picture, not a footnote. Entries whose fitted
lens was no better than the logit lens are struck through and excluded from the
training curves, because their geometry describes the lens rather than the
model, and a grid that renders them identically to real results invites exactly
the wrong reading.
"""

from __future__ import annotations

import base64
import json
import os
from collections.abc import Sequence

import numpy as np

from jlens.atlas import Atlas, json_safe, read_entry


def _quantize(C: np.ndarray) -> str:
    """CKA matrix -> base64 uint8, row-major.

    CKA is bounded in [0, 1] by construction, so a linear byte quantization is
    exact to 1/255 — finer than a screen pixel resolves, and ~8x smaller than
    shipping float32.
    """
    A = np.clip(np.asarray(C, dtype=np.float64), 0.0, 1.0)
    return base64.b64encode(np.round(A * 255).astype(np.uint8).tobytes()).decode()


def _payload(atlas: Atlas, matrices: dict[str, np.ndarray]) -> dict:
    """Manifest plus the quantized heatmaps, ready to inline.

    Keyed by :attr:`~jlens.atlas.AtlasEntry.key`, not by label — two families
    can publish the same ref name, and keying by label silently gives them one
    shared thumbnail.
    """
    manifest = atlas.manifest()
    grids = {}
    for entry in atlas.entries:
        C = matrices.get(entry.key)
        if C is None:
            continue
        grids[entry.key] = {"n": int(np.shape(C)[0]), "d": _quantize(C)}
    manifest["grids"] = grids
    return manifest


def collect_matrices(
    atlas: Atlas, search_dirs: Sequence[str] = ()
) -> dict[str, np.ndarray]:
    """Load each entry's CKA matrix, keyed by entry key.

    Prefers the entry's recorded ``npz_path`` and only falls back to searching
    for ``cka_<label>.npz``. Searching alone is not enough: labels collide
    across families, so the first file found would win for every entry sharing
    that label.

    Entries whose matrix cannot be found are kept in the manifest and render
    without a thumbnail — dropping them would quietly shrink the atlas.
    """
    import glob

    by_label: dict[str, list[str]] = {}
    for root in search_dirs:
        for path in glob.glob(os.path.join(root, "**", "cka_*.npz"), recursive=True):
            label = os.path.basename(path)[len("cka_"):-len(".npz")]
            by_label.setdefault(label, []).append(path)

    found: dict[str, np.ndarray] = {}
    for entry in atlas.entries:
        candidates = []
        if entry.npz_path:
            candidates.append(entry.npz_path)
        hits = by_label.get(entry.label, [])
        # An unambiguous match is safe to use; an ambiguous one is not, and
        # guessing would reintroduce exactly the bug this avoids.
        if len(hits) == 1:
            candidates.extend(hits)
        elif hits:
            candidates.extend(
                p for p in hits
                if entry.family and entry.family.replace("/", os.sep) in p
            )
        for path in candidates:
            try:
                found[entry.key] = read_entry(path)["C"]
                break
            except Exception:
                continue
    return found


_CSS = """
:root{
  --bg:#fbfbfa; --panel:#fff; --ink:#1a1a19; --dim:#6b6b68; --line:#e3e3e0;
  --accent:#b8532b; --warn:#8a5a00; --warnbg:#fdf6e3; --dead:#b02020;
}
@media (prefers-color-scheme:dark){:root{
  --bg:#151514; --panel:#1e1e1d; --ink:#eceae6; --dim:#9a9a95; --line:#33332f;
  --accent:#e0784a; --warn:#d6a83c; --warnbg:#2a2313; --dead:#e06a6a;
}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:14px/1.5 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif}
header{padding:20px 24px 12px;border-bottom:1px solid var(--line)}
h1{margin:0 0 4px;font-size:19px;font-weight:620;letter-spacing:-.01em}
.sub{color:var(--dim);font-size:13px}
.warn{margin:12px 24px 0;padding:10px 12px;background:var(--warnbg);
  border-left:3px solid var(--warn);font-size:13px;border-radius:0 4px 4px 0}
nav{display:flex;gap:2px;padding:12px 24px 0;border-bottom:1px solid var(--line)}
nav button{background:none;border:none;border-bottom:2px solid transparent;
  color:var(--dim);padding:7px 14px;font:inherit;font-size:13px;cursor:pointer}
nav button.on{color:var(--ink);border-bottom-color:var(--accent);font-weight:560}
.bar{display:flex;flex-wrap:wrap;gap:14px;align-items:center;
  padding:14px 24px;border-bottom:1px solid var(--line);font-size:13px}
.bar label{color:var(--dim);margin-right:5px}
select{background:var(--panel);color:var(--ink);border:1px solid var(--line);
  border-radius:5px;padding:4px 7px;font:inherit;font-size:13px}
main{padding:20px 24px 60px}
.grid{display:grid;gap:14px;grid-template-columns:repeat(auto-fill,minmax(158px,1fr))}
.card{background:var(--panel);border:1px solid var(--line);border-radius:7px;
  padding:9px;cursor:pointer}
.card:hover{border-color:var(--accent)}
.card canvas{width:100%;height:auto;image-rendering:pixelated;border-radius:3px;
  display:block;background:var(--bg)}
.card .nm{font-size:11.5px;margin-top:7px;word-break:break-all;line-height:1.35}
.card .mt{font-size:11px;color:var(--dim);margin-top:3px;
  font-variant-numeric:tabular-nums}
.card.dead .nm{text-decoration:line-through;color:var(--dead)}
.det{display:grid;gap:26px;grid-template-columns:minmax(300px,460px) 1fr;
  align-items:start}
@media (max-width:820px){.det{grid-template-columns:1fr}}
.det canvas{width:100%;height:auto;image-rendering:pixelated;
  border:1px solid var(--line);border-radius:4px;background:var(--bg);display:block}
.stack{position:relative}
.stack svg{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
table{border-collapse:collapse;font-size:13px;width:100%}
td{padding:4px 12px 4px 0;border-bottom:1px solid var(--line);
  font-variant-numeric:tabular-nums}
td:first-child{color:var(--dim);white-space:nowrap}
.axis{display:flex;justify-content:space-between;color:var(--dim);
  font-size:11px;margin-top:4px;font-variant-numeric:tabular-nums}
.note{color:var(--dim);font-size:12.5px;max-width:66ch;margin:14px 0 0}
.empty{color:var(--dim);padding:36px 0}
.sw{display:flex;align-items:center;gap:7px;color:var(--dim);font-size:11.5px}
.sw i{display:block;width:78px;height:9px;border-radius:2px;
  background:linear-gradient(90deg,#faf6f0,#e8b48a,#b8532b,#4a1d0c)}
"""

_JS = r"""
const A = ATLAS_PAYLOAD;
const E = A.entries, G = A.grids || {};
let tab = 'gallery', sel = null;

// Sequential ramp. CKA is a magnitude, so the scale is single-hue and
// monotone in lightness -- a diverging or rainbow ramp would invent a midpoint
// the data does not have.
const RAMP = [[250,246,240],[240,214,186],[224,180,138],[184,83,43],[116,40,18],[40,12,6]];
function ramp(t){
  t = Math.max(0, Math.min(1, t)) * (RAMP.length - 1);
  const i = Math.min(Math.floor(t), RAMP.length - 2), f = t - i;
  const a = RAMP[i], b = RAMP[i+1];
  return [a[0]+(b[0]-a[0])*f, a[1]+(b[1]-a[1])*f, a[2]+(b[2]-a[2])*f];
}
function decode(g){
  const raw = atob(g.d), n = g.n, out = new Float64Array(n*n);
  for (let i = 0; i < n*n; i++) out[i] = raw.charCodeAt(i) / 255;
  return {n, v: out};
}
function paint(cv, g, lo, hi){
  const {n, v} = decode(g);
  cv.width = n; cv.height = n;
  const ctx = cv.getContext('2d'), img = ctx.createImageData(n, n);
  const span = Math.max(hi - lo, 1e-9);
  for (let i = 0; i < n*n; i++){
    const c = ramp((v[i] - lo) / span);
    img.data[i*4] = c[0]; img.data[i*4+1] = c[1];
    img.data[i*4+2] = c[2]; img.data[i*4+3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return {n, v};
}
// Autoscale to the off-diagonal range. The diagonal is 1 by construction and
// carries no information; letting it set the top of the scale flattens every
// real difference into the same dark corner.
function offDiagRange(g){
  const {n, v} = decode(g);
  let lo = 1, hi = 0;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++){
    if (i === j) continue;
    const x = v[i*n+j];
    if (x < lo) lo = x;
    if (x > hi) hi = x;
  }
  return hi > lo ? [lo, hi] : [0, 1];
}
const fmt = (x, d=3) => (x === null || x === undefined || !isFinite(x)) ? '--' : (+x).toFixed(d);
function human(x){
  if (x === null || x === undefined || !isFinite(x)) return '--';
  const u = [[1e12,'T'],[1e9,'B'],[1e6,'M'],[1e3,'k']];
  for (const [m, s] of u) if (Math.abs(x) >= m) return (x/m).toFixed(x/m < 10 ? 1 : 0) + s;
  return '' + x;
}

function families(){ return [...new Set(E.map(e => e.family))].sort(); }
function visible(){
  const f = document.getElementById('fam').value;
  let list = f === '*' ? E.slice() : E.filter(e => e.family === f);
  const k = document.getElementById('sort').value;
  const key = e => k === 'label' ? e.label
    : k === 'blockiness' ? -(e.blocks?.blockiness ?? -1)
    : k === 'boundary' ? (e.blocks?.sensory_end ?? 9)
    : (e[k] ?? -1);
  list.sort((a, b) => {
    const x = key(a), y = key(b);
    if (typeof x === 'string' || typeof y === 'string') return ('' + x).localeCompare('' + y);
    return x - y;
  });
  return list;
}

function gallery(){
  const list = visible(), host = document.getElementById('main');
  if (!list.length){ host.innerHTML = '<p class="empty">Nothing matches.</p>'; return; }
  host.innerHTML = '<div class="grid"></div>';
  const grid = host.firstChild;
  for (const e of list){
    const card = document.createElement('div');
    card.className = 'card' + (e.lens_is_live === false ? ' dead' : '');
    const b = e.blocks || {};
    card.innerHTML = `<canvas></canvas><div class="nm">${e.label}</div>
      <div class="mt">${e.n_layers}L &middot; ${b.n_blocks ?? '?'} blocks &middot;
      B=${fmt(b.blockiness, 2)}</div>`;
    const g = G[e.key];
    if (g){ const [lo, hi] = offDiagRange(g); paint(card.firstChild, g, lo, hi); }
    card.onclick = () => { sel = e.key; show('detail'); };
    grid.appendChild(card);
  }
}

function detail(){
  const host = document.getElementById('main');
  const e = E.find(x => x.key === sel) || visible()[0];
  if (!e){ host.innerHTML = '<p class="empty">Nothing to show.</p>'; return; }
  sel = e.key;
  const b = e.blocks || {}, g = G[e.key];
  const rows = [
    ['model', e.repo || e.label], ['family', e.family],
    ['layers', e.n_layers], ['d_model', e.d_model],
    ['step', e.step ?? '--'], ['tokens', human(e.tokens)],
    ['compute (6ND)', e.compute_flops ? e.compute_flops.toExponential(2) : '--'],
    ['blocks', b.n_blocks ?? '--'],
    ['blockiness', fmt(b.blockiness)],
    ['mean within / between', fmt(b.mean_within, 2) + ' / ' + fmt(b.mean_between, 2)],
    ['layer coupling', fmt(b.layer_coupling)],
    ['boundaries (rho)', (b.boundaries_rho || []).map(x => fmt(x, 2)).join(', ') || '--'],
    ['lens vs logit lens', e.fidelity && e.fidelity.advantage !== undefined
      ? fmt(e.fidelity.advantage) : 'not measured'],
    ['budget', e.budget_fingerprint || 'not recorded'],
  ];
  host.innerHTML = `<div class="det"><div>
      <div class="stack"><canvas id="big"></canvas><svg id="ov"
        viewBox="0 0 100 100" preserveAspectRatio="none"></svg></div>
      <div class="axis"><span>rho 0.00</span><span>relative depth</span><span>1.00</span></div>
      <div class="axis" style="margin-top:8px"><span class="sw"><i></i></span>
        <span id="rng"></span></div>
      ${b.recomputed ? '<p class="note">Blocks recomputed here by the frozen rule &mdash; this entry shipped without a segmentation.</p>' : ''}
      ${e.lens_is_live === false ? '<p class="note" style="color:var(--dead)">This lens was no better than the logit lens. The structure below describes the instrument, not the model.</p>' : ''}
    </div><div><table>${rows.map(r =>
      `<tr><td>${r[0]}</td><td>${r[1]}</td></tr>`).join('')}</table>
    <p class="note">Cell (i,&nbsp;j) is the linear CKA between the J-lens vectors at
    layers i and j. Both axes run over relative depth, so models of different
    depth are directly comparable. Dark blocks on the diagonal are stretches of
    depth that share a readout geometry; the boundary between them is what the
    segmentation reports.</p></div></div>`;
  if (g){
    const [lo, hi] = offDiagRange(g);
    const cv = document.getElementById('big');
    paint(cv, g, lo, hi);
    document.getElementById('rng').textContent =
      `off-diagonal ${fmt(lo, 2)} to ${fmt(hi, 2)}`;
    // Boundaries go in an overlaid SVG rather than into the pixel canvas. The
    // canvas is n x n and displayed several times that size, so a line drawn
    // into it is either sub-pixel-thin or a whole cell wide depending on n.
    const n = g.n, parts = [];
    for (const rho of (b.boundaries_rho || [])){
      const idx = (e.rho || []).findIndex(r => r >= rho - 1e-9);
      if (idx <= 0) continue;
      const p = idx / n * 100;
      parts.push(`<line x1="${p}" y1="0" x2="${p}" y2="100"/>`,
                 `<line x1="0" y1="${p}" x2="100" y2="${p}"/>`);
    }
    document.getElementById('ov').innerHTML = parts.length
      ? `<g stroke="rgba(255,255,255,.9)" stroke-width=".5"
           vector-effect="non-scaling-stroke">${parts.join('')}</g>` : '';
  }
}

function training(){
  const host = document.getElementById('main');
  const xk = document.getElementById('xaxis').value;
  const yk = document.getElementById('yaxis').value;
  const yv = e => yk === 'boundary' ? e.blocks?.sensory_end
    : yk === 'n_blocks' ? e.blocks?.n_blocks
    : yk === 'coupling' ? e.blocks?.layer_coupling
    : e.blocks?.blockiness;
  // Dead-lens entries are excluded: a curve that includes them is partly a
  // curve of how well the lens fitted, which is the confound this whole
  // exercise is about.
  const pts = E.filter(e => e.lens_is_live !== false)
               .map(e => ({f: e.family, x: e[xk], y: yv(e), l: e.label}))
               .filter(p => p.x !== null && p.x !== undefined && isFinite(p.x)
                         && p.y !== null && p.y !== undefined && isFinite(p.y)
                         && (xk === 'step' ? p.x >= 0 : p.x > 0));
  if (pts.length < 2){
    host.innerHTML = `<p class="empty">Not enough entries carry both
      <code>${xk}</code> and that metric. A checkpoint sweep
      (<code>experiments/jspace_open.py</code>) fills this in; released-model
      atlases usually cannot.</p>`;
    return;
  }
  const W = 860, H = 420, P = 56;
  const log = xk !== 'step';
  const tx = v => log ? Math.log10(Math.max(v, 1)) : v;
  const xs = pts.map(p => tx(p.x)), ys = pts.map(p => p.y);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  let y0 = Math.min(...ys), y1 = Math.max(...ys);
  if (y1 - y0 < 1e-9){ y0 -= 0.5; y1 += 0.5; }
  const px = v => P + (tx(v) - x0) / Math.max(x1 - x0, 1e-9) * (W - P - 18);
  const py = v => H - P - (v - y0) / (y1 - y0) * (H - P - 22);
  const fams = [...new Set(pts.map(p => p.f))].sort();
  const hue = i => `hsl(${(i * 61) % 360} 62% 48%)`;
  let svg = `<svg viewBox="0 0 ${W} ${H}" width="100%" style="max-width:${W}px">
    <line x1="${P}" y1="${H-P}" x2="${W-18}" y2="${H-P}" stroke="var(--line)"/>
    <line x1="${P}" y1="22" x2="${P}" y2="${H-P}" stroke="var(--line)"/>`;
  for (let t = 0; t <= 4; t++){
    const v = y0 + (y1 - y0) * t / 4, y = py(v);
    svg += `<line x1="${P}" y1="${y}" x2="${W-18}" y2="${y}" stroke="var(--line)"
      stroke-dasharray="2 4"/><text x="${P-8}" y="${y+4}" text-anchor="end"
      fill="var(--dim)" font-size="11">${v.toFixed(2)}</text>`;
  }
  for (let t = 0; t <= 4; t++){
    const lv = x0 + (x1 - x0) * t / 4, x = P + (W - P - 18) * t / 4;
    const lab = log ? human(Math.pow(10, lv)) : human(lv);
    svg += `<text x="${x}" y="${H-P+18}" text-anchor="middle" fill="var(--dim)"
      font-size="11">${lab}</text>`;
  }
  fams.forEach((f, i) => {
    const fp = pts.filter(p => p.f === f).sort((a, b) => tx(a.x) - tx(b.x));
    svg += `<polyline fill="none" stroke="${hue(i)}" stroke-width="1.6"
      points="${fp.map(p => `${px(p.x)},${py(p.y)}`).join(' ')}"/>`;
    for (const p of fp)
      svg += `<circle cx="${px(p.x)}" cy="${py(p.y)}" r="3" fill="${hue(i)}">
        <title>${p.l}\n${xk}=${human(p.x)}  ${yk}=${fmt(p.y)}</title></circle>`;
    svg += `<text x="${W-18}" y="${30+i*16}" text-anchor="end" fill="${hue(i)}"
      font-size="11.5">${f}</text>`;
  });
  svg += `<text x="${(W+P)/2}" y="${H-10}" text-anchor="middle" fill="var(--dim)"
    font-size="12">${xk}${log ? ' (log)' : ''}</text></svg>`;
  host.innerHTML = svg + `<p class="note">Entries whose lens did not beat the
    logit lens are excluded &mdash; their geometry tracks how well the lens
    fitted, not the model. Step is a unit of optimizer, so two sizes at the same
    step have bought different amounts of learning; tokens and compute are the
    axes that make sizes comparable.</p>`;
}

function show(t){
  tab = t;
  for (const b of document.querySelectorAll('nav button'))
    b.classList.toggle('on', b.dataset.t === t);
  document.getElementById('galbar').style.display = t === 'training' ? 'none' : 'flex';
  document.getElementById('trbar').style.display = t === 'training' ? 'flex' : 'none';
  (t === 'gallery' ? gallery : t === 'detail' ? detail : training)();
}
window.addEventListener('DOMContentLoaded', () => {
  const fam = document.getElementById('fam');
  fam.innerHTML = '<option value="*">all families</option>' +
    families().map(f => `<option>${f}</option>`).join('');
  for (const el of document.querySelectorAll('select'))
    el.onchange = () => show(tab);
  for (const b of document.querySelectorAll('nav button'))
    b.onclick = () => show(b.dataset.t);
  // Default the training x-axis to one the atlas actually carries. A released-
  // model atlas often has steps but no token accounting, and defaulting to
  // tokens would show an empty chart that looks like a missing result.
  const xa = document.getElementById('xaxis');
  const covered = k => E.filter(e => e[k] !== null && e[k] !== undefined
                                  && isFinite(e[k])).length;
  const best = ['compute_flops', 'tokens', 'step']
    .map(k => [k, covered(k)]).sort((a, b) => b[1] - a[1])[0];
  if (best && best[1] > 0) xa.value = best[0];
  show('gallery');
});
"""


def build_page(atlas: Atlas, matrices: dict[str, np.ndarray]) -> str:
    """The whole viewer as one HTML string."""
    payload = json.dumps(json_safe(_payload(atlas, matrices)))
    manifest = atlas.manifest()
    warn = "".join(
        f'<div class="warn">{w}</div>' for w in manifest.get("warnings", [])
    )
    n_live = sum(1 for e in atlas.entries if e.lens_is_live)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{atlas.title}</title><style>{_CSS}</style></head><body>
<header>
  <h1>{atlas.title}</h1>
  <div class="sub">{len(atlas.entries)} fitted lenses &middot; {n_live} with a
  live lens &middot; linear CKA over J-lens vectors, on the relative-depth axis</div>
</header>
{warn}
<nav>
  <button data-t="gallery" class="on">Gallery</button>
  <button data-t="detail">Detail</button>
  <button data-t="training">Training</button>
</nav>
<div class="bar" id="galbar">
  <span><label>family</label><select id="fam"></select></span>
  <span><label>sort by</label><select id="sort">
    <option value="label">name</option>
    <option value="step">step</option>
    <option value="tokens">tokens</option>
    <option value="compute_flops">compute</option>
    <option value="n_layers">depth</option>
    <option value="blockiness">blockiness</option>
    <option value="boundary">boundary depth</option>
  </select></span>
</div>
<div class="bar" id="trbar" style="display:none">
  <span><label>x</label><select id="xaxis">
    <option value="tokens">tokens</option>
    <option value="compute_flops">compute (6ND)</option>
    <option value="step">step</option>
  </select></span>
  <span><label>y</label><select id="yaxis">
    <option value="blockiness">blockiness</option>
    <option value="boundary">first boundary (rho)</option>
    <option value="n_blocks">block count</option>
    <option value="coupling">layer coupling</option>
  </select></span>
</div>
<main id="main"></main>
<script>const ATLAS_PAYLOAD = {payload};</script>
<script>{_JS}</script>
</body></html>
"""


def write_page(
    atlas: Atlas,
    path: str,
    *,
    search_dirs: Sequence[str] | None = None,
    matrices: dict[str, np.ndarray] | None = None,
) -> str:
    """Render the atlas to a standalone HTML file.

    Args:
        search_dirs: Where to hunt for each entry's ``cka_*.npz``. Defaults to
            the directory the page is being written into.

    Returns:
        The path written.
    """
    if matrices is None:
        roots = list(search_dirs or [os.path.dirname(os.path.abspath(path)) or "."])
        matrices = collect_matrices(atlas, roots)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(build_page(atlas, matrices))
    return path
