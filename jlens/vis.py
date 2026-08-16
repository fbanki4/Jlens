"""The layer x position slice view.

One prompt, one page: a grid whose rows are layers (deepest at the top, the
way the paper draws it) and whose columns are token positions. Each cell shows
the top-ranked lens token there. Click a cell to *pin* its token and every cell
recolours by that token's rank, which is how you watch a concept appear at some
depth, persist across positions, and then get displaced.

The page is a single self-contained HTML file — no CDN, no build step — so it
can be opened locally or published as-is.
"""

from __future__ import annotations

import html
import json
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from jlens.lens import JacobianLens
from jlens.protocol import LensModel


@dataclass
class SliceData:
    """Everything the page needs.

    Attributes:
        tokens: Prompt tokens, as displayed.
        layers: Layer indices, ascending.
        top_tokens: ``[n_positions][n_layers][top_n]`` decoded strings.
        top_ranks: matching ranks (1-based) over the full vocabulary.
        top_ids: matching vocabulary ids.
        rank_lookup: ``{token_id: [n_positions][n_layers]}`` full-vocab rank of
            that token everywhere — populated for every token that appears as
            a top-1 anywhere, which is what makes pinning instant.
        model_top: the model's own top-1 next token at each position.
        title: page title.
    """

    tokens: list[str]
    layers: list[int]
    top_tokens: list[list[list[str]]]
    top_ranks: list[list[list[int]]]
    top_ids: list[list[list[int]]]
    rank_lookup: dict[int, list[list[int]]] = field(default_factory=dict)
    id_to_str: dict[int, str] = field(default_factory=dict)
    model_top: list[str] = field(default_factory=list)
    title: str = "J-lens slice"


@torch.no_grad()
def compute_slice(
    model: LensModel,
    lens: JacobianLens,
    prompt: str,
    *,
    layers: Sequence[int] | None = None,
    top_n: int = 8,
    max_seq_len: int = 128,
    use_jacobian: bool = True,
    title: str | None = None,
) -> SliceData:
    """Run the lens over every (position, layer) of one prompt."""
    layers = sorted(layers if layers is not None else lens.source_layers)
    lens_logits, model_logits, input_ids = lens.apply(
        model, prompt, layers=layers, max_seq_len=max_seq_len,
        use_jacobian=use_jacobian,
    )
    ids = input_ids[0].tolist()
    tokens = [model.tokenizer.decode([i]) for i in ids]
    n_pos = len(ids)

    # ranks[layer] : [n_pos, vocab] with 1 = highest score
    ranks = {}
    for layer in layers:
        order = lens_logits[layer].argsort(dim=-1, descending=True)
        r = torch.empty_like(order)
        arange = torch.arange(1, order.shape[-1] + 1).expand_as(order)
        r.scatter_(-1, order, arange)
        ranks[layer] = r

    top_tokens: list[list[list[str]]] = []
    top_ranks: list[list[list[int]]] = []
    top_ids: list[list[list[int]]] = []
    seen: set[int] = set()
    for p in range(n_pos):
        per_layer_tok, per_layer_rank, per_layer_id = [], [], []
        for layer in layers:
            top = lens_logits[layer][p].topk(top_n)
            tid = [int(i) for i in top.indices]
            per_layer_id.append(tid)
            per_layer_tok.append([model.tokenizer.decode([i]) for i in tid])
            per_layer_rank.append(list(range(1, top_n + 1)))
            seen.add(tid[0])
        top_tokens.append(per_layer_tok)
        top_ranks.append(per_layer_rank)
        top_ids.append(per_layer_id)

    rank_lookup = {
        tid: [[int(ranks[layer][p, tid]) for layer in layers] for p in range(n_pos)]
        for tid in sorted(seen)
    }
    return SliceData(
        tokens=tokens,
        layers=list(layers),
        top_tokens=top_tokens,
        top_ranks=top_ranks,
        top_ids=top_ids,
        rank_lookup=rank_lookup,
        id_to_str={tid: model.tokenizer.decode([tid]) for tid in sorted(seen)},
        model_top=[
            model.tokenizer.decode([int(model_logits[p].argmax())])
            for p in range(n_pos)
        ],
        title=title or f"J-lens slice — {prompt[:60]}",
    )


_PAGE = """<title>__TITLE__</title>
<style>
  :root {
    --bg:#fbfbfa; --fg:#1a1a19; --dim:#6b6b68; --line:#e3e2df;
    --accent:#b8552f; --cell:#ffffff; --hi:#f3ede4;
  }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#17171a; --fg:#e9e8e4; --dim:#96958f; --line:#2e2e33;
            --accent:#e08a5f; --cell:#1e1e22; --hi:#2a2620; }
  }
  :root[data-theme="dark"] { --bg:#17171a; --fg:#e9e8e4; --dim:#96958f;
    --line:#2e2e33; --accent:#e08a5f; --cell:#1e1e22; --hi:#2a2620; }
  :root[data-theme="light"] { --bg:#fbfbfa; --fg:#1a1a19; --dim:#6b6b68;
    --line:#e3e2df; --accent:#b8552f; --cell:#ffffff; --hi:#f3ede4; }
  body { margin:0; padding:1.5rem; background:var(--bg); color:var(--fg);
         font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
         font-size:14px; line-height:1.5; }
  h1 { font-size:1.05rem; font-weight:600; margin:0 0 .25rem; }
  .sub { color:var(--dim); font-size:.82rem; margin-bottom:1rem; }
  .wrap { overflow-x:auto; border:1px solid var(--line); border-radius:6px;
          background:var(--cell); }
  table { border-collapse:collapse; font-size:11px; white-space:nowrap; }
  th, td { border:1px solid var(--line); padding:2px 5px; text-align:center;
           font-variant-numeric:tabular-nums; }
  th.layer { position:sticky; left:0; background:var(--cell); z-index:2;
             color:var(--dim); font-weight:500; text-align:right; }
  td.cell { cursor:pointer; max-width:120px; overflow:hidden;
            text-overflow:ellipsis; }
  td.cell:hover { outline:2px solid var(--accent); outline-offset:-2px; }
  td.tok { color:var(--dim); font-weight:600; }
  sup { color:var(--dim); font-size:8px; margin-left:2px; }
  .pins { margin:.9rem 0 .4rem; display:flex; gap:.4rem; flex-wrap:wrap;
          align-items:center; }
  .pin { border:1px solid var(--accent); color:var(--accent); border-radius:99px;
         padding:.1rem .55rem; font-size:.78rem; cursor:pointer; }
  .pin.off { border-color:var(--line); color:var(--dim); }
  #detail { margin-top:1rem; font-size:.82rem; color:var(--dim);
            min-height:4.5rem; }
  #detail b { color:var(--fg); }
  code { background:var(--hi); padding:.05rem .3rem; border-radius:3px; }
</style>
<h1>__TITLE__</h1>
<div class="sub">Rows are layers, deepest at the top. Each cell is the
top-ranked J-lens token there. Click a cell to pin its token; pinned tokens
colour every cell by rank (dark = rank 1). Hover for the full top-k.</div>
<div class="pins" id="pins"><span style="color:var(--dim);font-size:.78rem">
click a cell to pin a token</span></div>
<div class="wrap"><table id="grid"></table></div>
<div id="detail"></div>
<script>
const DATA = __DATA__;
let pinned = null;
const nPos = DATA.tokens.length, nLay = DATA.layers.length;

function rankColor(r) {
  if (r === null || r === undefined) return "transparent";
  const t = Math.min(1, Math.log10(r) / 3.2);      // rank 1 -> 0, 1600+ -> 1
  const a = (1 - t) * 0.75;
  return `rgba(184,85,47,${a.toFixed(3)})`;
}

function build() {
  const rows = [];
  const head = ["<tr><th class='layer'>layer</th>"];
  for (let p = 0; p < nPos; p++)
    head.push(`<th class='tok'>${esc(DATA.tokens[p])}</th>`);
  head.push("</tr>");
  rows.push(head.join(""));

  for (let li = nLay - 1; li >= 0; li--) {
    const tr = [`<tr><th class='layer'>L${DATA.layers[li]}</th>`];
    for (let p = 0; p < nPos; p++) {
      const tok = DATA.top_tokens[p][li][0];
      const id  = DATA.top_ids[p][li][0];
      let bg = "transparent";
      if (pinned !== null) {
        const lut = DATA.rank_lookup[pinned];
        bg = lut ? rankColor(lut[p][li]) : "transparent";
      }
      tr.push(`<td class='cell' style="background:${bg}"
        data-p='${p}' data-l='${li}' data-id='${id}'>${esc(tok)}</td>`);
    }
    tr.push("</tr>");
    rows.push(tr.join(""));
  }
  const outRow = ["<tr><th class='layer'>output</th>"];
  for (let p = 0; p < nPos; p++)
    outRow.push(`<td class='tok'>${esc(DATA.model_top[p])}</td>`);
  outRow.push("</tr>");
  rows.push(outRow.join(""));
  document.getElementById("grid").innerHTML = rows.join("");

  document.querySelectorAll("td.cell").forEach(td => {
    td.onclick = () => {
      const id = +td.dataset.id;
      pinned = (pinned === id) ? null : id;
      build();
    };
    td.onmouseenter = () => showDetail(+td.dataset.p, +td.dataset.l);
  });
  renderPins();
}

function renderPins() {
  const el = document.getElementById("pins");
  if (pinned === null) {
    el.innerHTML = "<span style='color:var(--dim);font-size:.78rem'>" +
                   "click a cell to pin a token</span>";
    return;
  }
  el.innerHTML = `<span class='pin'>pinned: ${esc(DATA.id_to_str[pinned])}
    &nbsp;&times;</span>`;
  el.querySelector(".pin").onclick = () => { pinned = null; build(); };
}

function showDetail(p, li) {
  const toks = DATA.top_tokens[p][li];
  const list = toks.map((t, i) => `<code>${esc(t)}</code>`).join(" ");
  document.getElementById("detail").innerHTML =
    `<b>position ${p}</b> (<code>${esc(DATA.tokens[p])}</code>) ` +
    `&middot; <b>layer ${DATA.layers[li]}</b><br>top-k: ${list}`;
}

function esc(s) {
  return (s === "" ? "\\u2205" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/\\n/g, "\\u23ce");
}
build();
</script>
"""


def build_page(slice_data: SliceData) -> str:
    """Render a :class:`SliceData` to a self-contained HTML page body."""
    payload = {
        "tokens": slice_data.tokens,
        "layers": slice_data.layers,
        "top_tokens": slice_data.top_tokens,
        "top_ids": slice_data.top_ids,
        "rank_lookup": {str(k): v for k, v in slice_data.rank_lookup.items()},
        "id_to_str": {str(k): v for k, v in slice_data.id_to_str.items()},
        "model_top": slice_data.model_top,
    }
    return _PAGE.replace("__DATA__", json.dumps(payload)).replace(
        "__TITLE__", html.escape(slice_data.title)
    )


def write_page(slice_data: SliceData, path: str) -> str:
    with open(path, "w") as f:
        f.write(build_page(slice_data))
    return path
