"""jlens — the Jacobian lens, from scratch.

The Jacobian lens reads out what an internal activation is disposed to make a
language model *say*. It transports a residual-stream vector at any layer into
the final-layer basis with an average input-output Jacobian, then decodes it
with the model's own unembedding::

    lens_l(h) = unembed( J_l @ h )        J_l = E[ d h_final / d h_l ]

Quick start on a real open model::

    import transformers, jlens
    hf  = transformers.AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B").cuda()
    tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    model = jlens.from_hf(hf, tok)

    lens = jlens.fit(model, prompts)
    print(lens.readout(model, "The capital of France is", layer=12))

Quick start with no GPU and no downloads — train a toy model and lens it::

    python -m experiments.run_all

References: Gurnee et al., "Verbalizable Representations Form a Global
Workspace in Language Models", Transformer Circuits Thread, 2026.
"""

from jlens.fitting import (
    DELTA_BINS,
    SKIP_FIRST_N_POSITIONS,
    delta_jacobian_for_prompt,
    fit,
    fit_delta,
    jacobian_for_prompt,
)
from jlens.hooks import ActivationRecorder
from jlens.lens import DeltaJacobianLens, JacobianLens
from jlens.protocol import LensModel

__all__ = [
    "ActivationRecorder",
    "DELTA_BINS",
    "DeltaJacobianLens",
    "JacobianLens",
    "LensModel",
    "SKIP_FIRST_N_POSITIONS",
    "delta_jacobian_for_prompt",
    "fit",
    "fit_delta",
    "from_hf",
    "jacobian_for_prompt",
]


def from_hf(*args, **kwargs):
    """Lazy re-export of :func:`jlens.hf.from_hf`, keeping ``transformers`` an
    optional dependency."""
    from jlens.hf import from_hf as _from_hf

    return _from_hf(*args, **kwargs)
