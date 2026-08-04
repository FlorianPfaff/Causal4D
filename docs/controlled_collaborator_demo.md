# Controlled collaborator demo

The `Controlled collaborator demo video` workflow renders a compact visual
explanation of the delayed-contact benchmark. It creates:

- an H.264 MP4;
- an animated GIF;
- a high-resolution poster PNG;
- aggregate and representative-case summaries; and
- SHA-256 checksums for every presentation artifact.

The animation compares static contact persistence with the dynamic Causal4D
contact-path posterior. It shows material-point displacement, the posterior
sticking probability, the action signal, the causal prefix boundary, and the
90% predictive interval. The aggregate annotation is computed from the same 40
controlled cases used by the smoke evaluation: ten seeds crossed with contact
onsets at frames 4, 6, 8, and 10.

Run it from GitHub Actions or locally:

```bash
python -m pip install -e . matplotlib pillow imageio-ffmpeg
python -m causal4d.cli.dynamic_contact_demo \
  --output-dir controlled-collaborator-demo \
  --seed 0 \
  --prefix-frame-count 6 \
  --fps 4 \
  --require-gates
```

## Claim boundary

The video is controlled-simulation mechanism evidence. It demonstrates that an
action-conditioned dynamic contact belief can outperform a contact state frozen
at the prefix endpoint without reading future observations. It is not real-object
validation, a Bayesian-PhysTwin accuracy result, or evidence of overall state of
the art.

The rendered ground truth is shown only to explain the evaluation. The summary
must report `future_observations_read=0`, and the workflow fails if any of the 40
registered controlled cases fails its existing gate.
