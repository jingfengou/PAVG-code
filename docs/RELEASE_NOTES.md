# Release notes

## October 17, 2026

- Simplified installation, inference and example documentation.
- Simplified the scene-to-video configuration.
- Added user-facing setup verification instructions.
- Extended the source audit to detect Hugging Face access-token patterns.

## October 13, 2026

- Released single-phase and joint trajectory models, training and evaluation.
- Added shared input normalization, condition scaling and coordinate restoration.
- Added native camera projection and local-calibration projection.
- Set trajectory defaults to seed 42, DDIM 25 and CFG 1.0; DaS defaults to
  seed 20260910, 50 steps and CFG 6.0.
- Seeded global RNGs before tracking VAE encoding and enforced 49 frames at 24 fps.
- Added CPU input preparation and tracking-media checks.
- Added specialized Stage 1 training and an exposure-matched unified ablation.
- Applied phase-specific objectives, including gas divergence and solid
  deformation/contact constraints, before adding the interaction loss.
- Added per-sample evaluation seeds, complete-test aggregation and strict
  sample-level shard merging.
- Integrated the separately acquired PhysCtrl dependency with verified PAVG
  source extensions.

See [Verify your setup](VERIFICATION.md) for checks and
[third-party dependencies](THIRD_PARTY.md) for attribution and source terms.
