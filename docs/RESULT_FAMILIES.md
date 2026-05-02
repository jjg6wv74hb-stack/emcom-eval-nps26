# Result Families

The manuscript uses the zero-auxiliary-loss result family. The Quarto source maps logical family names to canonical directories under `artifacts/paper/`.

Core families:

- `base_gap`: base communication vs no-communication checkpoint gap.
- `message_source/status`: matched message-source control summaries.
- `message_source/train/*`: full learned, uniform, public-random, fixed0, and fixed1 training/evaluation outputs.
- `endpoint/frozen50k_expanded` and `endpoint/frozen150k_expanded`: frozen-policy intervention suites.
- `sender_encoding/natural_intended_150k`: sender encoding distribution and sensitivity outputs.
- `sender_causal/150k`: single-sender causal intervention outputs.
- `transfer/cross_seed_flip_matched`: cross-seed transfer and polarity-alignment outputs.
- `factorial/*`: communication-by-history factorial outputs.
- `noise_sweep/*`: evaluation-time observation-noise sweep outputs.
- `crossover/*`: train-source by test-stream cross-over matrix outputs.
- `role_allocation/cfg07_30k_framework_validation`: hidden-need role-allocation summary outputs and figures for the second-paradigm case study.

The artifact zip includes render-time summaries for these families, not the full
raw traces or checkpoints. Full analysis regeneration requires the larger
artifact set.
