# Commands

Run these from the repository root after installing the Python requirements.
Quarto and LaTeX are needed only for the manuscript render.

## Render

```bash
./scripts/reproduce/render_manuscript.sh
```

From an unpacked zip:

```bash
cd paper/neurips2026_comm_vecstraight
MPLCONFIGDIR="$PWD/_output/mplconfig" quarto render main.qmd --to pdf --execute-daemon false
```

## Checks

```bash
python3 -m py_compile $(find src tests scripts -name '*.py')
python3 -m pytest tests/test_wrapper.py tests/test_role_allocation_env.py tests/test_role_allocation_vectorized_training.py -q
./scripts/reproduce/check_cleanliness.sh
```

## Short Training Runs

EPGG smoke run:

```bash
python3 -m src.experiments_pgg_v0.train_ppo \
  --n_episodes 20 \
  --T 16 \
  --num_envs 1 \
  --env_backend serial \
  --count_env_episodes \
  --comm_enabled \
  --msg_source_mode learned \
  --condition_name smoke_learned \
  --seed 101 \
  --save_path outputs/smoke/epgg_learned_seed101.pt \
  --metrics_jsonl_path outputs/smoke/epgg_learned_seed101.jsonl \
  --checkpoint_interval 20 \
  --log_interval 10 \
  --regime_log_interval 10
```

Role-allocation smoke run:

```bash
python3 -m src.experiments_role_allocation.train_ppo_vec \
  --condition learned \
  --seed 101 \
  --total-episodes 32 \
  --num-envs 2 \
  --rollout-len 8 \
  --horizon 20 \
  --eval-episodes 4 \
  --out-dir outputs/smoke/role_allocation \
  --env-mode informant_executor \
  --checkpoint-interval-episodes 32
```

## Full Training Templates

Full EPGG reruns are much larger than the zip. They first need fixed-f
warm-start checkpoints and then the 15-seed phase-3 families.

Warm starts:

```bash
python3 -m src.experiments_pgg_v0.run_fixed_f_sweep \
  --f_values 5.0 \
  --seeds 101 202 303 404 505 606 707 808 909 1111 1212 1313 1414 1515 1616 \
  --n_episodes 50000 \
  --max_workers 4 \
  --save_dir outputs/train/fixed_f_grid \
  --log_dir outputs/train/fixed_f_grid/logs \
  --metrics_dir outputs/train/fixed_f_grid/metrics \
  --skip_existing
```

Main EPGG communication/no-communication family:

```bash
python3 -m src.experiments_pgg_v0.run_phase3_uninterrupted \
  --fixed_f_dir outputs/train/fixed_f_grid \
  --out_dir outputs/train/phase3_main_150k \
  --conditions cond1 cond2 \
  --seeds 101 202 303 404 505 606 707 808 909 1111 1212 1313 1414 1515 1616 \
  --n_episodes 150000 \
  --num_envs 8 \
  --checkpoint_interval 25000 \
  --max_workers 4 \
  --skip_existing
```

Matched message-source controls use the same phase-3 runner with the relevant
message source or intervention arguments in `src/experiments_pgg_v0/`. The
summary scripts under `src/analysis/` write the CSV files consumed by the
manuscript.

Role-allocation 30k reruns use the vectorized role-allocation trainer with each
condition and seed:

```bash
python3 -m src.experiments_role_allocation.train_ppo_vec \
  --condition learned \
  --seed 101 \
  --total-episodes 30000 \
  --num-envs 8 \
  --rollout-len 64 \
  --horizon 100 \
  --eval-episodes 200 \
  --out-dir outputs/train/role_allocation \
  --env-mode informant_executor \
  --checkpoint-interval-episodes 30000 \
  --eval-slot-permutation \
  --eval-message-shuffle
```

Repeat the role-allocation command over the seeds and conditions listed in
`configs/role_allocation/base.yaml`.
