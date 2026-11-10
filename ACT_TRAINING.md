# Local ACT training and simulation tests

The RTX 4080 (16 GB) in this PC supports this configuration. Run the commands
below **inside `chrono-orb`**, from `attempt-codex`. The isolated `.venv-act`
environment contains PyTorch 2.7.1/CUDA 12.8 and LeRobot 0.4.3; it leaves the
container's base Python installation in place. Nothing uploads datasets or
checkpoints. These tools do not connect to a hardware robot.

## This collection

`output/pick_and_place` contains 500 attempts, including 428 complete successful
episodes (128,400 synchronized frames at 30 Hz). The model uses those successes:
385 episodes for training and 43 for validation. Episodes sharing a randomized
tray layout within a worker remain in the same partition. The split and source
indices are saved in `output/act_data/split/split.json`.

The source NPZ targets are absolute next-tick joint positions in CAD radians.
The official LeRobot exports in `output/act_data/train` and `validation` use the
reference SO101 joint order: shoulder pan, shoulder lift, elbow flex, wrist
flex, wrist roll, gripper. Arm values are degrees; gripper values are percent.
The calibration and per-episode provenance are in `chrono_metadata.json`.
Images are `observation.images.wrist` and `observation.images.external`.
Ground-truth cube/tray positions, phases and contact labels are diagnostics,
and are not policy inputs. The raised tray is retained.

## Run and monitor

The initial baseline uses a ResNet18 visual backbone, both 640×480 cameras,
six joint positions, a 30-action chunk, batch size 8, BF16 training, and 20,000
optimizer steps. Early tests executed five actions before updating the plan.
The selected rollout setting uses standard ACT temporal ensembling (coefficient
0.01): it blends overlapping predicted chunks and updates the plan every frame.
This setting was chosen using held-out recorded layouts before fresh-layout
testing. The simulator rollout tool applies it by default. Pass
`--no-temporal-ensemble --action-steps 5` to reproduce the early executor.
It is a single-task pick-and-place policy; task text is metadata and does not
provide language conditioning.

After the exports finish, start a new baseline with:

```bash
./train-act-baseline.sh
```

The current end-to-end experiment was started with:

```bash
source ../../chrono_env.sh
.venv-act/bin/python tools/run_act_experiment.py --wait-for-exports 1800
```

That driver merges six independent official exports, trains, measures held-out
prediction error, and runs five held-out layouts plus five fresh seeded
layouts through Chrono. It saves videos for the first three in each set.
It refuses to overwrite an existing training run. Re-running it after a partial
experiment requires explicitly resuming training and running remaining tests.

```bash
cat output/act_setup/experiment_status.json
tail -f output/act_setup/train_full.log
```

Checkpoints are saved every 2,000 steps under
`output/act_runs/raised_act_001/checkpoints`. Pause training with Ctrl-C when
running the trainer interactively; resume from the most recent saved checkpoint:

```bash
./train-act.sh --resume=true \
  --config_path=output/act_runs/raised_act_001/checkpoints/last/pretrained_model/train_config.json
```

This resumes optimizer state and normalization as well as weights. Work since
the latest checkpoint is repeated. For a separate experiment, override
`--output_dir` on `train-act-baseline.sh` with a fresh directory.

## Evaluate a saved checkpoint

```bash
source ../../chrono_env.sh
model=output/act_runs/raised_act_001/checkpoints/last/pretrained_model
.venv-act/bin/python tools/evaluate_act_offline.py \
  --checkpoint "$model" --dataset output/act_data/validation \
  --output output/act_runs/raised_act_001/validation.json
.venv-act/bin/python tools/rollout_act.py \
  --checkpoint "$model" --output output/act_runs/raised_act_001/rollouts_held_out \
  --episodes 5 --held-out-split output/act_data/split/split.json
.venv-act/bin/python tools/rollout_act.py \
  --checkpoint "$model" --output output/act_runs/raised_act_001/rollouts_fresh \
  --episodes 5 --seed 1000
```

Choose a new rollout output directory if one already exists. Offline scoring
uses the model's saved training normalization and inference prior latent;
validation actions never supply the VAE latent. Errors are reported per joint,
with arm errors in degrees and gripper errors in percentage points.

## Completed baseline and refinement

The 20,000-step baseline is under `output/act_runs/raised_act_001`. It failed all
five recorded held-out physical layouts. Its fresh-layout evaluation was
stopped after one completed failed test to give GPU time to refinement.
Completed results and comparison videos remain in that run's `index.html`.

A separate refinement, `output/act_runs/raised_act_refine_001`, starts from the
baseline's 14,000-step checkpoint (which demonstrated a physical grasp and
transport), then trains for 6,000 updates at learning rate 1e-5 and backbone
rate 1e-6, with seed 2027. Its data partition and architecture stay the same.
This gives 20,000 total updates along that model's training lineage.
`tools/finish_act_refinement.py` packages the model, scores held-out prediction
errors and runs three recorded held-out layouts plus three fresh layouts
(seeds 2000–2002), saving videos and `index.html`. Check the physical results
before treating either model as a usable task controller.

The refinement completed **0/6 valid placements**: 0/3 recorded held-out layouts
and 0/3 fresh layouts. Two tests had bilateral jaw contact; one lifted the cube.
Its first-action held-out MAEs were approximately 1.15°, 0.73°, 0.92°, 0.69°,
1.50°, and 0.48 gripper percentage points. Those errors still allow approach
drift and failed physical grasps. The saved model is an experimental prototype,
not a reliable pick-and-place controller.

The refinement's standard inference bundle is
`output/act_runs/raised_act_refine_001/policy`; resumable weights and optimizer
state remain in its `checkpoints/last` directory. To continue refinement for
additional updates, for example to 20,000 refinement updates (34,000 total):

```bash
./train-act.sh --resume=true \
  --config_path=output/act_runs/raised_act_refine_001/checkpoints/last/pretrained_model/train_config.json \
  --steps=20000
```

Inference always uses six SO101 joint outputs. Neither rollout control nor
training adds cube poses, target poses or phase labels to the policy inputs.

During physical rollouts, only camera images and measured joints enter ACT.
All control actions come from the network. Targets are converted back to CAD
coordinates, bounded by joint limits and a 180 deg/s limit, and interpolated
between 30 Hz control ticks with 1 ms physics steps. That speed bound exceeds
the maximum recorded speed of every joint in the successful demonstrations.
The cube remains dynamic; its pose is set only at episode initialization.
Layout sampling may use IK to reject unreachable initial scenes, but IK does
not produce rollout control actions.

Each `results.json` reports grasp, lift, release, settled placement and contact
penetration separately. `success` requires a physical grasp and lift, release,
the entire cube footprint inside the tray, a settled cube, and jaw/environment
penetration below 0.5 mm. Each recorded episode has `cameras.mp4` (wrist and
external side by side), snapshots, and a full `rollout.json` trace.
Tests allow up to 20 seconds of control plus 1 second to settle. A released
cube settled in the tray for 0.5 seconds ends a test early; an object leaving
the workspace also ends the test and remains a failure.

`tools/package_act_policy.py CHECKPOINT OUTPUT` copies a standard LeRobot ACT
inference bundle with the selected ensembling settings and the calibration
metadata, leaving the original resumable training checkpoint intact.
The packaged model's `config.json` uses `n_action_steps=1` and
`temporal_ensemble_coeff=0.01`, so standard LeRobot inference also enables the
chosen execution mode. No weights are changed during packaging.

The tray has only about 8 mm clearance per side around an aligned cube.
Prediction error alone therefore cannot establish task success. A diagnostic
replay of source episode 0 using its recorded expert targets and this 30 Hz
interpolation ended about 5.5 mm from its original continuously controlled
result. That replay is explicitly labeled as expert control and is not
counted as a learned-policy result. The physics/servo fit and hardware transfer
remain separate from learning this synthetic task.
Additional expert-only diagnostic replays reduced the final discrepancy to
2.25 mm with PCHIP interpolation and 2.03 mm with cubic interpolation. These
use recorded future expert targets, not learned actions, and are not policy
successes. The learned tests above retain their stated linear control bridge.
This remaining execution mismatch also needs attention before hardware transfer.

## Recreate preparation

`tools/setup_act_env.sh` creates the isolated environment. Exact installed
versions are saved in `output/act_setup/requirements-frozen.txt`.
`tools/prepare_act_split.py SOURCE OUTPUT` creates a fresh grouped split.
`tools/export_lerobot.py SOURCE OUTPUT --so101 --episode-selection LIST.json`
exports a partition using the official writer; `tools/merge_act_exports.py`
combines disjoint shards through LeRobot's official aggregation code.
The interrupted serial conversion is retained separately as
`output/act_data/train_incomplete_serial` and is not used for training.
