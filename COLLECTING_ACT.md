# Collecting pick-and-place demonstrations

Run these commands from the `attempt-codex` directory on the host. The existing
Chrono container runs the simulation; LeRobot conversion/training can run in a
separate Python environment. Nothing here uploads data or starts a hardware arm.

## Watch the controller first

```bash
./run-docker.sh --auto --episodes 5 --tray-every 1 --seed 42
./run-docker.sh --auto --global-camera right-45 --episodes 5
./run-docker.sh --auto --global-camera right-45 --start-pose side --episodes 5
```

`--global-camera right-45` is the default installed-camera view, with the fitted
pose and wide-angle projection from the setup photos. The wrist view remains.
Use `--global-camera top-down` for the optional virtual overhead view.

The arm's mounting base stays fixed. The red tray moves between episodes only.
Both cube spawns and the tray are on the centered matte black 18 × 12 inch
sheet. Its 1 mm thickness is included in placement and grasp heights; the
collection manifest records the work-surface dimensions and elevation.
Cube position/yaw changes every episode; tray position/yaw changes at the chosen
interval. IK plans a downward grasp and aligns the object with the tray before
dropping it. Layout sampling rejects unreachable goals and requires 7.5 cm
separation between cube/tray centers. Each episode lasts 10 simulated seconds.
The arm starts and finishes in a compact left-side rest, with the elbow folded
back and wrist/gripper pointing down, clearing the overhead view of the spawn
region. Occlusion during the actual approach/grasp remains possible.
Choose `--start-pose folded` (default) for that rest, or `--start-pose side`
for the earlier fully extended left-side position. Both the start and return
use the selected pose, independently of the global camera configuration.
Episode metadata records `start_pose` and `home_joint_rad` alongside the layout.

The first five joints follow prescribed trajectories; gripping uses **physical
jaw contacts**, with the full STEP tooth geometry and a torque-limited dynamic
moving jaw. The cube is never attached or made collision-free. The actual jaw
stops against the object under load. Descent and lift follow Cartesian vertical
paths, and release is above the tray's rim so the wider jaws need not enter it.
A successful episode requires bilateral jaw contact, actual lifting, settled
placement, and acceptable contact/tracking quality. A premature approach hit,
more than 0.5 mm penetration/tracking error, failed lift or missed placement
marks the episode failed. Servo/friction parameters remain uncalibrated, and
general arm self-collision is not modeled.

Collections labeled `kinematic_constraint_assist` predate this fix and can have
fixed-tooth overlap without an object response. Recollect these for physical
grasp training. The exporter rejects the legacy mode unless `--allow-assisted`
is explicitly supplied for pipeline experiments. Existing recordings are not
modified. You can repeat a layout with current physics:

```bash
./run-docker.sh --auto --replay-episode output/picks_003/episodes/000089/episode.json \
  --collect-dir output/episode89_recollected
```

## Collect synchronized observations and actions

```bash
./run-docker.sh --auto --headless --no-realtime \
  --episodes 100 --tray-every 5 --seed 42 --rate 30 \
  --collect-dir output/picks_001
```

The global files default to `right_45`. To use the optional overhead view, add
`--global-camera top-down` and choose a new collection directory.

Omit `--headless` to keep all three windows. `--no-realtime` permits faster/slower
wall-clock execution while preserving the requested simulated FPS. Default
rendering uses the GC21B tube at 50%, fixed spatial sampling, upright
wrist image, and no learned denoiser. Match FPS/resolution to the intended real
collection setup. Start with a small run and inspect the MP4s before scaling up.
The box interior is 35 × 24 × 20 inches. New dataset manifests record the
appearance profile, light CCT/brightness, white balance and estimated materials;
the October 19 appearance differs from earlier ceiling-panel datasets.

Every episode directory contains:

| File | Contents |
| --- | --- |
| `images/wrist/*.png`, `images/right_45/*.png` | Lossless, synchronized RGB observations |
| `wrist.mp4`, `right_45.mp4` | Convenience playback if ffmpeg is installed |
| `trajectory.npz` | Uniform timestamps, joint state/action/velocity, TCP and object poses, camera timestamps/counts, task phase, physical grasp flag, jaw forces/torque/penetration |
| `episode.json` | Layout, seed, success, completion, IK error, grasp mode and task |

With optional `top-down`, substitute `top_down` for `right_45` in the image/video paths.
`dataset.json` records the camera names, global view, resolution, FOV, the fixed
global camera transform and wrist camera transform relative to its arm link.

`dataset.json` indexes episodes. Interrupted episodes are retained and marked
incomplete. The exporter selects complete successful episodes by default.
Each observation at time `t` pairs with the **absolute joint target at `t+1/FPS`**.
The physical gripper's measured angle need not equal that target; contacts and
the finite servo torque limit determine its motion. The first five joint states
remain prescribed simulation coordinates. `grasp_assisted` is always false;
`grasp_contact` records a current bilateral physical grasp.
Both cameras finish rendering the same scene state before that row is accepted.
Cube/tray/TCP poses are privileged diagnostic labels; they are not ACT inputs.

Joint order is shoulder pan, shoulder lift, elbow flex, wrist flex, wrist roll,
gripper. Native state/action coordinates are **radians offset from the STEP
assembly pose**, including the gripper. Body poses are
`[x,y,z,qw,qx,qy,qz]` with positions in meters; TCP poses are 4 × 4 transforms. Image timestamps use
global simulation time; trajectory timestamps restart at zero per episode.
Chrono's float camera scheduling can add a few microseconds to image timestamps;
the scene does not move during that scheduling adjustment.

## Export using the official LeRobot writer

In an environment with NumPy and Pillow, validate without installing LeRobot:

```bash
python3 tools/export_lerobot.py output/picks_001 output/lerobot_picks_001 --validate-only
```

Install LeRobot using its [installation guide](https://huggingface.co/docs/lerobot/installation)
in a separate environment, then export:

```bash
python3 tools/export_lerobot.py output/picks_001 output/lerobot_picks_001 \
  --repo-id local/attempt-codex-picks
```

The exporter calls `LeRobotDataset.create`, `add_frame`, `save_episode`, and
`finalize`, producing the official dataset format rather than hand-written
Parquet/video metadata. Its features are `observation.state`, `action`,
`observation.images.wrist`, and `observation.images.right_45` (or
`observation.images.top_down` for the optional overhead view). Export uses the camera names
from the manifest. Keep a consistent configuration for training and deployment.
See the
[LeRobot dataset documentation](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)
and [writer source](https://github.com/huggingface/lerobot/blob/main/src/lerobot/datasets/lerobot_dataset.py).
The Chrono environment does not need PyTorch or LeRobot to collect native data.

### Collect multiple recordings concurrently

Use independent worker processes for separate episodes. Each worker owns its
Chrono system, physical contacts, cameras and output files. SynChrono is designed
for agents that exchange state in a distributed simulation, as described in
the [Chrono state-synchronization documentation](https://api.projectchrono.org/development/state_synchronization.html);
these independent recordings do not require a shared simulation clock or state.

```bash
./run-parallel-docker.sh --workers 4 --episodes 100 --seed 42 \
  --output output/parallel_picks_001 -- \
  --rate 30 --global-camera right-45 --start-pose folded
```

This requests 100 attempts total, split across four headless processes. Shared
demo flags follow `--`; seeds, worker counts, episode counts and destinations
are managed by the launcher. Each worker uses a distinct seed and directory.
`--tray-every` applies within each worker's episode sequence. A worker failure
does not discard other workers' completed data; inspect `worker_NNN.log`.

`batch.json` reports completed and successful episodes per hour, including
startup and recording/encoding time. Its combined `dataset.json` gives every
episode a unique batch index and records the worker/local index and seed. No
images or videos are copied during aggregation. The normal exporter accepts
the batch root and still filters failed/incomplete episodes:

```bash
python3 tools/export_lerobot.py output/parallel_picks_001 output/lerobot_parallel_001 --validate-only
python3 tools/export_lerobot.py output/parallel_picks_001 output/lerobot_parallel_001
```

Start with a small worker count and measure total successful throughput before
increasing it. All workers share GPU memory/rendering capacity and disk I/O.
`--threads-per-worker 1` is the default for OpenMP/BLAS and MP4 encoding; increase
it only after measuring. Camera resolution, FPS and samples remain the normal
demo settings unless passed explicitly. Within the existing container, use
`./run-parallel.sh` with the same arguments. Ctrl-C requests graceful cleanup
of all recording worker process groups and retains finished/partial episodes.

Before mixing with real demonstrations, fit the motor sign/offset/scale against
the real follower's calibration. `--joint-map mapping.json` applies six scales
and offsets to both state and action: `hardware = scale * sim_radians + offset`.
The JSON must contain `scale` (six numbers), `offset` (six numbers), and a `units`
description. No guessed hardware calibration is supplied. The real LeRobot
SO follower can use degrees or normalized arm coordinates, while its gripper
uses 0–100 normalization; see the
[SO follower implementation](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/so_follower/so_follower.py).
The default export remains in simulation radians and writes that fact into
`chrono_metadata.json`. Camera calibration, wrist mounting and latency also
need matching before combining simulated and real data.

## Train ACT locally

After export, in the LeRobot environment:

```bash
lerobot-train \
  --dataset.repo_id=local/attempt-codex-picks \
  --dataset.root="$PWD/output/lerobot_picks_001" \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --output_dir=output/train/act_picks_001 --job_name=act_picks_001 \
  --batch_size=8 --steps=100000 --wandb.enable=false
```

ACT consumes the camera images plus current joint positions and learns chunks
of future joint targets. The command follows the
[official ACT training workflow](https://huggingface.co/docs/lerobot/act), with
local dataset loading and uploading disabled. Keep collection FPS, feature
names, units and action timing consistent at inference.

As an initial experiment, collect 20 episodes to validate the pipeline, then
expand to 100–300 successful demonstrations covering the reachable spawn/yaw
range. Those counts are starting points, not a guarantee of policy quality.
Hold out entire tray-layout groups and separate seeds for evaluation; splitting
adjacent frames or repeated tray layouts leaks nearly identical scenes into
both sets. Inspect the worst trajectories as well as random successes. Add real
demonstrations and fine-tune to address camera appearance, calibration, tracking
error and physical grasp differences. Evaluate actual pickup and settled
placement success; offline prediction loss alone cannot establish that.

This change provides automated IK demonstrations and recording/export tooling.
ACT training, learned-policy rollouts and hardware control are separate steps;
the demo does not currently load an ACT checkpoint. Physical contacts remove
the earlier attachment artifact, but uncalibrated servo, friction and rigid
material models still require real-world validation and fine-tuning.

## Checks

```bash
python3 -m unittest discover -s tests -v
python3 demo_attempt_codex.py --validate
```

The CPU checks exercise randomized IK goals, joint bounds, continuity,
reproducibility, unreachable targets, full-jaw clearance, vertical grasp paths
and the camera float-deadline regression. Inside the Chrono environment,
`python tools/check_jaw_physics.py --episode EPISODE_JSON --output DIR` runs a
render-free physical replay and writes contact/trajectory diagnostics.
`--probe-fixed-jaw` intentionally recreates the old overlap and asserts that
the fixed teeth exert contact force and displace the cube.
The diagnostic tool also writes `profile.json`, with phase timings for collision
detection, solving, contact reporting and control. Its RTF is simulated seconds
divided by wall seconds; startup and output encoding are excluded. Add
`--with-sensors --rate 30` to include headless camera-rendering time.
