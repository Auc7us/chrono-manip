# attempt-codex

PyChrono SO-101 follower in a white lightbox. Launching the demo opens three windows:

1. An interactive third-person view, looking through the open front.
2. A Chrono Sensor wrist camera attached to the gripper and supplied STL mount.
3. The installed right-wall global camera, using an approximate centered mount,
   45° downward tilt and wide-angle lens. `--global-camera top-down` retains an
   optional virtual view.

The included OBJ files were converted from the supplied STEP assembly. Running the
demo does **not** require FreeCAD, CadQuery, a STEP importer, or the original Downloads
folder. It uses the existing Chrono container, its Sensor/OptiX build, NVIDIA GPU,
and desktop display.

## Run

From the host, with `chrono-orb` already running:

```bash
cd /home/auc7us/sbel/chrono/chrono-docker/mountdir/manip-sim2real/attempt-codex
./run-docker.sh
```

Or inside that container:

```bash
cd ~/mountdir/manip-sim2real/attempt-codex
./run.sh
```

For another container name, set `CHRONO_CONTAINER=...` when invoking `run-docker.sh`.
The launcher uses the container's existing `DISPLAY` and `chrono_env.sh`.

Mouse left/right buttons rotate and pan the third-person view, the wheel zooms,
and the arrow/PageUp/PageDown keys move it. Close the third-person window or press
Ctrl-C in the launching terminal to stop. The two camera windows can be moved
independently. All views show the same scene and joint coordinates.

```bash
./run-docker.sh --motion                       # slow joint sweep
./run-docker.sh --assembly-pose                # original STEP joint pose
./run-docker.sh --joint-deg 0 105 -70 -5 90 65  # six offsets, in degrees
./run-docker.sh --fast                         # cheaper direct lighting
./run-docker.sh --path-traced --samples 16     # optional diffuse bounce, slower
./run-docker.sh --no-sensors                   # third-person CAD inspection
./run-docker.sh --duration 5 --save-dir output/example
./run-docker.sh --headless --duration 3 --save-dir output/headless
./run-docker.sh --headless --duration 2 --record-dir output/sequence
./run-docker.sh --auto --episodes 5             # IK pick/drop, three windows
./run-docker.sh --auto --global-camera top-down # optional virtual overhead view
./run-docker.sh --auto --start-pose side       # fully extended left-side start
./run-docker.sh --auto --episodes 20 --tray-every 5 --collect-dir output/picks_001
python3 demo_attempt_codex.py --validate       # asset/fit checks; no PyChrono needed
```

The appearance now follows the October 19 setup photographs: a **35-inch-wide**
white enclosure, charcoal sheet with rounded corners and subtle creases, ivory
printed links, black camera mount, and a narrow black-backed GC21B tube near the
rear ceiling edge. The open front looks into a dim room approximation.
For the default wall-camera view, run:

```bash
./run-docker.sh --global-camera right-45
./run-docker.sh --light-kelvin 4200 --white-balance-kelvin 4800 --light-brightness .5
```

Default CCT is an **estimated 4600 K**, between the two suspected settings;
the rendered response includes an approximate fixed 4600 K camera white balance.
The photographs cannot establish the lamp CCT independently of USB-camera white
balance. `--light-kelvin 4900` compares the other candidate. `--exposure` adjusts
relative display exposure. The 21 W electrical rating is not treated as optical
watts, and the 1300 lux rating is not treated as lumens. The manufacturer specifies
[1300 lux at 0.5 m](https://neewer.com/products/neewer-gc21b-21w-bi-color-handheld-led-tube-light-66606507);
50% brightness is approximated as 650 lux at that reference position. Absolute
camera radiometry and the dimmer curve are not calibrated. `appearance.py`
records the fixture specs separately from the photographic estimates.

`--global-camera right-45` is now the default. There is no overhead camera in
that scene. `--global-camera top-down` moves the modeled global camera to the
optional virtual overhead pose; the wrist remains the other sensor in both modes.
The tube and surrounding room retain the same physical positions.

The default mount follows the user's approximate placement: near the top center
of the side wall, about 40 mm inward. The default profile is
`assets/nominal_camera.json`; the inset refers to the **optical center**.
The vertical clearance and orientation are nominal assumptions, since the
reference photograph was slightly tilted and precise pose matching is deferred.

| Nominal optical-center quantity | Value |
| --- | --- |
| Inward from right wall | 4.0 cm |
| Distance from open front | 30.48 cm (side-wall center) |
| Height above floor | 48.8 cm (assumed 2 cm below ceiling) |
| Downward pitch | 45° |
| Yaw / roll | 0° / 0° |

The earlier photograph-fitted pose remains available explicitly:

```bash
./run-docker.sh --camera-profile assets/reference_camera.json
```

That optional profile retains its eight floor/mat landmarks, 2.64 px RMS residual,
2.17 cm inset, 39.42 cm front distance, 49.92 cm height and tilted orientation.
It is an estimate from one image and nominal dimensions. The nominal default
has no landmark-fit score. Both profiles retain the same estimated wide-angle
lens. `--camera-profile FILE` loads another profile.
The fitted equidistant polynomial generates rays directly in
OptiX for live windows, PNG recordings and datasets. `--lens-model pinhole`
selects the earlier rectified projection; `--hfov-deg` controls that option.
The wrist has a separately estimated optical alignment from its visible fixed
jaw/screw features, stored in `assets/reference_wrist.json`.

Both USB cameras use fixed photographic response estimates from
`assets/reference_response.json`. `--raw-response` bypasses them for fitting.
This response is fixed per camera; no frame-dependent autoexposure or noise is
introduced. The [photo-fit comparison report](preview/reference_match/index.html) includes
both original photos, actual sensor renders, opacity overlays, difference maps
and complete-frame RGB errors. It uses the optional photograph-fitted pose,
not the nominal default mount. Its metrics apply no image registration or crop.

The default camera images are 640 × 480 at 15 Hz. Rendering uses a fixed 2 × 2
subpixel grid, followed by a spatial box average, **without a learned denoiser**.
The tube diffuser uses 48 fixed integration points with the renderer that
correctly accumulates their contributions. Fixed image/light sampling removes
the frame-varying grain and denoiser-induced edge wobble from stationary views.
The default uses **one narrow inflatable tube shining inward and downward**,
with a broad emission cone, white-floor bounce and a second ceiling return.
The deterministic diffuse cells use a finite-area near-field approximation to
avoid point-light dots along seams. A small ambient term covers residual room
illumination and higher-order bounce. These are approximate radiosity terms,
not a full spectral or measured light-transport solution.
The visible shell has a white luminous face and opaque black backing. The tube
provides directional shading and soft contact shadows.
The default uses the LEGACY integrator, which sums all fixed integration points.
No sensor-noise filter or temporal image averaging is added. The optional
`--stochastic-area` and `--path-traced` modes use the PATH integrator and one
optical area emitter, avoiding its last-light-only accumulation bug. These
comparison modes can have random grain. `--denoiser on` is available only for
those stochastic modes and may reintroduce the changing-edge artifact.
Use `--width 1280`, `--rate 30`, or `--samples 4` for higher quality, at a
higher GPU cost. `--samples` is a spatial factor per axis: 4 means a 4 × 4 grid
per output pixel. In stochastic modes it instead sets random rays per axis.
`--exposure` scales sensor illumination. `--fast` replaces the tube integration
with one point source at its luminous face, giving harder shadows, and skips denoising.

`--save-dir` writes `wrist.png`, `right_45.png` (or optional `top_down.png`), `camera_poses.json`, and, when a
third-person window is open, `third_person.png`. PIL is optional; without it,
sensor images are saved as PPM. Camera poses use meters, +X optical forward,
+Y image left, and +Z image up. Saved images have the usual top-left origin.
The wrist sensor is nominally rolled 180° about its optical axis relative to the
CAD mounting frame, followed by the photographic alignment correction. The ceiling
camera keeps its original orientation when selected.
`object_and_tray.json` records the cube dimensions, mass, final pose, and tray
dimensions alongside those images.

`--record-dir DIR` saves each newly published camera frame as a PNG, with camera
poses and timestamps in `DIR/frames.json` (Pillow required). To compare two runs:

```bash
python3 tools/compare_camera_sequences.py output/before output/after --output output/review
```

Open `output/review/index.html` to play or scrub the actual consecutive frames.
The report also measures camera motion and frame-to-frame pixel changes. In the
current stationary check, settled frames from each default camera were pixel-identical;
the earlier denoised camera frames changed despite identical camera transforms.

## Automated pick-and-place and ACT data

`--auto` enables seeded cube spawns and a numerical IK controller: approach,
descend, close, lift, transfer, lower, open, retreat, and return home. Each episode
lasts 10 simulated seconds. The **red tray** moves every `--tray-every N` episodes
(default 5); `0` keeps its initial sampled pose. The arm stays clamped at the front
center. The tray and cube are reset between episodes, while the cameras and arm
remain continuous. `--seed` reproduces layouts; `--fixed-cube` fixes its spawn.
The default automated home pose (`--start-pose folded`) parks the arm compactly to the left, with the elbow
folded back and wrist/gripper pointing down, so the global camera can
see the cube and tray at episode start and after the arm returns home. The arm
can still occlude the cube while approaching and grasping it.
Use `--start-pose side` for the earlier fully extended left-side pose. This
selection controls both the start and return pose in every automated episode,
combines with either global-camera configuration, and is recorded in metadata.

Automatic layouts use a tested downward-grasp workspace near the front of the
box, rather than the static inspection tray's Y=45 cm position. Cube centers are
sampled at X=±7.5 cm, Y=22–28.5 cm and tray centers at X=±7.5 cm, Y=23.5–28.5 cm,
rejecting unreachable and overlapping layouts. Cube/tray yaw is also randomized.
The solver uses the measured CAD axes, joint bounds, a downward approach, and
the cube/tray orientation. Cartesian waypoint paths keep descent/lift/retreat
vertical; quintic timing gives smooth starts and stops.
This is not a general obstacle-avoidance or self-collision planner.

`--collect-dir` records both camera images at the same simulation/control tick,
six joint positions, the following tick's six absolute targets, joint velocities,
TCP/cube/tray poses, task phase, seed, and success. Physics waits for both newly
rendered camera buffers, so slow rendering changes wall time rather than silently
reusing frames. Each episode has lossless PNG sequences, `trajectory.npz`,
`episode.json`, and two MP4 previews when ffmpeg is available. `dataset.json`
indexes the run. Existing collection directories are never overwritten.
With `--global-camera right-45`, the global images/video are named `right_45`
and exported as `observation.images.right_45`. The manifest records the selected
configuration and camera transforms, keeping angled datasets distinguishable
from top-down datasets.

**Grasping uses physical contacts.** Both jaw collision meshes retain the full
STEP tessellation, including the fixed jaw's teeth: 11,788 non-degenerate fixed
jaw triangles and 9,438 moving jaw triangles, without decimation or a convex
hull. The cube stays dynamic and collision-enabled throughout pickup, carry
and release; it is never attached to the hand. The moving jaw has a revolute
torque actuator with a 0.12 Nm limit. Contact limits its actual closing angle,
so recorded gripper state can differ from the commanded target under load.
The fixed jaw follows the arm's prescribed motion as an active contact body,
preserving tangential friction velocity. Physics uses a 1 ms step.

The grasp target clears the fixed teeth during descent, and release occurs
above the tray rim because the gripper is wider than the 5 cm tray. Success
requires actual bilateral contact, a physical lift, settled placement, no
premature approach contact, and collision/tracking errors below 0.5 mm.
Contact force, jaw torque and penetration diagnostics are saved with each run;
the dataset mode is `physical_jaw_contacts`.

Earlier collections labeled `kinematic_constraint_assist` had no physical jaw
contacts and can contain tooth/cube overlap. Recollect them for grasp training.
The exporter rejects that legacy mode by default; `--allow-assisted` is an
explicit opt-in for old pipeline experiments only.

To re-test the exact layout from an existing episode using the current physics:

```bash
./run-docker.sh --auto --replay-episode output/picks_003/episodes/000089/episode.json
```

See [COLLECTING_ACT.md](COLLECTING_ACT.md) for collection, LeRobot export, local
ACT training commands, the coordinate mapping required for real motors, and
evaluation recommendations.

## Parallel collection

Independent recording processes can collect episodes concurrently on multiple
CPU cores while sharing the GPU. The launcher keeps the full contact model,
camera settings and synchronized observations in every worker:

```bash
./run-parallel-docker.sh --workers 4 --episodes 100 --seed 42 \
  --output output/parallel_picks_001 -- \
  --rate 30 --global-camera right-45 --start-pose folded --tray-every 5
```

`--episodes` is the total attempt count across the batch. Shared demo options
follow `--`. Worker i uses seed `42+i`, its own output folder and log, and no
windows or real-time waiting. Thread limits prevent OpenMP/BLAS and MP4 encoding
from each claiming all CPU cores. `run-parallel.sh` is the equivalent launcher
inside the container. Parent processes do not import CUDA or clone a live GPU
context; each worker starts a fresh Python interpreter.

The batch writes `batch.json` with completion/success throughput and a combined
`dataset.json` referencing each worker's files. Export the batch root directly:

```bash
python3 tools/export_lerobot.py output/parallel_picks_001 output/lerobot_parallel_001
```

Failed workers retain their logs and completed recordings. Ctrl-C stops the
workers and preserves partial episodes. Existing batch directories are never
overwritten. Try increasing worker counts while checking successful episodes
per hour in `batch.json`; GPU memory, rendering load, CPU scheduling and storage
can eventually limit scaling. Logical CPU count alone does not set the optimum.

## Geometry and mounting

- Clear interior: **35 inches wide × 24 inches deep × 20 inches high**
  (0.8890 × 0.6096 × 0.5080 m). Dimensions are interpreted as interior dimensions.
- Five slightly bowed white panel surfaces with uneven 1.6–3.0 mm wall/floor
  gaps, exposed brown board cores, a curled floor edge and overlapping seam tabs.
  Their bow is approximately 1.3 mm. The floor contact collider stays 6 mm thick.
  The front at Y=0 is open. Z is up; +Y goes into the box; X spans its width.
- A matte black **18 × 12 inch sheet**, 1 mm thick, is centered on the floor:
  18 inches across X and 12 inches front-to-back, centered at Y=12 inches.
  It provides contrast against the cream arm. Cube spawns and the tray sit on
  its collidable top surface; grasp heights include the sheet thickness.
  The visual surface adds rounded 7 mm corners, fine grain and creases below
  0.5 mm; the contact surface stays flat at 1 mm to preserve grasp heights.
- The CAD base is centered on the front edge and held by two modeled screw clamps.
  The support board and clamp dimensions are visual approximations.
- Cream printed links, black servo housings, metal hardware, and visible red,
  black, and white strands between all six daisy-chained motors.
- One green **3 × 3 × 3 cm PLA cube, mass 30 g**, resting in a red tray.
  The tray's outside footprint is **5 × 5 cm**, with a **1 cm raised edge** above
  its inside floor. Its base and walls are 2 mm thick, giving a 4.6 × 4.6 cm
  usable floor. The static tray is fitted at X=6.57 cm, Y=29.05 cm and yaw 147.84°;
  its four observed rim corners fit to 1.09 px RMS (0.77 px per coordinate). Automated episodes
  sample reachable cube/tray layouts as described above.
  The cube is a dynamic body with the specified mass and uniform cube inertia;
  the fixed tray has collision geometry for its floor and all four edges.
- The rigid jaws come from `SO101 Assembly.step`. The soft fingers are **not**
  substituted. `soarm_soft_gripper.stp` supplies the mount-to-gripper alignment.
- The wrist uses the actual `IMXmountSO101.stl`, placed using matching wrist-roll
  bores and dovetail screw axes. The small difference between the rigid and soft
  gripper revisions is corrected using the rigid gripper's own dovetail bores.
  The supplied mount and modeled fixed-camera supports are black.
- The STL camera plate differs from the older STEP reference plate: the STL has
  **21 × 13.5 mm** circular mounting slots; the STEP plate has a 28 × 28 mm pattern.
  The nominal CAD camera is centered on the supplied adapter's camera plate.
  No additional adapter is added. The slot pattern is
  approximately 0.017 mm from that plate center. Digital centering and bore/slot
  fit measurements are recorded in `assets/scene_assets.json`; these are not
  claims about printed-part tolerances.
- The nominal CAD wrist PCB's lens-facing surface sits **5 mm behind the mount's rear
  surface**, on 5 mm standoffs. The plate is 4.2 mm thick, and the modeled lens
  extends 9.8 mm beyond its front face through the central opening. The asset
  generator checks the lens-housing footprint against the actual STL opening.
  The default photographic fit adds an estimated optical/PCB placement correction
  to align the visible jaw and hardware; the modeled PCB/lens moves with that fit.
  Thus these nominal CAD clearances describe the source assembly, rather than
  measured mounting clearances in the physical setup. Two longer adapter bolts
  now have exposed polished heads with Phillips recesses above the black plate;
  the original STEP heads sit beneath the added adapter.
- Both cameras use the same 25 × 24 mm board mesh with actual through-holes,
  screws, connector, black M12 barrel, and lens glass. The lens housing is a
  dimensioned visual approximation rather than a manufacturer CAD import.
- The [NEEWER GC21B](https://neewer.com/products/neewer-gc21b-21w-bi-color-handheld-led-tube-light-66606507)
  has a nominal 90 cm inflated length, 21 W maximum electrical input, CRI95 and
  2700–6500 K CCT range. The visible body/backing silhouette is fitted to the
  reference image, giving a 90.15 cm long capsule, estimated 5.0 cm diameter,
  and a slight diagonal ceiling mounting. Its physical section is not measured.
  It has an inward/downward white diffuser and black backing, rather than a
  luminous ceiling. A 16 × 3 fixed quadrature approximates the luminous face.
  The optional `--path-traced` mode adds stochastic wall bounce and needs many
  more samples. Irrlicht's inspection window approximates the tube with a
  directional light; the sensor views are used for appearance comparisons.
- Panel reflectance, PLA color/roughness, sheet grain and shallow folds are
  photographic estimates, generated deterministically without baking the
  reference photos into a texture. The room includes an olive wall, outlet,
  varied book spines, wire cart/handles and the visible rust-colored chair seat.
  Depths of those partly visible objects are approximate.
  Saved camera/object metadata and new dataset manifests include an `appearance`
  record with dimensions, light settings, material reflectances and assumptions.

The static inspection pose and jaw opening are fitted jointly to the fixed and
wrist images and stored in `assets/reference_layout.json`. Automated start/return poses retain their
separate `--start-pose` selection. The original extended inspection pose is
available with `--joint-deg 0 105 -70 -5 90 65`.
Joint values are **offsets from the CAD assembly pose**, not
servo encoder zeros. The optional sweep uses the six measured servo axes and
updates the wrist camera and cable routing with the links. Arbitrary poses and
sweeps should be checked for clearance before collecting data.

## Simulation limits

The arm's first five joints follow prescribed CAD trajectories. Automated
gripping uses detailed rigid contacts and a physical torque-limited moving jaw.
Jaw mass/inertia, servo gains, friction and torque limit are approximate and
have not been fitted to the real robot; detailed geometry does not establish
calibrated real-world grasp performance. The 30 g cube moves under gravity and
contact forces; it does not model PLA elasticity. Full arm dynamics and cable
mechanics are not simulated, and general arm self-collision is not checked.
Arbitrary `--joint-deg` settings are not checked for self-collision.

The physical [Waveshare IMX335 5MP USB Camera (B)](https://www.waveshare.com/product/ai/cameras/single-cameras/imx335-5mp-usb-camera-b.htm)
is specified as a 25 × 24 mm module with a **175° diagonal** wide-angle lens and
2592 × 1944 maximum image size. The default now uses an image-fitted equidistant
polynomial rather than a pinhole. Its focal scale and principal point are scaled
with the requested output width. See the [manufacturer wiki](https://www.waveshare.com/wiki/IMX335_5MP_USB_Camera_%28B%29)
for the nominal hardware specifications. The fitted lens/pose and per-camera
photographic response reduce reference-image error; they do not establish a
unique metric calibration or measured shutter/ISO, noise or spectral response.
The installed `ChPhysCameraSensor` was considered; its exposure/noise/defocus
parameters require measured camera data to improve fidelity. The existing
`ChCameraSensor` already uses physical material shading. This change adjusts
geometry, illumination, reflectance, image-fitted lens/response and white balance.
Quantum efficiency, noise and defocus remain uncalibrated. CRI95 is recorded as a
fixture specification; this RGB renderer does not simulate its spectral CRI.

Irrlicht's third-person window runs in a spawned process, synchronized to the
sensor process's simulation time. This avoids a reproducible native graphics
shutdown crash when Irrlicht and Sensor's GLFW windows share this container's
Python process. Both sensor windows still use `ChCameraSensor` and
`ChFilterVisualize` directly. Headless mode needs no viewport process.

The first sensor launch builds demo-local ray/material shaders with the existing
container's `nvcc` and caches PTX under `output/shader_cache/`. Concurrent workers
share a file-locked build. These shaders are selected only for the demo process;
the installed Chrono files are unchanged. The local additions are direct fisheye
rays and bounded finite-cell bounce attenuation. The adapter source and upstream
license are under `assets/shaders/`.

## Assets and regeneration

`assets/cad/` contains unchanged copies of all four user-provided files, including
the follower STEP used as a secondary geometry reference. Source SHA-256 hashes
are stored in `scene_assets.json`. `assets/meshes/` contains runtime meshes in
meters; each link is grouped by material, preserving the assembly placements.
Hidden servo interiors are omitted, and single-color CAD cable solids are
replaced by the three colored strands.

To regenerate, install the conversion dependencies in a separate environment:

```bash
python3 -m pip install cadquery-ocp==7.9.3.1.1 trimesh scipy matplotlib
python3 tools/inspect_cad.py 'SO101 Assembly.step' --mesh
python3 tools/inspect_cad.py soarm_soft_gripper.stp --mesh
python3 tools/prepare_assets.py
python3 tools/prepare_appearance.py
python3 tools/prepare_jaw_collision.py
python3 demo_attempt_codex.py --validate
```

Optional inspection sheets: `python3 tools/preview_cad.py` and
`python3 tools/mount_features.py`. Only conversion tools need those dependencies;
normal execution needs PyChrono and NumPy, plus optional PIL for PNG export.

Saved run previews are in [preview/](preview/). Temporary camera captures,
conversion caches, and Python bytecode are ignored by Git.

## Reproducing the image fit

The fit utilities use SciPy/OpenCV already available in the Chrono container.
The ordinary demo uses the saved JSON profiles and does not import those tools.
The corner/feature measurements are hand-located for the two supplied photos;
edit the measurements before fitting different photos. With copies named
`output/reference_inputs/reference_global.jpg` and `reference_wrist.jpg`, run
inside the configured container environment:

```bash
python3 tools/fit_reference_camera.py
python3 tools/fit_reference_tube.py output/reference_inputs/reference_global.jpg
python3 tools/fit_reference_layout.py
python3 tools/fit_wrist_mount.py
python3 tools/fit_reference_layout.py
python3 tools/prepare_appearance.py
./run.sh --headless --duration .3 --no-realtime --camera-profile assets/reference_camera.json --raw-response --save-dir output/refit_raw
python3 tools/fit_reference_response.py output/refit_raw
./run.sh --headless --duration .3 --no-realtime --camera-profile assets/reference_camera.json --save-dir output/refit
python3 tools/review_reference_match.py --renders output/refit --camera-profile assets/reference_camera.json
```

The review metrics include the entire RGB frame. The automatically selected flat-region
mask is used only to fit a monotonic camera response, not to exclude pixels from
reported error. Room depths, material properties and single-view lens/pose
ambiguities remain explicit estimates in the saved profiles.
