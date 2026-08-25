# Fresh-Machine Setup — Dependencies & Docker

A from-scratch installation guide: everything you need on a brand-new machine
before you can follow [getting_started.md](getting_started.md) or
[getting_started_realsense.md](getting_started_realsense.md). Those two assume
the rig is already set up; this doc is how it *got* set up.

It consolidates what's scattered across the [README](../README.md#2-setup),
[external/README.md](../external/README.md), and
[Dockerfile.thesisnewcuda](../Dockerfile.thesisnewcuda), fills in the one step
none of them spell out (actually creating the `vision` container — the scripts
only `docker start`/`stop` an existing one), and covers the **host-side** ROS 2
workspace (`~/franka_ros2_ws`) that `mv_launch` and `fp_debug_msgs` live in,
since neither of those packages ships its own README.

## 0. Mental model

Two separate things run on two separate "sides", and each has its own
dependency stack:

| Side | What runs there | Where its deps are installed |
|---|---|---|
| **Host** | camera drivers (ZED/RealSense), the robot bridge, Foxglove, tmux, visualization nodes — started by `scripts/launch_host*.sh`, which call into `mv_launch` launch files in `~/franka_ros2_ws` | apt + a `franka_ros2_ws` colcon workspace, sourced from `~/.bashrc` or manually |
| **Container** (`vision`) | the perception pipeline itself (GDINO/SAM2/DINOv2/FoundationPose/ICP) — started by `scripts/launch_pipeline*.sh`, which `docker exec` into the container | `Dockerfile.thesisnewcuda`'s apt/pip layers, plus this repo's own `colcon build` run *inside* the container |

`mv_launch` (a plain ROS 2 launch-file package) and `fp_debug_msgs` (a ROS 2
message package) both belong to the **host** workspace, `~/franka_ros2_ws` —
that's why they're not inside `Masterthesis-vision` itself. Confusingly,
`Masterthesis-vision` also carries its *own* copy of `fp_debug_msgs` as a git
submodule at `src/fp_debug_msgs` — that copy gets built a second time, inside
the container, so the pipeline process can publish/subscribe the same message
types. You end up building `fp_debug_msgs` twice, once per side; see §3 and §5.

---

## 1. Host prerequisites

- **Ubuntu 22.04** (this is what everything below — ROS 2 Humble, the ZED SDK
  version pinned in `zed-ros2-wrapper`, the Dockerfile's base image — is
  pinned to).
- **An NVIDIA GPU + driver.** `nvidia-smi` must work on the host before
  anything else. The container needs driver ≥470 (CUDA 12.6 requires ≥525 in
  practice — anything reasonably current works).
- **Docker Engine** + the **NVIDIA Container Toolkit** (`nvidia-ctk`), so
  `docker run --gpus all` works:

  ```bash
  # Docker Engine, if not already installed:
  curl -fsSL https://get.docker.com | sh
  sudo usermod -aG docker "$USER"      # log out/in after this

  # NVIDIA Container Toolkit:
  # https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
  ```

  Verify:

  ```bash
  docker run --rm --gpus all nvidia/cuda:12.6.1-base-ubuntu22.04 nvidia-smi
  ```

- **ROS 2 Humble** (desktop install recommended — you'll want RViz/rqt on the
  host for visualization):
  https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html

  ```bash
  echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
  ```

- **tmux** (all the `launch_host*.sh` / `launch_pipeline*.sh` scripts run
  their windows inside a tmux session): `sudo apt install tmux`

- **colcon + rosdep**:

  ```bash
  sudo apt install python3-colcon-common-extensions python3-rosdep python3-vcstool
  sudo rosdep init   # only if not already done on this machine
  rosdep update
  ```

### 1.1 Camera SDKs (host only — cameras are never touched inside the container)

- **ZED SDK** (v5.x, matching `zed-ros2-wrapper`'s pin — see
  `~/franka_ros2_ws/src/zed-ros2-wrapper/README.md`):
  https://www.stereolabs.com/developers/release — install the CUDA-matching
  build for Ubuntu 22.04 before building `zed-ros2-wrapper`.
- **Intel RealSense SDK (librealsense)** — needed only for the RealSense-trio
  variant (§6 of the main README):

  ```bash
  sudo apt install ros-humble-librealsense2* || true   # or build librealsense from source
  rs-enumerate-devices -s      # sanity check once installed
  ```

- **KUKA FRI** (only if you're driving the real arm — the `lbr_fri_ros2_stack`
  package in the workspace below wraps it). Not needed to exercise the vision
  pipeline against a fixed/identity transform (see
  [getting_started_realsense.md §1 Step 1, Option B](getting_started_realsense.md#step-1--get-a-baseflange-transform-into-tf2-terminal-1)).

---

## 2. Host ROS 2 workspace (`~/franka_ros2_ws`) — builds `mv_launch` and `fp_debug_msgs`

This is a separate colcon workspace from `Masterthesis-vision` — it's where
the camera/robot driver packages live, `mv_launch` included. If it doesn't
exist yet on this machine:

```bash
mkdir -p ~/franka_ros2_ws/src
cd ~/franka_ros2_ws/src

# Pull in (clone or copy from another machine) at minimum:
#   mv_launch/            — launch files + flange_pose_publisher node (this repo's camera bring-up)
#   fp_debug_msgs/         — message defs (host-side copy)
#   zed-ros2-wrapper/ + zed-ros2-interfaces/   — ZED driver (needs ZED SDK from §1.1 first)
#   realsense-ros/         — RealSense driver (RealSense variant only)
#   lbr_fri_ros2_stack/ + lbr_fri_idl/ + fri/  — KUKA LBR bridge
#   any MoveIt2 config packages your calibration/planning scripts need
```

`mv_launch`'s own `package.xml` dependencies are all core ROS 2 packages
(`launch`, `launch_ros`, `rclpy`, `tf2_ros`, `geometry_msgs`) — nothing extra
to install for the package itself. But its launch files
(`zed2i_pair.launch.py`, `zed_realsense_trio.launch.py`,
`thesis_stack.launch.py`) start `zed_wrapper` / `realsense2_camera` nodes, so
those driver packages must be present and built in the **same** workspace
first. `fp_debug_msgs`'s `package.xml` declares a pure `rosidl` interface
package (`builtin_interfaces`, `geometry_msgs`, `action_msgs`) — no extra deps
either.

Resolve + build:

```bash
cd ~/franka_ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
echo "source ~/franka_ros2_ws/install/setup.bash" >> ~/.bashrc
source ~/franka_ros2_ws/install/setup.bash
```

Sanity check:

```bash
ros2 pkg list | grep -E "mv_launch|fp_debug_msgs"
ros2 launch mv_launch zed2i_pair.launch.py --show-args
```

---

## 3. This repo (`Masterthesis-vision`)

```bash
git clone <this-repo-url> ~/Masterthesis-vision
cd ~/Masterthesis-vision

git submodule update --init --recursive
bash external/apply_patches.sh        # idempotent FoundationPose thesis patch
```

`external/sam2` and `external/dinov2` come in via the submodule update above;
`external/Cutie` has no public remote and is git-ignored, so clone it
separately and pin the commit:

```bash
git clone <cutie-remote> external/Cutie
git -C external/Cutie checkout ec5cdd4cf16f75c73ad785a2f96fb97dbad4125a
```

See [external/README.md](../external/README.md) for the full rationale (why
these are submodules/sys.path tricks instead of `pip install`s).

### 3.1 The `Data/` folder (git-ignored — you must create it)

Meshes + reference images aren't in git. Create the layout by hand (or copy
from another machine) — full structure and naming rules in
[README §2.3](../README.md#23-the-data-folder-you-must-create-this):

```
Data/CAD_Models_centered/<assembly>/<object_id>.obj
Data/ZED_screens/<assembly>/<object_id>/*.png
Data/reference_renders/<assembly>/<object_id>/*.png   # optional
```

---

## 4. Docker — building the image

`Dockerfile.thesisnewcuda` is the only build input; there's no build script,
so build it directly:

```bash
cd ~/Masterthesis-vision
docker build -t masterthesis-vision:latest -f Dockerfile.thesisnewcuda .
```

This layers, in order: CUDA 12.6.1 + cuDNN devel base (Ubuntu 22.04) → ROS 2
Humble base + cv_bridge/sensor_msgs/geometry_msgs/image_transport →
`/opt/thesis-venv` (a `--system-site-packages` venv — deliberately *not*
isolated from the apt/ROS Python install, see
[external/README.md](../external/README.md#the-venv-why---system-site-packages)
for why and its one known footgun) → the ML stack via pip (torch 2.7.0/cu126,
open3d, kornia, transformers, …) → `nvdiffrast` and `pytorch3d` built from
source against that torch. Expect this to take a while and produce a large
image (the existing one on this machine is ~27 GB).

The Dockerfile has **no `COPY`** — the repo only ever enters the container via
a bind mount at container run time, not at build time. That also means the
image has no `moveit_msgs` etc. by default beyond what's explicitly
`apt install`ed; check `docker exec vision dpkg -l | grep moveit` if the
pipeline's `CollisionObject`/planning-scene publishing (README §7) errors on
missing message packages, and add whatever's missing to the Dockerfile's ROS
apt layer.

---

## 5. Docker — creating the `vision` container

**This step exists nowhere in the repo's scripts** — `scripts/launch_pipeline*.sh`
only ever `docker stop`/`docker start` a container named `vision`; they never
create one. Create it once per machine:

```bash
docker run -d \
  --name vision \
  --gpus all \
  --network host \
  -v "$HOME/Masterthesis-vision:/workspace/MasterThesis" \
  masterthesis-vision:latest \
  tail -f /dev/null
```

Why each flag:
- `--network host` — the pipeline (in the container) and the camera/viz nodes
  (on the host) talk over ROS 2 / FastDDS, which relies on UDP
  multicast/dynamic ports for discovery; the launch scripts also set
  `FASTDDS_BUILTIN_TRANSPORTS=UDPv4`. Docker's default bridge network breaks
  this discovery — host networking is what makes `docker exec`'d pipeline
  nodes visible to host-side subscribers (and vice versa) at all.
- `-v ...:/workspace/MasterThesis` — the repo is never copied into the image;
  this bind mount is the *only* way the container sees your code (and picks
  up edits live, no rebuild needed for Python changes).
- `tail -f /dev/null` — keeps the container alive with nothing running; the
  launch scripts `docker exec -it` into it on demand.
- No `--gpus all` fallback: if `--gpus all` fails, re-check §1's
  `nvidia-ctk runtime configure` step.

Override the container name via `CONTAINER=other-name` if you run more than
one (all the launch scripts respect this env var).

### 5.1 First-time in-container build (repeat after any `external/` update)

Everything above only gets you a running, empty-of-build-artifacts container.
`fp_debug_msgs` (this repo's own submodule copy, not the `franka_ros2_ws` one
from §2) and FoundationPose's `mycpp` pybind11 extension are colcon/cmake
packages that must be built **from inside the container**, using the
container's own mount path — building on the host bakes host paths into
`CMakeCache.txt` and breaks the next in-container build (see
[external/README.md](../external/README.md#mycpp-and-fp_debug_msgs-build-inside-the-container-not-the-host)
if you ever hit a `CMakeCache.txt ... different directory` error; fix is
`rm -rf build install log` and rebuild).

```bash
docker exec -it vision bash -lc '
  source /opt/ros/humble/setup.bash
  source /opt/thesis-venv/bin/activate
  cd /workspace/MasterThesis
  colcon build
  source install/setup.bash
'

# sam2 is pip-installed editable, deliberately without its own pinned deps
# (its setup.py pins torch>=2.5.1 etc. — letting pip resolve here would risk
# touching the torch/numpy versions the rest of the stack depends on):
docker exec -it vision bash -lc '
  source /opt/thesis-venv/bin/activate
  cd /workspace/MasterThesis
  pip install --no-deps -e external/sam2
'
```

After this, `scripts/launch_pipeline.sh` (see
[docs/getting_started.md](getting_started.md)) sources
`/workspace/MasterThesis/install/setup.bash` automatically via the container's
`~/.bashrc` (written by the Dockerfile) — no need to re-source manually on
every `docker exec`.

---

## 6. End-to-end verification

```bash
# Host side
nvidia-smi                                   # GPU visible on host
ros2 pkg list | grep -E "mv_launch|fp_debug_msgs"
rs-enumerate-devices -s                      # RealSense variant only

# Container side
docker ps -a --filter name=vision            # container listed, "Up"
docker exec -it vision bash -lc "source /opt/thesis-venv/bin/activate && python3 -c 'import torch, open3d, cv2; print(torch.__version__, torch.cuda.is_available())'"
docker exec -it vision bash -lc "source /workspace/MasterThesis/install/setup.bash && ros2 interface show fp_debug_msgs/msg/DebugPoseItem"
```

`torch.cuda.is_available()` must print `True` — if it doesn't, the GPU
passthrough (`--gpus all` / NVIDIA Container Toolkit) isn't working; nothing
downstream in the container will work either. Then proceed to
[getting_started.md](getting_started.md) (3-ZED) or
[getting_started_realsense.md](getting_started_realsense.md) (RealSense trio)
for the calibrate → run workflow.

---

## 7. Troubleshooting

| Symptom | Likely cause |
|---|---|
| `docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]` | NVIDIA Container Toolkit not installed/configured — redo §1's `nvidia-ctk runtime configure` + `systemctl restart docker`. |
| Container up, but host nodes never see pipeline topics (or vice versa) | Container wasn't created with `--network host`, or `FASTDDS_BUILTIN_TRANSPORTS=UDPv4` isn't set for one side — check `docker inspect vision --format '{{.HostConfig.NetworkMode}}'`. |
| `AttributeError: module 'X' has no attribute 'Y'` for a package that exists both via apt and pip, or `cv2` import errors mentioning `_ARRAY_API` | pip/apt package shadowing inside `/opt/thesis-venv` (it's `--system-site-packages`) — see [external/README.md](../external/README.md#the-venv-why---system-site-packages). |
| `colcon build` fails with `CMakeCache.txt ... is different than the directory ... where CMakeCache.txt was created` | `colcon build` was run on the **host** at some point instead of inside the container — `rm -rf build install log` and rebuild from inside `docker exec`. |
| `ros2 launch mv_launch ...` can't find `zed_wrapper` / `realsense2_camera` | `zed-ros2-wrapper` / `realsense-ros` aren't built in `~/franka_ros2_ws`, or the ZED SDK / librealsense weren't installed before building them (§1.1, §2). |
| Pipeline errors publishing `moveit_msgs/CollisionObject` | `ros-humble-moveit-msgs` (or the full `ros-humble-moveit`) isn't installed in the container image — it's not in the base `Dockerfile.thesisnewcuda` apt list; `apt install` it in the running container or add it to the Dockerfile and rebuild. |
| Camera shows `CAMERA NOT DETECTED` in the host `cams` tmux window | USB enumeration flakiness — see [README §4 step 1](../README.md#4-camera-to-base-calibration) for the `--cam-ids` workaround. |
