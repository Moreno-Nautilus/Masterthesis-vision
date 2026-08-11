# Third-party Dependencies (`external/`)

The heavy perception repositories live under `external/` so this repo stays
small and the thesis changes stay visible. Most are git submodules; Cutie is
cloned manually because it is git-ignored in this repo.

## What Lives Here

| Path | Source | How this repo uses it |
|---|---|---|
| `external/FoundationPose` | `https://github.com/NVlabs/FoundationPose.git` | Pinned submodule plus the local thesis patch in `patches/FoundationPose_thesis_changes.patch`. Imported by adding the repo root to `sys.path` at runtime. |
| `external/dinov2` | `https://github.com/facebookresearch/dinov2.git` | Pinned submodule kept as local source/reference. The running pipeline loads the selected DINOv2 backbone through `torch.hub`. No colcon build is needed for DINOv2. |
| `external/sam2` | `https://github.com/facebookresearch/sam2.git` | Pinned submodule installed editable inside the container with `pip install --no-deps -e external/sam2`. |
| `external/Cutie` | `https://github.com/hkchengrex/Cutie.git` | Manual clone, not a submodule. Imported by adding the checkout to `sys.path` in `src/perception/tracking/cutie_tracker.py`. |

`src/fp_debug_msgs` is not under `external/`, but it is also a submodule and is
part of the fresh-machine setup. Build it with colcon from the repo root inside
the Docker container:

```bash
colcon build --packages-select fp_debug_msgs
source install/setup.bash
```

## Checkout

From the repo root:

```bash
git submodule update --init --recursive
bash external/apply_patches.sh
```

`external/apply_patches.sh` applies the FoundationPose thesis patch and is safe
to rerun. If `src/fp_debug_msgs` fails to clone, the workstation probably needs
GitHub SSH access to `git@github.com:Moreno-Nautilus/fp_debug_msgs.git`, or the
submodule URL needs to be changed to an accessible HTTPS URL.

Cutie is cloned separately:

```bash
git clone https://github.com/hkchengrex/Cutie.git external/Cutie
git -C external/Cutie checkout ec5cdd4cf16f75c73ad785a2f96fb97dbad4125a
```

## Container Path

The canonical Docker mount is:

```text
/workspace/Masterthesis-vision
```

The current pipeline launch scripts still source and `cd` through the legacy
path `/workspace/MasterThesis`, so every new workstation should create this
compatibility symlink once inside the container:

```bash
ln -sfn /workspace/Masterthesis-vision /workspace/MasterThesis
```

## How these submodules actually get imported

None of these repos are meant to be `pip install`-ed the normal way (which would
try to pull their own pinned torch/numpy/etc. and clash with the versions this
project actually runs — see [Dockerfile.thesisnewcuda](../Dockerfile.thesisnewcuda)).
Instead:

| Repo | How it becomes importable |
|------|---------------------------|
| `external/FoundationPose` | No `setup.py`. Imported by adding the repo root to `sys.path` at runtime — see `_ensure_repo_on_path()` in [../src/perception/learned/FP/pose_foundation.py](../src/perception/learned/FP/pose_foundation.py). |
| `external/Cutie` | Same pattern, done at import time in [../src/perception/tracking/cutie_tracker.py](../src/perception/tracking/cutie_tracker.py) (module-level `sys.path.insert` before any `cutie.*` import). |
| `external/sam2` | Has a real `setup.py`, so it's `pip install`-ed **editable**, **without its pinned deps** (see below). |
| `external/dinov2` | Built via `colcon build` (it's set up as an `ament_python`-identifiable package) — see the workspace build step below. |

Because `FoundationPose`/`Cutie` rely on `sys.path` tricks against the *live*
checkout and `sam2` is installed *editable*, none of this can happen at `docker
build` time — the Dockerfile has no `COPY`, the repo only exists inside the
container via the runtime bind mount (`-v <repo>:/workspace/MasterThesis`). So
after the container is up, from `/workspace/MasterThesis`, run once (or after any
`external/` update):

```bash
pip install --no-deps -e external/sam2
```

`--no-deps` is deliberate: `sam2`'s `setup.py` pins `torch>=2.5.1` etc., which our
installed torch/numpy already satisfy — letting pip resolve deps here risks it
touching versions the rest of the stack depends on.

### The venv: why `--system-site-packages`

`/opt/thesis-venv` (created in the Dockerfile) uses
`python3 -m venv --system-site-packages`, not a fully isolated venv. This is
required for ROS 2: `rclpy`, `catkin_pkg`, `ament_index_python`, and the
`rosidl`/`colcon` build tooling are apt/Debian packages installed against the
*system* Python (`/usr/lib/python3/dist-packages`), not pip packages — there's no
supported way to `pip install rclpy`. A normal isolated venv can't see any of
that. `--system-site-packages` lets the venv layer the pip-only ML stack (torch,
open3d, sam2, ...) on top of the system ROS install instead of replacing it.

The cost: pip packages in the venv can shadow apt packages of the same import
name with a different, possibly incompatible, build. This already bit us once —
apt's `python3-opencv` (built against system NumPy 1.21) got shadowed by pip's
NumPy 2.x, crashing `cv2` on import (`_ARRAY_API not found`). Fixed by installing
`opencv-python` from pip too, so both `cv2` and `numpy` resolve to matching pip
builds. If a similar "works on host, breaks in container" or
`AttributeError: module 'X' has no attribute 'Y'` shows up for a package that
exists both via apt and pip, suspect this shadowing first.

### `mycpp` and `fp_debug_msgs`: build inside the container, not the host

`external/FoundationPose/mycpp` (a pybind11 module) and `../src/fp_debug_msgs`
(a ROS 2 `rosidl` message package) are `colcon`-built, ROS-style packages. They
**must** be built inside the container, from the container's own path
(`/workspace/MasterThesis`), not on the host:

```bash
cd /workspace/MasterThesis
colcon build
source install/setup.bash
```

Building on the host instead bakes host paths (e.g.
`/home/<user>/Masterthesis-vision/...`) into `build/*/CMakeCache.txt`, which then
breaks (`CMake Error: ... directory ... is different than the directory ...
where CMakeCache.txt was created`) the next time `colcon build` runs inside the
container, since the mount point differs
(`/workspace/MasterThesis` ≠ the host path). If you hit that error, wipe and
rebuild clean inside the container:

```bash
rm -rf build install log
colcon build
```

`external/sam2` and `external/FoundationPose/bundlesdf/mycuda` carry
`COLCON_IGNORE` markers (untracked, container-local files — not committed to the
submodules) so `colcon build` doesn't also try and fail to identify them as
colcon packages; they're installed via pip/sys.path as described above instead.
