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

## One-Time Container Builds

The Docker image provides `/opt/thesis-venv`, Torch, PyTorch3D, nvdiffrast, ROS 2
Humble, and the common Python dependencies. After the container is created and
the repo is mounted, run the repo-local setup once inside the container:

```bash
ln -sfn /workspace/Masterthesis-vision /workspace/MasterThesis

cd /workspace/Masterthesis-vision
source /opt/ros/humble/setup.bash
source /opt/thesis-venv/bin/activate

pip install --no-deps -e external/sam2

cd external/FoundationPose
bash build_all_conda.sh
cd /workspace/Masterthesis-vision

colcon build --packages-select fp_debug_msgs
source install/setup.bash
```

Use `build_all_conda.sh` for FoundationPose even though the environment is a
venv; the upstream `build_all.sh` assumes the original FoundationPose `/kaolin`
Docker layout. The script builds FoundationPose's local C++/CUDA helpers,
including the `mycpp` module imported by `Utils.py`.

`external/COLCON_IGNORE` is tracked on purpose. It prevents `colcon build` from
trying to identify or build all third-party code under `external/`; only this
repo's ROS packages, such as `src/fp_debug_msgs`, should be built by colcon.
The resulting git-ignored `install/` directory is also sourced by the host launch
scripts so their visualizer windows can import `fp_debug_msgs`.

## Python Environment

`/opt/thesis-venv` is created in the Dockerfile with:

```text
python3 -m venv --system-site-packages /opt/thesis-venv
```

That is intentional. ROS Python packages such as `rclpy`,
`ament_index_python`, and the `rosidl` tooling come from Ubuntu/ROS apt packages
in the system Python. The venv layers the pip ML stack on top while still seeing
those ROS packages.

## Weights

Model weights are not stored in git. The main README lists the download links
and exact target paths for SAM2, FoundationPose, Cutie, DINOv2, and
Grounding-DINO.
