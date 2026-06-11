# Third-party dependencies (`external/`)

The pose / segmentation pipeline depends on several upstream repositories. They are
**not vendored** into this repo — they are pinned as git submodules (or, for Cutie,
cloned manually) so this repo stays small and it is always clear what was changed
versus upstream.

## Submodules

| Path                     | Upstream                                          | Notes |
|--------------------------|---------------------------------------------------|-------|
| `external/FoundationPose`| https://github.com/NVlabs/FoundationPose.git      | Pinned commit + local thesis patch (see below) |
| `external/dinov2`        | https://github.com/facebookresearch/dinov2.git    | Clean, pinned to upstream commit |
| `external/sam2`          | https://github.com/facebookresearch/sam2.git      | Clean, pinned to upstream commit |
| `external/sam3`          | https://github.com/facebookresearch/sam3.git      | Clean, pinned to upstream commit |

### Setup

```bash
git submodule update --init --recursive
bash external/apply_patches.sh    # applies the FoundationPose thesis changes
```

### FoundationPose local changes

Our modifications to FoundationPose are kept as a patch rather than committed into the
submodule, because the submodule only tracks an upstream commit pointer:

- `patches/FoundationPose_thesis_changes.patch`

`external/apply_patches.sh` applies it on top of the pinned upstream commit and is
idempotent (safe to re-run). If you change FoundationPose, regenerate the patch with:

```bash
git -C external/FoundationPose diff > patches/FoundationPose_thesis_changes.patch
```

## Cutie (not a submodule)

`external/Cutie` has no public upstream remote, so it cannot be a submodule. It is
git-ignored. Obtain it separately and check out the pinned commit:

```
ec5cdd4cf16f75c73ad785a2f96fb97dbad4125a
```
