builtin_interfaces/Time stamp
string cam_id

int32 max_candidate_draw
bool show_axes

int32[] roi_polygon_xy_flat
bool has_tiny_roi
int32[4] tiny_roi_xyxy

bool update_sam
bool update_dino

fp_debug_msgs/DebugCandidate[] sam_candidates
fp_debug_msgs/DebugCandidate[] dino_ranked_candidates
fp_debug_msgs/DebugPoseItem[] pose_items

# Tracking mask from Cutie (when in track mode)
bool has_track_mask
int32[4] track_mask_bbox_xyxy
fp_debug_msgs/DebugMaskCrop track_mask
string track_object_id
float32 track_icp_fitness
float32 track_icp_rmse_mm
