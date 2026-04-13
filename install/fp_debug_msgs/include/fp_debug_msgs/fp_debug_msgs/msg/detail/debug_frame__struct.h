// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from fp_debug_msgs:msg/DebugFrame.idl
// generated code does not contain a copyright notice

#ifndef FP_DEBUG_MSGS__MSG__DETAIL__DEBUG_FRAME__STRUCT_H_
#define FP_DEBUG_MSGS__MSG__DETAIL__DEBUG_FRAME__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'stamp'
#include "builtin_interfaces/msg/detail/time__struct.h"
// Member 'cam_id'
#include "rosidl_runtime_c/string.h"
// Member 'roi_polygon_xy_flat'
#include "rosidl_runtime_c/primitives_sequence.h"
// Member 'sam_candidates'
// Member 'dino_ranked_candidates'
#include "fp_debug_msgs/msg/detail/debug_candidate__struct.h"
// Member 'pose_items'
#include "fp_debug_msgs/msg/detail/debug_pose_item__struct.h"

/// Struct defined in msg/DebugFrame in the package fp_debug_msgs.
typedef struct fp_debug_msgs__msg__DebugFrame
{
  builtin_interfaces__msg__Time stamp;
  rosidl_runtime_c__String cam_id;
  int32_t max_candidate_draw;
  bool show_axes;
  rosidl_runtime_c__int32__Sequence roi_polygon_xy_flat;
  bool has_tiny_roi;
  int32_t tiny_roi_xyxy[4];
  bool update_sam;
  bool update_dino;
  fp_debug_msgs__msg__DebugCandidate__Sequence sam_candidates;
  fp_debug_msgs__msg__DebugCandidate__Sequence dino_ranked_candidates;
  fp_debug_msgs__msg__DebugPoseItem__Sequence pose_items;
} fp_debug_msgs__msg__DebugFrame;

// Struct for a sequence of fp_debug_msgs__msg__DebugFrame.
typedef struct fp_debug_msgs__msg__DebugFrame__Sequence
{
  fp_debug_msgs__msg__DebugFrame * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} fp_debug_msgs__msg__DebugFrame__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // FP_DEBUG_MSGS__MSG__DETAIL__DEBUG_FRAME__STRUCT_H_
