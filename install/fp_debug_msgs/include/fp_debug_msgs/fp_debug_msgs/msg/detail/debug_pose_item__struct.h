// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from fp_debug_msgs:msg/DebugPoseItem.idl
// generated code does not contain a copyright notice

#ifndef FP_DEBUG_MSGS__MSG__DETAIL__DEBUG_POSE_ITEM__STRUCT_H_
#define FP_DEBUG_MSGS__MSG__DETAIL__DEBUG_POSE_ITEM__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'object_id'
// Member 'mode'
#include "rosidl_runtime_c/string.h"
// Member 'mask'
#include "fp_debug_msgs/msg/detail/debug_mask_crop__struct.h"
// Member 'pose_camera'
// Member 'pose_base'
#include "geometry_msgs/msg/detail/pose__struct.h"

/// Struct defined in msg/DebugPoseItem in the package fp_debug_msgs.
typedef struct fp_debug_msgs__msg__DebugPoseItem
{
  rosidl_runtime_c__String object_id;
  rosidl_runtime_c__String mode;
  float score;
  bool has_bbox;
  int32_t bbox_xyxy[4];
  bool has_mask;
  fp_debug_msgs__msg__DebugMaskCrop mask;
  geometry_msgs__msg__Pose pose_camera;
  geometry_msgs__msg__Pose pose_base;
  float axis_len_m;
} fp_debug_msgs__msg__DebugPoseItem;

// Struct for a sequence of fp_debug_msgs__msg__DebugPoseItem.
typedef struct fp_debug_msgs__msg__DebugPoseItem__Sequence
{
  fp_debug_msgs__msg__DebugPoseItem * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} fp_debug_msgs__msg__DebugPoseItem__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // FP_DEBUG_MSGS__MSG__DETAIL__DEBUG_POSE_ITEM__STRUCT_H_
