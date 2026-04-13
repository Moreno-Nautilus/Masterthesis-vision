// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from fp_debug_msgs:msg/DebugCandidate.idl
// generated code does not contain a copyright notice

#ifndef FP_DEBUG_MSGS__MSG__DETAIL__DEBUG_CANDIDATE__STRUCT_H_
#define FP_DEBUG_MSGS__MSG__DETAIL__DEBUG_CANDIDATE__STRUCT_H_

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
#include "rosidl_runtime_c/string.h"
// Member 'mask'
#include "fp_debug_msgs/msg/detail/debug_mask_crop__struct.h"

/// Struct defined in msg/DebugCandidate in the package fp_debug_msgs.
typedef struct fp_debug_msgs__msg__DebugCandidate
{
  rosidl_runtime_c__String object_id;
  float score;
  int32_t bbox_xyxy[4];
  bool has_mask;
  fp_debug_msgs__msg__DebugMaskCrop mask;
} fp_debug_msgs__msg__DebugCandidate;

// Struct for a sequence of fp_debug_msgs__msg__DebugCandidate.
typedef struct fp_debug_msgs__msg__DebugCandidate__Sequence
{
  fp_debug_msgs__msg__DebugCandidate * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} fp_debug_msgs__msg__DebugCandidate__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // FP_DEBUG_MSGS__MSG__DETAIL__DEBUG_CANDIDATE__STRUCT_H_
