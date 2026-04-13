// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from fp_debug_msgs:msg/DebugMaskCrop.idl
// generated code does not contain a copyright notice

#ifndef FP_DEBUG_MSGS__MSG__DETAIL__DEBUG_MASK_CROP__STRUCT_H_
#define FP_DEBUG_MSGS__MSG__DETAIL__DEBUG_MASK_CROP__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'data'
#include "rosidl_runtime_c/primitives_sequence.h"

/// Struct defined in msg/DebugMaskCrop in the package fp_debug_msgs.
typedef struct fp_debug_msgs__msg__DebugMaskCrop
{
  uint32_t width;
  uint32_t height;
  rosidl_runtime_c__uint8__Sequence data;
} fp_debug_msgs__msg__DebugMaskCrop;

// Struct for a sequence of fp_debug_msgs__msg__DebugMaskCrop.
typedef struct fp_debug_msgs__msg__DebugMaskCrop__Sequence
{
  fp_debug_msgs__msg__DebugMaskCrop * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} fp_debug_msgs__msg__DebugMaskCrop__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // FP_DEBUG_MSGS__MSG__DETAIL__DEBUG_MASK_CROP__STRUCT_H_
