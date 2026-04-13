// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from fp_debug_msgs:msg/DebugPoseItem.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "fp_debug_msgs/msg/detail/debug_pose_item__rosidl_typesupport_introspection_c.h"
#include "fp_debug_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "fp_debug_msgs/msg/detail/debug_pose_item__functions.h"
#include "fp_debug_msgs/msg/detail/debug_pose_item__struct.h"


// Include directives for member types
// Member `object_id`
// Member `mode`
#include "rosidl_runtime_c/string_functions.h"
// Member `mask`
#include "fp_debug_msgs/msg/debug_mask_crop.h"
// Member `mask`
#include "fp_debug_msgs/msg/detail/debug_mask_crop__rosidl_typesupport_introspection_c.h"
// Member `pose_camera`
// Member `pose_base`
#include "geometry_msgs/msg/pose.h"
// Member `pose_camera`
// Member `pose_base`
#include "geometry_msgs/msg/detail/pose__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void fp_debug_msgs__msg__DebugPoseItem__rosidl_typesupport_introspection_c__DebugPoseItem_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  fp_debug_msgs__msg__DebugPoseItem__init(message_memory);
}

void fp_debug_msgs__msg__DebugPoseItem__rosidl_typesupport_introspection_c__DebugPoseItem_fini_function(void * message_memory)
{
  fp_debug_msgs__msg__DebugPoseItem__fini(message_memory);
}

size_t fp_debug_msgs__msg__DebugPoseItem__rosidl_typesupport_introspection_c__size_function__DebugPoseItem__bbox_xyxy(
  const void * untyped_member)
{
  (void)untyped_member;
  return 4;
}

const void * fp_debug_msgs__msg__DebugPoseItem__rosidl_typesupport_introspection_c__get_const_function__DebugPoseItem__bbox_xyxy(
  const void * untyped_member, size_t index)
{
  const int32_t * member =
    (const int32_t *)(untyped_member);
  return &member[index];
}

void * fp_debug_msgs__msg__DebugPoseItem__rosidl_typesupport_introspection_c__get_function__DebugPoseItem__bbox_xyxy(
  void * untyped_member, size_t index)
{
  int32_t * member =
    (int32_t *)(untyped_member);
  return &member[index];
}

void fp_debug_msgs__msg__DebugPoseItem__rosidl_typesupport_introspection_c__fetch_function__DebugPoseItem__bbox_xyxy(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const int32_t * item =
    ((const int32_t *)
    fp_debug_msgs__msg__DebugPoseItem__rosidl_typesupport_introspection_c__get_const_function__DebugPoseItem__bbox_xyxy(untyped_member, index));
  int32_t * value =
    (int32_t *)(untyped_value);
  *value = *item;
}

void fp_debug_msgs__msg__DebugPoseItem__rosidl_typesupport_introspection_c__assign_function__DebugPoseItem__bbox_xyxy(
  void * untyped_member, size_t index, const void * untyped_value)
{
  int32_t * item =
    ((int32_t *)
    fp_debug_msgs__msg__DebugPoseItem__rosidl_typesupport_introspection_c__get_function__DebugPoseItem__bbox_xyxy(untyped_member, index));
  const int32_t * value =
    (const int32_t *)(untyped_value);
  *item = *value;
}

static rosidl_typesupport_introspection_c__MessageMember fp_debug_msgs__msg__DebugPoseItem__rosidl_typesupport_introspection_c__DebugPoseItem_message_member_array[10] = {
  {
    "object_id",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(fp_debug_msgs__msg__DebugPoseItem, object_id),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "mode",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(fp_debug_msgs__msg__DebugPoseItem, mode),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "score",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(fp_debug_msgs__msg__DebugPoseItem, score),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "has_bbox",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(fp_debug_msgs__msg__DebugPoseItem, has_bbox),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "bbox_xyxy",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    4,  // array size
    false,  // is upper bound
    offsetof(fp_debug_msgs__msg__DebugPoseItem, bbox_xyxy),  // bytes offset in struct
    NULL,  // default value
    fp_debug_msgs__msg__DebugPoseItem__rosidl_typesupport_introspection_c__size_function__DebugPoseItem__bbox_xyxy,  // size() function pointer
    fp_debug_msgs__msg__DebugPoseItem__rosidl_typesupport_introspection_c__get_const_function__DebugPoseItem__bbox_xyxy,  // get_const(index) function pointer
    fp_debug_msgs__msg__DebugPoseItem__rosidl_typesupport_introspection_c__get_function__DebugPoseItem__bbox_xyxy,  // get(index) function pointer
    fp_debug_msgs__msg__DebugPoseItem__rosidl_typesupport_introspection_c__fetch_function__DebugPoseItem__bbox_xyxy,  // fetch(index, &value) function pointer
    fp_debug_msgs__msg__DebugPoseItem__rosidl_typesupport_introspection_c__assign_function__DebugPoseItem__bbox_xyxy,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "has_mask",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(fp_debug_msgs__msg__DebugPoseItem, has_mask),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "mask",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(fp_debug_msgs__msg__DebugPoseItem, mask),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "pose_camera",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(fp_debug_msgs__msg__DebugPoseItem, pose_camera),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "pose_base",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(fp_debug_msgs__msg__DebugPoseItem, pose_base),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "axis_len_m",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(fp_debug_msgs__msg__DebugPoseItem, axis_len_m),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers fp_debug_msgs__msg__DebugPoseItem__rosidl_typesupport_introspection_c__DebugPoseItem_message_members = {
  "fp_debug_msgs__msg",  // message namespace
  "DebugPoseItem",  // message name
  10,  // number of fields
  sizeof(fp_debug_msgs__msg__DebugPoseItem),
  fp_debug_msgs__msg__DebugPoseItem__rosidl_typesupport_introspection_c__DebugPoseItem_message_member_array,  // message members
  fp_debug_msgs__msg__DebugPoseItem__rosidl_typesupport_introspection_c__DebugPoseItem_init_function,  // function to initialize message memory (memory has to be allocated)
  fp_debug_msgs__msg__DebugPoseItem__rosidl_typesupport_introspection_c__DebugPoseItem_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t fp_debug_msgs__msg__DebugPoseItem__rosidl_typesupport_introspection_c__DebugPoseItem_message_type_support_handle = {
  0,
  &fp_debug_msgs__msg__DebugPoseItem__rosidl_typesupport_introspection_c__DebugPoseItem_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_fp_debug_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, fp_debug_msgs, msg, DebugPoseItem)() {
  fp_debug_msgs__msg__DebugPoseItem__rosidl_typesupport_introspection_c__DebugPoseItem_message_member_array[6].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, fp_debug_msgs, msg, DebugMaskCrop)();
  fp_debug_msgs__msg__DebugPoseItem__rosidl_typesupport_introspection_c__DebugPoseItem_message_member_array[7].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, geometry_msgs, msg, Pose)();
  fp_debug_msgs__msg__DebugPoseItem__rosidl_typesupport_introspection_c__DebugPoseItem_message_member_array[8].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, geometry_msgs, msg, Pose)();
  if (!fp_debug_msgs__msg__DebugPoseItem__rosidl_typesupport_introspection_c__DebugPoseItem_message_type_support_handle.typesupport_identifier) {
    fp_debug_msgs__msg__DebugPoseItem__rosidl_typesupport_introspection_c__DebugPoseItem_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &fp_debug_msgs__msg__DebugPoseItem__rosidl_typesupport_introspection_c__DebugPoseItem_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
