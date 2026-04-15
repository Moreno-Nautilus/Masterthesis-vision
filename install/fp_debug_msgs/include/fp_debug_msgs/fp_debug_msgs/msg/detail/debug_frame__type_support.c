// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from fp_debug_msgs:msg/DebugFrame.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "fp_debug_msgs/msg/detail/debug_frame__rosidl_typesupport_introspection_c.h"
#include "fp_debug_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "fp_debug_msgs/msg/detail/debug_frame__functions.h"
#include "fp_debug_msgs/msg/detail/debug_frame__struct.h"


// Include directives for member types
// Member `stamp`
#include "builtin_interfaces/msg/time.h"
// Member `stamp`
#include "builtin_interfaces/msg/detail/time__rosidl_typesupport_introspection_c.h"
// Member `cam_id`
// Member `track_object_id`
#include "rosidl_runtime_c/string_functions.h"
// Member `roi_polygon_xy_flat`
#include "rosidl_runtime_c/primitives_sequence_functions.h"
// Member `sam_candidates`
// Member `dino_ranked_candidates`
#include "fp_debug_msgs/msg/debug_candidate.h"
// Member `sam_candidates`
// Member `dino_ranked_candidates`
#include "fp_debug_msgs/msg/detail/debug_candidate__rosidl_typesupport_introspection_c.h"
// Member `pose_items`
#include "fp_debug_msgs/msg/debug_pose_item.h"
// Member `pose_items`
#include "fp_debug_msgs/msg/detail/debug_pose_item__rosidl_typesupport_introspection_c.h"
// Member `track_mask`
#include "fp_debug_msgs/msg/debug_mask_crop.h"
// Member `track_mask`
#include "fp_debug_msgs/msg/detail/debug_mask_crop__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__DebugFrame_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  fp_debug_msgs__msg__DebugFrame__init(message_memory);
}

void fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__DebugFrame_fini_function(void * message_memory)
{
  fp_debug_msgs__msg__DebugFrame__fini(message_memory);
}

size_t fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__size_function__DebugFrame__roi_polygon_xy_flat(
  const void * untyped_member)
{
  const rosidl_runtime_c__int32__Sequence * member =
    (const rosidl_runtime_c__int32__Sequence *)(untyped_member);
  return member->size;
}

const void * fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__get_const_function__DebugFrame__roi_polygon_xy_flat(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__int32__Sequence * member =
    (const rosidl_runtime_c__int32__Sequence *)(untyped_member);
  return &member->data[index];
}

void * fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__get_function__DebugFrame__roi_polygon_xy_flat(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__int32__Sequence * member =
    (rosidl_runtime_c__int32__Sequence *)(untyped_member);
  return &member->data[index];
}

void fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__fetch_function__DebugFrame__roi_polygon_xy_flat(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const int32_t * item =
    ((const int32_t *)
    fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__get_const_function__DebugFrame__roi_polygon_xy_flat(untyped_member, index));
  int32_t * value =
    (int32_t *)(untyped_value);
  *value = *item;
}

void fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__assign_function__DebugFrame__roi_polygon_xy_flat(
  void * untyped_member, size_t index, const void * untyped_value)
{
  int32_t * item =
    ((int32_t *)
    fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__get_function__DebugFrame__roi_polygon_xy_flat(untyped_member, index));
  const int32_t * value =
    (const int32_t *)(untyped_value);
  *item = *value;
}

bool fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__resize_function__DebugFrame__roi_polygon_xy_flat(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__int32__Sequence * member =
    (rosidl_runtime_c__int32__Sequence *)(untyped_member);
  rosidl_runtime_c__int32__Sequence__fini(member);
  return rosidl_runtime_c__int32__Sequence__init(member, size);
}

size_t fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__size_function__DebugFrame__tiny_roi_xyxy(
  const void * untyped_member)
{
  (void)untyped_member;
  return 4;
}

const void * fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__get_const_function__DebugFrame__tiny_roi_xyxy(
  const void * untyped_member, size_t index)
{
  const int32_t * member =
    (const int32_t *)(untyped_member);
  return &member[index];
}

void * fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__get_function__DebugFrame__tiny_roi_xyxy(
  void * untyped_member, size_t index)
{
  int32_t * member =
    (int32_t *)(untyped_member);
  return &member[index];
}

void fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__fetch_function__DebugFrame__tiny_roi_xyxy(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const int32_t * item =
    ((const int32_t *)
    fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__get_const_function__DebugFrame__tiny_roi_xyxy(untyped_member, index));
  int32_t * value =
    (int32_t *)(untyped_value);
  *value = *item;
}

void fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__assign_function__DebugFrame__tiny_roi_xyxy(
  void * untyped_member, size_t index, const void * untyped_value)
{
  int32_t * item =
    ((int32_t *)
    fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__get_function__DebugFrame__tiny_roi_xyxy(untyped_member, index));
  const int32_t * value =
    (const int32_t *)(untyped_value);
  *item = *value;
}

size_t fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__size_function__DebugFrame__sam_candidates(
  const void * untyped_member)
{
  const fp_debug_msgs__msg__DebugCandidate__Sequence * member =
    (const fp_debug_msgs__msg__DebugCandidate__Sequence *)(untyped_member);
  return member->size;
}

const void * fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__get_const_function__DebugFrame__sam_candidates(
  const void * untyped_member, size_t index)
{
  const fp_debug_msgs__msg__DebugCandidate__Sequence * member =
    (const fp_debug_msgs__msg__DebugCandidate__Sequence *)(untyped_member);
  return &member->data[index];
}

void * fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__get_function__DebugFrame__sam_candidates(
  void * untyped_member, size_t index)
{
  fp_debug_msgs__msg__DebugCandidate__Sequence * member =
    (fp_debug_msgs__msg__DebugCandidate__Sequence *)(untyped_member);
  return &member->data[index];
}

void fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__fetch_function__DebugFrame__sam_candidates(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const fp_debug_msgs__msg__DebugCandidate * item =
    ((const fp_debug_msgs__msg__DebugCandidate *)
    fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__get_const_function__DebugFrame__sam_candidates(untyped_member, index));
  fp_debug_msgs__msg__DebugCandidate * value =
    (fp_debug_msgs__msg__DebugCandidate *)(untyped_value);
  *value = *item;
}

void fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__assign_function__DebugFrame__sam_candidates(
  void * untyped_member, size_t index, const void * untyped_value)
{
  fp_debug_msgs__msg__DebugCandidate * item =
    ((fp_debug_msgs__msg__DebugCandidate *)
    fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__get_function__DebugFrame__sam_candidates(untyped_member, index));
  const fp_debug_msgs__msg__DebugCandidate * value =
    (const fp_debug_msgs__msg__DebugCandidate *)(untyped_value);
  *item = *value;
}

bool fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__resize_function__DebugFrame__sam_candidates(
  void * untyped_member, size_t size)
{
  fp_debug_msgs__msg__DebugCandidate__Sequence * member =
    (fp_debug_msgs__msg__DebugCandidate__Sequence *)(untyped_member);
  fp_debug_msgs__msg__DebugCandidate__Sequence__fini(member);
  return fp_debug_msgs__msg__DebugCandidate__Sequence__init(member, size);
}

size_t fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__size_function__DebugFrame__dino_ranked_candidates(
  const void * untyped_member)
{
  const fp_debug_msgs__msg__DebugCandidate__Sequence * member =
    (const fp_debug_msgs__msg__DebugCandidate__Sequence *)(untyped_member);
  return member->size;
}

const void * fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__get_const_function__DebugFrame__dino_ranked_candidates(
  const void * untyped_member, size_t index)
{
  const fp_debug_msgs__msg__DebugCandidate__Sequence * member =
    (const fp_debug_msgs__msg__DebugCandidate__Sequence *)(untyped_member);
  return &member->data[index];
}

void * fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__get_function__DebugFrame__dino_ranked_candidates(
  void * untyped_member, size_t index)
{
  fp_debug_msgs__msg__DebugCandidate__Sequence * member =
    (fp_debug_msgs__msg__DebugCandidate__Sequence *)(untyped_member);
  return &member->data[index];
}

void fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__fetch_function__DebugFrame__dino_ranked_candidates(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const fp_debug_msgs__msg__DebugCandidate * item =
    ((const fp_debug_msgs__msg__DebugCandidate *)
    fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__get_const_function__DebugFrame__dino_ranked_candidates(untyped_member, index));
  fp_debug_msgs__msg__DebugCandidate * value =
    (fp_debug_msgs__msg__DebugCandidate *)(untyped_value);
  *value = *item;
}

void fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__assign_function__DebugFrame__dino_ranked_candidates(
  void * untyped_member, size_t index, const void * untyped_value)
{
  fp_debug_msgs__msg__DebugCandidate * item =
    ((fp_debug_msgs__msg__DebugCandidate *)
    fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__get_function__DebugFrame__dino_ranked_candidates(untyped_member, index));
  const fp_debug_msgs__msg__DebugCandidate * value =
    (const fp_debug_msgs__msg__DebugCandidate *)(untyped_value);
  *item = *value;
}

bool fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__resize_function__DebugFrame__dino_ranked_candidates(
  void * untyped_member, size_t size)
{
  fp_debug_msgs__msg__DebugCandidate__Sequence * member =
    (fp_debug_msgs__msg__DebugCandidate__Sequence *)(untyped_member);
  fp_debug_msgs__msg__DebugCandidate__Sequence__fini(member);
  return fp_debug_msgs__msg__DebugCandidate__Sequence__init(member, size);
}

size_t fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__size_function__DebugFrame__pose_items(
  const void * untyped_member)
{
  const fp_debug_msgs__msg__DebugPoseItem__Sequence * member =
    (const fp_debug_msgs__msg__DebugPoseItem__Sequence *)(untyped_member);
  return member->size;
}

const void * fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__get_const_function__DebugFrame__pose_items(
  const void * untyped_member, size_t index)
{
  const fp_debug_msgs__msg__DebugPoseItem__Sequence * member =
    (const fp_debug_msgs__msg__DebugPoseItem__Sequence *)(untyped_member);
  return &member->data[index];
}

void * fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__get_function__DebugFrame__pose_items(
  void * untyped_member, size_t index)
{
  fp_debug_msgs__msg__DebugPoseItem__Sequence * member =
    (fp_debug_msgs__msg__DebugPoseItem__Sequence *)(untyped_member);
  return &member->data[index];
}

void fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__fetch_function__DebugFrame__pose_items(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const fp_debug_msgs__msg__DebugPoseItem * item =
    ((const fp_debug_msgs__msg__DebugPoseItem *)
    fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__get_const_function__DebugFrame__pose_items(untyped_member, index));
  fp_debug_msgs__msg__DebugPoseItem * value =
    (fp_debug_msgs__msg__DebugPoseItem *)(untyped_value);
  *value = *item;
}

void fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__assign_function__DebugFrame__pose_items(
  void * untyped_member, size_t index, const void * untyped_value)
{
  fp_debug_msgs__msg__DebugPoseItem * item =
    ((fp_debug_msgs__msg__DebugPoseItem *)
    fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__get_function__DebugFrame__pose_items(untyped_member, index));
  const fp_debug_msgs__msg__DebugPoseItem * value =
    (const fp_debug_msgs__msg__DebugPoseItem *)(untyped_value);
  *item = *value;
}

bool fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__resize_function__DebugFrame__pose_items(
  void * untyped_member, size_t size)
{
  fp_debug_msgs__msg__DebugPoseItem__Sequence * member =
    (fp_debug_msgs__msg__DebugPoseItem__Sequence *)(untyped_member);
  fp_debug_msgs__msg__DebugPoseItem__Sequence__fini(member);
  return fp_debug_msgs__msg__DebugPoseItem__Sequence__init(member, size);
}

size_t fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__size_function__DebugFrame__track_mask_bbox_xyxy(
  const void * untyped_member)
{
  (void)untyped_member;
  return 4;
}

const void * fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__get_const_function__DebugFrame__track_mask_bbox_xyxy(
  const void * untyped_member, size_t index)
{
  const int32_t * member =
    (const int32_t *)(untyped_member);
  return &member[index];
}

void * fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__get_function__DebugFrame__track_mask_bbox_xyxy(
  void * untyped_member, size_t index)
{
  int32_t * member =
    (int32_t *)(untyped_member);
  return &member[index];
}

void fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__fetch_function__DebugFrame__track_mask_bbox_xyxy(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const int32_t * item =
    ((const int32_t *)
    fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__get_const_function__DebugFrame__track_mask_bbox_xyxy(untyped_member, index));
  int32_t * value =
    (int32_t *)(untyped_value);
  *value = *item;
}

void fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__assign_function__DebugFrame__track_mask_bbox_xyxy(
  void * untyped_member, size_t index, const void * untyped_value)
{
  int32_t * item =
    ((int32_t *)
    fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__get_function__DebugFrame__track_mask_bbox_xyxy(untyped_member, index));
  const int32_t * value =
    (const int32_t *)(untyped_value);
  *item = *value;
}

static rosidl_typesupport_introspection_c__MessageMember fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__DebugFrame_message_member_array[18] = {
  {
    "stamp",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(fp_debug_msgs__msg__DebugFrame, stamp),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "cam_id",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(fp_debug_msgs__msg__DebugFrame, cam_id),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "max_candidate_draw",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(fp_debug_msgs__msg__DebugFrame, max_candidate_draw),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "show_axes",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(fp_debug_msgs__msg__DebugFrame, show_axes),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "roi_polygon_xy_flat",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(fp_debug_msgs__msg__DebugFrame, roi_polygon_xy_flat),  // bytes offset in struct
    NULL,  // default value
    fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__size_function__DebugFrame__roi_polygon_xy_flat,  // size() function pointer
    fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__get_const_function__DebugFrame__roi_polygon_xy_flat,  // get_const(index) function pointer
    fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__get_function__DebugFrame__roi_polygon_xy_flat,  // get(index) function pointer
    fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__fetch_function__DebugFrame__roi_polygon_xy_flat,  // fetch(index, &value) function pointer
    fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__assign_function__DebugFrame__roi_polygon_xy_flat,  // assign(index, value) function pointer
    fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__resize_function__DebugFrame__roi_polygon_xy_flat  // resize(index) function pointer
  },
  {
    "has_tiny_roi",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(fp_debug_msgs__msg__DebugFrame, has_tiny_roi),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "tiny_roi_xyxy",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    4,  // array size
    false,  // is upper bound
    offsetof(fp_debug_msgs__msg__DebugFrame, tiny_roi_xyxy),  // bytes offset in struct
    NULL,  // default value
    fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__size_function__DebugFrame__tiny_roi_xyxy,  // size() function pointer
    fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__get_const_function__DebugFrame__tiny_roi_xyxy,  // get_const(index) function pointer
    fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__get_function__DebugFrame__tiny_roi_xyxy,  // get(index) function pointer
    fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__fetch_function__DebugFrame__tiny_roi_xyxy,  // fetch(index, &value) function pointer
    fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__assign_function__DebugFrame__tiny_roi_xyxy,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "update_sam",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(fp_debug_msgs__msg__DebugFrame, update_sam),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "update_dino",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(fp_debug_msgs__msg__DebugFrame, update_dino),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "sam_candidates",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(fp_debug_msgs__msg__DebugFrame, sam_candidates),  // bytes offset in struct
    NULL,  // default value
    fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__size_function__DebugFrame__sam_candidates,  // size() function pointer
    fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__get_const_function__DebugFrame__sam_candidates,  // get_const(index) function pointer
    fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__get_function__DebugFrame__sam_candidates,  // get(index) function pointer
    fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__fetch_function__DebugFrame__sam_candidates,  // fetch(index, &value) function pointer
    fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__assign_function__DebugFrame__sam_candidates,  // assign(index, value) function pointer
    fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__resize_function__DebugFrame__sam_candidates  // resize(index) function pointer
  },
  {
    "dino_ranked_candidates",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(fp_debug_msgs__msg__DebugFrame, dino_ranked_candidates),  // bytes offset in struct
    NULL,  // default value
    fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__size_function__DebugFrame__dino_ranked_candidates,  // size() function pointer
    fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__get_const_function__DebugFrame__dino_ranked_candidates,  // get_const(index) function pointer
    fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__get_function__DebugFrame__dino_ranked_candidates,  // get(index) function pointer
    fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__fetch_function__DebugFrame__dino_ranked_candidates,  // fetch(index, &value) function pointer
    fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__assign_function__DebugFrame__dino_ranked_candidates,  // assign(index, value) function pointer
    fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__resize_function__DebugFrame__dino_ranked_candidates  // resize(index) function pointer
  },
  {
    "pose_items",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(fp_debug_msgs__msg__DebugFrame, pose_items),  // bytes offset in struct
    NULL,  // default value
    fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__size_function__DebugFrame__pose_items,  // size() function pointer
    fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__get_const_function__DebugFrame__pose_items,  // get_const(index) function pointer
    fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__get_function__DebugFrame__pose_items,  // get(index) function pointer
    fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__fetch_function__DebugFrame__pose_items,  // fetch(index, &value) function pointer
    fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__assign_function__DebugFrame__pose_items,  // assign(index, value) function pointer
    fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__resize_function__DebugFrame__pose_items  // resize(index) function pointer
  },
  {
    "has_track_mask",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(fp_debug_msgs__msg__DebugFrame, has_track_mask),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "track_mask_bbox_xyxy",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    4,  // array size
    false,  // is upper bound
    offsetof(fp_debug_msgs__msg__DebugFrame, track_mask_bbox_xyxy),  // bytes offset in struct
    NULL,  // default value
    fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__size_function__DebugFrame__track_mask_bbox_xyxy,  // size() function pointer
    fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__get_const_function__DebugFrame__track_mask_bbox_xyxy,  // get_const(index) function pointer
    fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__get_function__DebugFrame__track_mask_bbox_xyxy,  // get(index) function pointer
    fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__fetch_function__DebugFrame__track_mask_bbox_xyxy,  // fetch(index, &value) function pointer
    fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__assign_function__DebugFrame__track_mask_bbox_xyxy,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "track_mask",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(fp_debug_msgs__msg__DebugFrame, track_mask),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "track_object_id",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(fp_debug_msgs__msg__DebugFrame, track_object_id),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "track_icp_fitness",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(fp_debug_msgs__msg__DebugFrame, track_icp_fitness),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "track_icp_rmse_mm",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(fp_debug_msgs__msg__DebugFrame, track_icp_rmse_mm),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__DebugFrame_message_members = {
  "fp_debug_msgs__msg",  // message namespace
  "DebugFrame",  // message name
  18,  // number of fields
  sizeof(fp_debug_msgs__msg__DebugFrame),
  fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__DebugFrame_message_member_array,  // message members
  fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__DebugFrame_init_function,  // function to initialize message memory (memory has to be allocated)
  fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__DebugFrame_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__DebugFrame_message_type_support_handle = {
  0,
  &fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__DebugFrame_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_fp_debug_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, fp_debug_msgs, msg, DebugFrame)() {
  fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__DebugFrame_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, builtin_interfaces, msg, Time)();
  fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__DebugFrame_message_member_array[9].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, fp_debug_msgs, msg, DebugCandidate)();
  fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__DebugFrame_message_member_array[10].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, fp_debug_msgs, msg, DebugCandidate)();
  fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__DebugFrame_message_member_array[11].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, fp_debug_msgs, msg, DebugPoseItem)();
  fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__DebugFrame_message_member_array[14].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, fp_debug_msgs, msg, DebugMaskCrop)();
  if (!fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__DebugFrame_message_type_support_handle.typesupport_identifier) {
    fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__DebugFrame_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &fp_debug_msgs__msg__DebugFrame__rosidl_typesupport_introspection_c__DebugFrame_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
