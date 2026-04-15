// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from fp_debug_msgs:msg/DebugFrame.idl
// generated code does not contain a copyright notice
#include "fp_debug_msgs/msg/detail/debug_frame__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `stamp`
#include "builtin_interfaces/msg/detail/time__functions.h"
// Member `cam_id`
// Member `track_object_id`
#include "rosidl_runtime_c/string_functions.h"
// Member `roi_polygon_xy_flat`
#include "rosidl_runtime_c/primitives_sequence_functions.h"
// Member `sam_candidates`
// Member `dino_ranked_candidates`
#include "fp_debug_msgs/msg/detail/debug_candidate__functions.h"
// Member `pose_items`
#include "fp_debug_msgs/msg/detail/debug_pose_item__functions.h"
// Member `track_mask`
#include "fp_debug_msgs/msg/detail/debug_mask_crop__functions.h"

bool
fp_debug_msgs__msg__DebugFrame__init(fp_debug_msgs__msg__DebugFrame * msg)
{
  if (!msg) {
    return false;
  }
  // stamp
  if (!builtin_interfaces__msg__Time__init(&msg->stamp)) {
    fp_debug_msgs__msg__DebugFrame__fini(msg);
    return false;
  }
  // cam_id
  if (!rosidl_runtime_c__String__init(&msg->cam_id)) {
    fp_debug_msgs__msg__DebugFrame__fini(msg);
    return false;
  }
  // max_candidate_draw
  // show_axes
  // roi_polygon_xy_flat
  if (!rosidl_runtime_c__int32__Sequence__init(&msg->roi_polygon_xy_flat, 0)) {
    fp_debug_msgs__msg__DebugFrame__fini(msg);
    return false;
  }
  // has_tiny_roi
  // tiny_roi_xyxy
  // update_sam
  // update_dino
  // sam_candidates
  if (!fp_debug_msgs__msg__DebugCandidate__Sequence__init(&msg->sam_candidates, 0)) {
    fp_debug_msgs__msg__DebugFrame__fini(msg);
    return false;
  }
  // dino_ranked_candidates
  if (!fp_debug_msgs__msg__DebugCandidate__Sequence__init(&msg->dino_ranked_candidates, 0)) {
    fp_debug_msgs__msg__DebugFrame__fini(msg);
    return false;
  }
  // pose_items
  if (!fp_debug_msgs__msg__DebugPoseItem__Sequence__init(&msg->pose_items, 0)) {
    fp_debug_msgs__msg__DebugFrame__fini(msg);
    return false;
  }
  // has_track_mask
  // track_mask_bbox_xyxy
  // track_mask
  if (!fp_debug_msgs__msg__DebugMaskCrop__init(&msg->track_mask)) {
    fp_debug_msgs__msg__DebugFrame__fini(msg);
    return false;
  }
  // track_object_id
  if (!rosidl_runtime_c__String__init(&msg->track_object_id)) {
    fp_debug_msgs__msg__DebugFrame__fini(msg);
    return false;
  }
  // track_icp_fitness
  // track_icp_rmse_mm
  return true;
}

void
fp_debug_msgs__msg__DebugFrame__fini(fp_debug_msgs__msg__DebugFrame * msg)
{
  if (!msg) {
    return;
  }
  // stamp
  builtin_interfaces__msg__Time__fini(&msg->stamp);
  // cam_id
  rosidl_runtime_c__String__fini(&msg->cam_id);
  // max_candidate_draw
  // show_axes
  // roi_polygon_xy_flat
  rosidl_runtime_c__int32__Sequence__fini(&msg->roi_polygon_xy_flat);
  // has_tiny_roi
  // tiny_roi_xyxy
  // update_sam
  // update_dino
  // sam_candidates
  fp_debug_msgs__msg__DebugCandidate__Sequence__fini(&msg->sam_candidates);
  // dino_ranked_candidates
  fp_debug_msgs__msg__DebugCandidate__Sequence__fini(&msg->dino_ranked_candidates);
  // pose_items
  fp_debug_msgs__msg__DebugPoseItem__Sequence__fini(&msg->pose_items);
  // has_track_mask
  // track_mask_bbox_xyxy
  // track_mask
  fp_debug_msgs__msg__DebugMaskCrop__fini(&msg->track_mask);
  // track_object_id
  rosidl_runtime_c__String__fini(&msg->track_object_id);
  // track_icp_fitness
  // track_icp_rmse_mm
}

bool
fp_debug_msgs__msg__DebugFrame__are_equal(const fp_debug_msgs__msg__DebugFrame * lhs, const fp_debug_msgs__msg__DebugFrame * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // stamp
  if (!builtin_interfaces__msg__Time__are_equal(
      &(lhs->stamp), &(rhs->stamp)))
  {
    return false;
  }
  // cam_id
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->cam_id), &(rhs->cam_id)))
  {
    return false;
  }
  // max_candidate_draw
  if (lhs->max_candidate_draw != rhs->max_candidate_draw) {
    return false;
  }
  // show_axes
  if (lhs->show_axes != rhs->show_axes) {
    return false;
  }
  // roi_polygon_xy_flat
  if (!rosidl_runtime_c__int32__Sequence__are_equal(
      &(lhs->roi_polygon_xy_flat), &(rhs->roi_polygon_xy_flat)))
  {
    return false;
  }
  // has_tiny_roi
  if (lhs->has_tiny_roi != rhs->has_tiny_roi) {
    return false;
  }
  // tiny_roi_xyxy
  for (size_t i = 0; i < 4; ++i) {
    if (lhs->tiny_roi_xyxy[i] != rhs->tiny_roi_xyxy[i]) {
      return false;
    }
  }
  // update_sam
  if (lhs->update_sam != rhs->update_sam) {
    return false;
  }
  // update_dino
  if (lhs->update_dino != rhs->update_dino) {
    return false;
  }
  // sam_candidates
  if (!fp_debug_msgs__msg__DebugCandidate__Sequence__are_equal(
      &(lhs->sam_candidates), &(rhs->sam_candidates)))
  {
    return false;
  }
  // dino_ranked_candidates
  if (!fp_debug_msgs__msg__DebugCandidate__Sequence__are_equal(
      &(lhs->dino_ranked_candidates), &(rhs->dino_ranked_candidates)))
  {
    return false;
  }
  // pose_items
  if (!fp_debug_msgs__msg__DebugPoseItem__Sequence__are_equal(
      &(lhs->pose_items), &(rhs->pose_items)))
  {
    return false;
  }
  // has_track_mask
  if (lhs->has_track_mask != rhs->has_track_mask) {
    return false;
  }
  // track_mask_bbox_xyxy
  for (size_t i = 0; i < 4; ++i) {
    if (lhs->track_mask_bbox_xyxy[i] != rhs->track_mask_bbox_xyxy[i]) {
      return false;
    }
  }
  // track_mask
  if (!fp_debug_msgs__msg__DebugMaskCrop__are_equal(
      &(lhs->track_mask), &(rhs->track_mask)))
  {
    return false;
  }
  // track_object_id
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->track_object_id), &(rhs->track_object_id)))
  {
    return false;
  }
  // track_icp_fitness
  if (lhs->track_icp_fitness != rhs->track_icp_fitness) {
    return false;
  }
  // track_icp_rmse_mm
  if (lhs->track_icp_rmse_mm != rhs->track_icp_rmse_mm) {
    return false;
  }
  return true;
}

bool
fp_debug_msgs__msg__DebugFrame__copy(
  const fp_debug_msgs__msg__DebugFrame * input,
  fp_debug_msgs__msg__DebugFrame * output)
{
  if (!input || !output) {
    return false;
  }
  // stamp
  if (!builtin_interfaces__msg__Time__copy(
      &(input->stamp), &(output->stamp)))
  {
    return false;
  }
  // cam_id
  if (!rosidl_runtime_c__String__copy(
      &(input->cam_id), &(output->cam_id)))
  {
    return false;
  }
  // max_candidate_draw
  output->max_candidate_draw = input->max_candidate_draw;
  // show_axes
  output->show_axes = input->show_axes;
  // roi_polygon_xy_flat
  if (!rosidl_runtime_c__int32__Sequence__copy(
      &(input->roi_polygon_xy_flat), &(output->roi_polygon_xy_flat)))
  {
    return false;
  }
  // has_tiny_roi
  output->has_tiny_roi = input->has_tiny_roi;
  // tiny_roi_xyxy
  for (size_t i = 0; i < 4; ++i) {
    output->tiny_roi_xyxy[i] = input->tiny_roi_xyxy[i];
  }
  // update_sam
  output->update_sam = input->update_sam;
  // update_dino
  output->update_dino = input->update_dino;
  // sam_candidates
  if (!fp_debug_msgs__msg__DebugCandidate__Sequence__copy(
      &(input->sam_candidates), &(output->sam_candidates)))
  {
    return false;
  }
  // dino_ranked_candidates
  if (!fp_debug_msgs__msg__DebugCandidate__Sequence__copy(
      &(input->dino_ranked_candidates), &(output->dino_ranked_candidates)))
  {
    return false;
  }
  // pose_items
  if (!fp_debug_msgs__msg__DebugPoseItem__Sequence__copy(
      &(input->pose_items), &(output->pose_items)))
  {
    return false;
  }
  // has_track_mask
  output->has_track_mask = input->has_track_mask;
  // track_mask_bbox_xyxy
  for (size_t i = 0; i < 4; ++i) {
    output->track_mask_bbox_xyxy[i] = input->track_mask_bbox_xyxy[i];
  }
  // track_mask
  if (!fp_debug_msgs__msg__DebugMaskCrop__copy(
      &(input->track_mask), &(output->track_mask)))
  {
    return false;
  }
  // track_object_id
  if (!rosidl_runtime_c__String__copy(
      &(input->track_object_id), &(output->track_object_id)))
  {
    return false;
  }
  // track_icp_fitness
  output->track_icp_fitness = input->track_icp_fitness;
  // track_icp_rmse_mm
  output->track_icp_rmse_mm = input->track_icp_rmse_mm;
  return true;
}

fp_debug_msgs__msg__DebugFrame *
fp_debug_msgs__msg__DebugFrame__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  fp_debug_msgs__msg__DebugFrame * msg = (fp_debug_msgs__msg__DebugFrame *)allocator.allocate(sizeof(fp_debug_msgs__msg__DebugFrame), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(fp_debug_msgs__msg__DebugFrame));
  bool success = fp_debug_msgs__msg__DebugFrame__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
fp_debug_msgs__msg__DebugFrame__destroy(fp_debug_msgs__msg__DebugFrame * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    fp_debug_msgs__msg__DebugFrame__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
fp_debug_msgs__msg__DebugFrame__Sequence__init(fp_debug_msgs__msg__DebugFrame__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  fp_debug_msgs__msg__DebugFrame * data = NULL;

  if (size) {
    data = (fp_debug_msgs__msg__DebugFrame *)allocator.zero_allocate(size, sizeof(fp_debug_msgs__msg__DebugFrame), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = fp_debug_msgs__msg__DebugFrame__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        fp_debug_msgs__msg__DebugFrame__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
fp_debug_msgs__msg__DebugFrame__Sequence__fini(fp_debug_msgs__msg__DebugFrame__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      fp_debug_msgs__msg__DebugFrame__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

fp_debug_msgs__msg__DebugFrame__Sequence *
fp_debug_msgs__msg__DebugFrame__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  fp_debug_msgs__msg__DebugFrame__Sequence * array = (fp_debug_msgs__msg__DebugFrame__Sequence *)allocator.allocate(sizeof(fp_debug_msgs__msg__DebugFrame__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = fp_debug_msgs__msg__DebugFrame__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
fp_debug_msgs__msg__DebugFrame__Sequence__destroy(fp_debug_msgs__msg__DebugFrame__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    fp_debug_msgs__msg__DebugFrame__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
fp_debug_msgs__msg__DebugFrame__Sequence__are_equal(const fp_debug_msgs__msg__DebugFrame__Sequence * lhs, const fp_debug_msgs__msg__DebugFrame__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!fp_debug_msgs__msg__DebugFrame__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
fp_debug_msgs__msg__DebugFrame__Sequence__copy(
  const fp_debug_msgs__msg__DebugFrame__Sequence * input,
  fp_debug_msgs__msg__DebugFrame__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(fp_debug_msgs__msg__DebugFrame);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    fp_debug_msgs__msg__DebugFrame * data =
      (fp_debug_msgs__msg__DebugFrame *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!fp_debug_msgs__msg__DebugFrame__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          fp_debug_msgs__msg__DebugFrame__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!fp_debug_msgs__msg__DebugFrame__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
