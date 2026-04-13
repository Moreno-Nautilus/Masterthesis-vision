// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from fp_debug_msgs:msg/DebugCandidate.idl
// generated code does not contain a copyright notice
#include "fp_debug_msgs/msg/detail/debug_candidate__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `object_id`
#include "rosidl_runtime_c/string_functions.h"
// Member `mask`
#include "fp_debug_msgs/msg/detail/debug_mask_crop__functions.h"

bool
fp_debug_msgs__msg__DebugCandidate__init(fp_debug_msgs__msg__DebugCandidate * msg)
{
  if (!msg) {
    return false;
  }
  // object_id
  if (!rosidl_runtime_c__String__init(&msg->object_id)) {
    fp_debug_msgs__msg__DebugCandidate__fini(msg);
    return false;
  }
  // score
  // bbox_xyxy
  // has_mask
  // mask
  if (!fp_debug_msgs__msg__DebugMaskCrop__init(&msg->mask)) {
    fp_debug_msgs__msg__DebugCandidate__fini(msg);
    return false;
  }
  return true;
}

void
fp_debug_msgs__msg__DebugCandidate__fini(fp_debug_msgs__msg__DebugCandidate * msg)
{
  if (!msg) {
    return;
  }
  // object_id
  rosidl_runtime_c__String__fini(&msg->object_id);
  // score
  // bbox_xyxy
  // has_mask
  // mask
  fp_debug_msgs__msg__DebugMaskCrop__fini(&msg->mask);
}

bool
fp_debug_msgs__msg__DebugCandidate__are_equal(const fp_debug_msgs__msg__DebugCandidate * lhs, const fp_debug_msgs__msg__DebugCandidate * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // object_id
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->object_id), &(rhs->object_id)))
  {
    return false;
  }
  // score
  if (lhs->score != rhs->score) {
    return false;
  }
  // bbox_xyxy
  for (size_t i = 0; i < 4; ++i) {
    if (lhs->bbox_xyxy[i] != rhs->bbox_xyxy[i]) {
      return false;
    }
  }
  // has_mask
  if (lhs->has_mask != rhs->has_mask) {
    return false;
  }
  // mask
  if (!fp_debug_msgs__msg__DebugMaskCrop__are_equal(
      &(lhs->mask), &(rhs->mask)))
  {
    return false;
  }
  return true;
}

bool
fp_debug_msgs__msg__DebugCandidate__copy(
  const fp_debug_msgs__msg__DebugCandidate * input,
  fp_debug_msgs__msg__DebugCandidate * output)
{
  if (!input || !output) {
    return false;
  }
  // object_id
  if (!rosidl_runtime_c__String__copy(
      &(input->object_id), &(output->object_id)))
  {
    return false;
  }
  // score
  output->score = input->score;
  // bbox_xyxy
  for (size_t i = 0; i < 4; ++i) {
    output->bbox_xyxy[i] = input->bbox_xyxy[i];
  }
  // has_mask
  output->has_mask = input->has_mask;
  // mask
  if (!fp_debug_msgs__msg__DebugMaskCrop__copy(
      &(input->mask), &(output->mask)))
  {
    return false;
  }
  return true;
}

fp_debug_msgs__msg__DebugCandidate *
fp_debug_msgs__msg__DebugCandidate__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  fp_debug_msgs__msg__DebugCandidate * msg = (fp_debug_msgs__msg__DebugCandidate *)allocator.allocate(sizeof(fp_debug_msgs__msg__DebugCandidate), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(fp_debug_msgs__msg__DebugCandidate));
  bool success = fp_debug_msgs__msg__DebugCandidate__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
fp_debug_msgs__msg__DebugCandidate__destroy(fp_debug_msgs__msg__DebugCandidate * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    fp_debug_msgs__msg__DebugCandidate__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
fp_debug_msgs__msg__DebugCandidate__Sequence__init(fp_debug_msgs__msg__DebugCandidate__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  fp_debug_msgs__msg__DebugCandidate * data = NULL;

  if (size) {
    data = (fp_debug_msgs__msg__DebugCandidate *)allocator.zero_allocate(size, sizeof(fp_debug_msgs__msg__DebugCandidate), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = fp_debug_msgs__msg__DebugCandidate__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        fp_debug_msgs__msg__DebugCandidate__fini(&data[i - 1]);
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
fp_debug_msgs__msg__DebugCandidate__Sequence__fini(fp_debug_msgs__msg__DebugCandidate__Sequence * array)
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
      fp_debug_msgs__msg__DebugCandidate__fini(&array->data[i]);
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

fp_debug_msgs__msg__DebugCandidate__Sequence *
fp_debug_msgs__msg__DebugCandidate__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  fp_debug_msgs__msg__DebugCandidate__Sequence * array = (fp_debug_msgs__msg__DebugCandidate__Sequence *)allocator.allocate(sizeof(fp_debug_msgs__msg__DebugCandidate__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = fp_debug_msgs__msg__DebugCandidate__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
fp_debug_msgs__msg__DebugCandidate__Sequence__destroy(fp_debug_msgs__msg__DebugCandidate__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    fp_debug_msgs__msg__DebugCandidate__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
fp_debug_msgs__msg__DebugCandidate__Sequence__are_equal(const fp_debug_msgs__msg__DebugCandidate__Sequence * lhs, const fp_debug_msgs__msg__DebugCandidate__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!fp_debug_msgs__msg__DebugCandidate__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
fp_debug_msgs__msg__DebugCandidate__Sequence__copy(
  const fp_debug_msgs__msg__DebugCandidate__Sequence * input,
  fp_debug_msgs__msg__DebugCandidate__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(fp_debug_msgs__msg__DebugCandidate);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    fp_debug_msgs__msg__DebugCandidate * data =
      (fp_debug_msgs__msg__DebugCandidate *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!fp_debug_msgs__msg__DebugCandidate__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          fp_debug_msgs__msg__DebugCandidate__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!fp_debug_msgs__msg__DebugCandidate__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
