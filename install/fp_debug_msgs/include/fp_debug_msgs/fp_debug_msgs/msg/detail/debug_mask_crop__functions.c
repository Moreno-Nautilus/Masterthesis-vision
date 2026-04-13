// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from fp_debug_msgs:msg/DebugMaskCrop.idl
// generated code does not contain a copyright notice
#include "fp_debug_msgs/msg/detail/debug_mask_crop__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `data`
#include "rosidl_runtime_c/primitives_sequence_functions.h"

bool
fp_debug_msgs__msg__DebugMaskCrop__init(fp_debug_msgs__msg__DebugMaskCrop * msg)
{
  if (!msg) {
    return false;
  }
  // width
  // height
  // data
  if (!rosidl_runtime_c__uint8__Sequence__init(&msg->data, 0)) {
    fp_debug_msgs__msg__DebugMaskCrop__fini(msg);
    return false;
  }
  return true;
}

void
fp_debug_msgs__msg__DebugMaskCrop__fini(fp_debug_msgs__msg__DebugMaskCrop * msg)
{
  if (!msg) {
    return;
  }
  // width
  // height
  // data
  rosidl_runtime_c__uint8__Sequence__fini(&msg->data);
}

bool
fp_debug_msgs__msg__DebugMaskCrop__are_equal(const fp_debug_msgs__msg__DebugMaskCrop * lhs, const fp_debug_msgs__msg__DebugMaskCrop * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // width
  if (lhs->width != rhs->width) {
    return false;
  }
  // height
  if (lhs->height != rhs->height) {
    return false;
  }
  // data
  if (!rosidl_runtime_c__uint8__Sequence__are_equal(
      &(lhs->data), &(rhs->data)))
  {
    return false;
  }
  return true;
}

bool
fp_debug_msgs__msg__DebugMaskCrop__copy(
  const fp_debug_msgs__msg__DebugMaskCrop * input,
  fp_debug_msgs__msg__DebugMaskCrop * output)
{
  if (!input || !output) {
    return false;
  }
  // width
  output->width = input->width;
  // height
  output->height = input->height;
  // data
  if (!rosidl_runtime_c__uint8__Sequence__copy(
      &(input->data), &(output->data)))
  {
    return false;
  }
  return true;
}

fp_debug_msgs__msg__DebugMaskCrop *
fp_debug_msgs__msg__DebugMaskCrop__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  fp_debug_msgs__msg__DebugMaskCrop * msg = (fp_debug_msgs__msg__DebugMaskCrop *)allocator.allocate(sizeof(fp_debug_msgs__msg__DebugMaskCrop), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(fp_debug_msgs__msg__DebugMaskCrop));
  bool success = fp_debug_msgs__msg__DebugMaskCrop__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
fp_debug_msgs__msg__DebugMaskCrop__destroy(fp_debug_msgs__msg__DebugMaskCrop * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    fp_debug_msgs__msg__DebugMaskCrop__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
fp_debug_msgs__msg__DebugMaskCrop__Sequence__init(fp_debug_msgs__msg__DebugMaskCrop__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  fp_debug_msgs__msg__DebugMaskCrop * data = NULL;

  if (size) {
    data = (fp_debug_msgs__msg__DebugMaskCrop *)allocator.zero_allocate(size, sizeof(fp_debug_msgs__msg__DebugMaskCrop), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = fp_debug_msgs__msg__DebugMaskCrop__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        fp_debug_msgs__msg__DebugMaskCrop__fini(&data[i - 1]);
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
fp_debug_msgs__msg__DebugMaskCrop__Sequence__fini(fp_debug_msgs__msg__DebugMaskCrop__Sequence * array)
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
      fp_debug_msgs__msg__DebugMaskCrop__fini(&array->data[i]);
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

fp_debug_msgs__msg__DebugMaskCrop__Sequence *
fp_debug_msgs__msg__DebugMaskCrop__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  fp_debug_msgs__msg__DebugMaskCrop__Sequence * array = (fp_debug_msgs__msg__DebugMaskCrop__Sequence *)allocator.allocate(sizeof(fp_debug_msgs__msg__DebugMaskCrop__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = fp_debug_msgs__msg__DebugMaskCrop__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
fp_debug_msgs__msg__DebugMaskCrop__Sequence__destroy(fp_debug_msgs__msg__DebugMaskCrop__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    fp_debug_msgs__msg__DebugMaskCrop__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
fp_debug_msgs__msg__DebugMaskCrop__Sequence__are_equal(const fp_debug_msgs__msg__DebugMaskCrop__Sequence * lhs, const fp_debug_msgs__msg__DebugMaskCrop__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!fp_debug_msgs__msg__DebugMaskCrop__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
fp_debug_msgs__msg__DebugMaskCrop__Sequence__copy(
  const fp_debug_msgs__msg__DebugMaskCrop__Sequence * input,
  fp_debug_msgs__msg__DebugMaskCrop__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(fp_debug_msgs__msg__DebugMaskCrop);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    fp_debug_msgs__msg__DebugMaskCrop * data =
      (fp_debug_msgs__msg__DebugMaskCrop *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!fp_debug_msgs__msg__DebugMaskCrop__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          fp_debug_msgs__msg__DebugMaskCrop__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!fp_debug_msgs__msg__DebugMaskCrop__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
