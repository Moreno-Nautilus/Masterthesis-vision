// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from fp_debug_msgs:msg/DebugFrame.idl
// generated code does not contain a copyright notice

#ifndef FP_DEBUG_MSGS__MSG__DETAIL__DEBUG_FRAME__FUNCTIONS_H_
#define FP_DEBUG_MSGS__MSG__DETAIL__DEBUG_FRAME__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "fp_debug_msgs/msg/rosidl_generator_c__visibility_control.h"

#include "fp_debug_msgs/msg/detail/debug_frame__struct.h"

/// Initialize msg/DebugFrame message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * fp_debug_msgs__msg__DebugFrame
 * )) before or use
 * fp_debug_msgs__msg__DebugFrame__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_fp_debug_msgs
bool
fp_debug_msgs__msg__DebugFrame__init(fp_debug_msgs__msg__DebugFrame * msg);

/// Finalize msg/DebugFrame message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_fp_debug_msgs
void
fp_debug_msgs__msg__DebugFrame__fini(fp_debug_msgs__msg__DebugFrame * msg);

/// Create msg/DebugFrame message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * fp_debug_msgs__msg__DebugFrame__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_fp_debug_msgs
fp_debug_msgs__msg__DebugFrame *
fp_debug_msgs__msg__DebugFrame__create();

/// Destroy msg/DebugFrame message.
/**
 * It calls
 * fp_debug_msgs__msg__DebugFrame__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_fp_debug_msgs
void
fp_debug_msgs__msg__DebugFrame__destroy(fp_debug_msgs__msg__DebugFrame * msg);

/// Check for msg/DebugFrame message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_fp_debug_msgs
bool
fp_debug_msgs__msg__DebugFrame__are_equal(const fp_debug_msgs__msg__DebugFrame * lhs, const fp_debug_msgs__msg__DebugFrame * rhs);

/// Copy a msg/DebugFrame message.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source message pointer.
 * \param[out] output The target message pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer is null
 *   or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_fp_debug_msgs
bool
fp_debug_msgs__msg__DebugFrame__copy(
  const fp_debug_msgs__msg__DebugFrame * input,
  fp_debug_msgs__msg__DebugFrame * output);

/// Initialize array of msg/DebugFrame messages.
/**
 * It allocates the memory for the number of elements and calls
 * fp_debug_msgs__msg__DebugFrame__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_fp_debug_msgs
bool
fp_debug_msgs__msg__DebugFrame__Sequence__init(fp_debug_msgs__msg__DebugFrame__Sequence * array, size_t size);

/// Finalize array of msg/DebugFrame messages.
/**
 * It calls
 * fp_debug_msgs__msg__DebugFrame__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_fp_debug_msgs
void
fp_debug_msgs__msg__DebugFrame__Sequence__fini(fp_debug_msgs__msg__DebugFrame__Sequence * array);

/// Create array of msg/DebugFrame messages.
/**
 * It allocates the memory for the array and calls
 * fp_debug_msgs__msg__DebugFrame__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_fp_debug_msgs
fp_debug_msgs__msg__DebugFrame__Sequence *
fp_debug_msgs__msg__DebugFrame__Sequence__create(size_t size);

/// Destroy array of msg/DebugFrame messages.
/**
 * It calls
 * fp_debug_msgs__msg__DebugFrame__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_fp_debug_msgs
void
fp_debug_msgs__msg__DebugFrame__Sequence__destroy(fp_debug_msgs__msg__DebugFrame__Sequence * array);

/// Check for msg/DebugFrame message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_fp_debug_msgs
bool
fp_debug_msgs__msg__DebugFrame__Sequence__are_equal(const fp_debug_msgs__msg__DebugFrame__Sequence * lhs, const fp_debug_msgs__msg__DebugFrame__Sequence * rhs);

/// Copy an array of msg/DebugFrame messages.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source array pointer.
 * \param[out] output The target array pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer
 *   is null or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_fp_debug_msgs
bool
fp_debug_msgs__msg__DebugFrame__Sequence__copy(
  const fp_debug_msgs__msg__DebugFrame__Sequence * input,
  fp_debug_msgs__msg__DebugFrame__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // FP_DEBUG_MSGS__MSG__DETAIL__DEBUG_FRAME__FUNCTIONS_H_
