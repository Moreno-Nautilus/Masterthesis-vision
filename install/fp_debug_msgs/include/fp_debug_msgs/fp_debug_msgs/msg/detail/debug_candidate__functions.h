// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from fp_debug_msgs:msg/DebugCandidate.idl
// generated code does not contain a copyright notice

#ifndef FP_DEBUG_MSGS__MSG__DETAIL__DEBUG_CANDIDATE__FUNCTIONS_H_
#define FP_DEBUG_MSGS__MSG__DETAIL__DEBUG_CANDIDATE__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "fp_debug_msgs/msg/rosidl_generator_c__visibility_control.h"

#include "fp_debug_msgs/msg/detail/debug_candidate__struct.h"

/// Initialize msg/DebugCandidate message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * fp_debug_msgs__msg__DebugCandidate
 * )) before or use
 * fp_debug_msgs__msg__DebugCandidate__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_fp_debug_msgs
bool
fp_debug_msgs__msg__DebugCandidate__init(fp_debug_msgs__msg__DebugCandidate * msg);

/// Finalize msg/DebugCandidate message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_fp_debug_msgs
void
fp_debug_msgs__msg__DebugCandidate__fini(fp_debug_msgs__msg__DebugCandidate * msg);

/// Create msg/DebugCandidate message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * fp_debug_msgs__msg__DebugCandidate__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_fp_debug_msgs
fp_debug_msgs__msg__DebugCandidate *
fp_debug_msgs__msg__DebugCandidate__create();

/// Destroy msg/DebugCandidate message.
/**
 * It calls
 * fp_debug_msgs__msg__DebugCandidate__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_fp_debug_msgs
void
fp_debug_msgs__msg__DebugCandidate__destroy(fp_debug_msgs__msg__DebugCandidate * msg);

/// Check for msg/DebugCandidate message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_fp_debug_msgs
bool
fp_debug_msgs__msg__DebugCandidate__are_equal(const fp_debug_msgs__msg__DebugCandidate * lhs, const fp_debug_msgs__msg__DebugCandidate * rhs);

/// Copy a msg/DebugCandidate message.
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
fp_debug_msgs__msg__DebugCandidate__copy(
  const fp_debug_msgs__msg__DebugCandidate * input,
  fp_debug_msgs__msg__DebugCandidate * output);

/// Initialize array of msg/DebugCandidate messages.
/**
 * It allocates the memory for the number of elements and calls
 * fp_debug_msgs__msg__DebugCandidate__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_fp_debug_msgs
bool
fp_debug_msgs__msg__DebugCandidate__Sequence__init(fp_debug_msgs__msg__DebugCandidate__Sequence * array, size_t size);

/// Finalize array of msg/DebugCandidate messages.
/**
 * It calls
 * fp_debug_msgs__msg__DebugCandidate__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_fp_debug_msgs
void
fp_debug_msgs__msg__DebugCandidate__Sequence__fini(fp_debug_msgs__msg__DebugCandidate__Sequence * array);

/// Create array of msg/DebugCandidate messages.
/**
 * It allocates the memory for the array and calls
 * fp_debug_msgs__msg__DebugCandidate__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_fp_debug_msgs
fp_debug_msgs__msg__DebugCandidate__Sequence *
fp_debug_msgs__msg__DebugCandidate__Sequence__create(size_t size);

/// Destroy array of msg/DebugCandidate messages.
/**
 * It calls
 * fp_debug_msgs__msg__DebugCandidate__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_fp_debug_msgs
void
fp_debug_msgs__msg__DebugCandidate__Sequence__destroy(fp_debug_msgs__msg__DebugCandidate__Sequence * array);

/// Check for msg/DebugCandidate message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_fp_debug_msgs
bool
fp_debug_msgs__msg__DebugCandidate__Sequence__are_equal(const fp_debug_msgs__msg__DebugCandidate__Sequence * lhs, const fp_debug_msgs__msg__DebugCandidate__Sequence * rhs);

/// Copy an array of msg/DebugCandidate messages.
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
fp_debug_msgs__msg__DebugCandidate__Sequence__copy(
  const fp_debug_msgs__msg__DebugCandidate__Sequence * input,
  fp_debug_msgs__msg__DebugCandidate__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // FP_DEBUG_MSGS__MSG__DETAIL__DEBUG_CANDIDATE__FUNCTIONS_H_
