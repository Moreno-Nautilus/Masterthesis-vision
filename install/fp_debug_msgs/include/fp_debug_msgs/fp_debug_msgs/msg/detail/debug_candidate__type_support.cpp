// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from fp_debug_msgs:msg/DebugCandidate.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "fp_debug_msgs/msg/detail/debug_candidate__struct.hpp"
#include "rosidl_typesupport_introspection_cpp/field_types.hpp"
#include "rosidl_typesupport_introspection_cpp/identifier.hpp"
#include "rosidl_typesupport_introspection_cpp/message_introspection.hpp"
#include "rosidl_typesupport_introspection_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_introspection_cpp/visibility_control.h"

namespace fp_debug_msgs
{

namespace msg
{

namespace rosidl_typesupport_introspection_cpp
{

void DebugCandidate_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) fp_debug_msgs::msg::DebugCandidate(_init);
}

void DebugCandidate_fini_function(void * message_memory)
{
  auto typed_message = static_cast<fp_debug_msgs::msg::DebugCandidate *>(message_memory);
  typed_message->~DebugCandidate();
}

size_t size_function__DebugCandidate__bbox_xyxy(const void * untyped_member)
{
  (void)untyped_member;
  return 4;
}

const void * get_const_function__DebugCandidate__bbox_xyxy(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<int32_t, 4> *>(untyped_member);
  return &member[index];
}

void * get_function__DebugCandidate__bbox_xyxy(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<int32_t, 4> *>(untyped_member);
  return &member[index];
}

void fetch_function__DebugCandidate__bbox_xyxy(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const int32_t *>(
    get_const_function__DebugCandidate__bbox_xyxy(untyped_member, index));
  auto & value = *reinterpret_cast<int32_t *>(untyped_value);
  value = item;
}

void assign_function__DebugCandidate__bbox_xyxy(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<int32_t *>(
    get_function__DebugCandidate__bbox_xyxy(untyped_member, index));
  const auto & value = *reinterpret_cast<const int32_t *>(untyped_value);
  item = value;
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember DebugCandidate_message_member_array[5] = {
  {
    "object_id",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(fp_debug_msgs::msg::DebugCandidate, object_id),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "score",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(fp_debug_msgs::msg::DebugCandidate, score),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "bbox_xyxy",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    4,  // array size
    false,  // is upper bound
    offsetof(fp_debug_msgs::msg::DebugCandidate, bbox_xyxy),  // bytes offset in struct
    nullptr,  // default value
    size_function__DebugCandidate__bbox_xyxy,  // size() function pointer
    get_const_function__DebugCandidate__bbox_xyxy,  // get_const(index) function pointer
    get_function__DebugCandidate__bbox_xyxy,  // get(index) function pointer
    fetch_function__DebugCandidate__bbox_xyxy,  // fetch(index, &value) function pointer
    assign_function__DebugCandidate__bbox_xyxy,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "has_mask",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(fp_debug_msgs::msg::DebugCandidate, has_mask),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "mask",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<fp_debug_msgs::msg::DebugMaskCrop>(),  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(fp_debug_msgs::msg::DebugCandidate, mask),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers DebugCandidate_message_members = {
  "fp_debug_msgs::msg",  // message namespace
  "DebugCandidate",  // message name
  5,  // number of fields
  sizeof(fp_debug_msgs::msg::DebugCandidate),
  DebugCandidate_message_member_array,  // message members
  DebugCandidate_init_function,  // function to initialize message memory (memory has to be allocated)
  DebugCandidate_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t DebugCandidate_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &DebugCandidate_message_members,
  get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace msg

}  // namespace fp_debug_msgs


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<fp_debug_msgs::msg::DebugCandidate>()
{
  return &::fp_debug_msgs::msg::rosidl_typesupport_introspection_cpp::DebugCandidate_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, fp_debug_msgs, msg, DebugCandidate)() {
  return &::fp_debug_msgs::msg::rosidl_typesupport_introspection_cpp::DebugCandidate_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
