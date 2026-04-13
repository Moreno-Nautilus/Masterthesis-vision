// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from fp_debug_msgs:msg/DebugFrame.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "fp_debug_msgs/msg/detail/debug_frame__struct.hpp"
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

void DebugFrame_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) fp_debug_msgs::msg::DebugFrame(_init);
}

void DebugFrame_fini_function(void * message_memory)
{
  auto typed_message = static_cast<fp_debug_msgs::msg::DebugFrame *>(message_memory);
  typed_message->~DebugFrame();
}

size_t size_function__DebugFrame__roi_polygon_xy_flat(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<int32_t> *>(untyped_member);
  return member->size();
}

const void * get_const_function__DebugFrame__roi_polygon_xy_flat(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<int32_t> *>(untyped_member);
  return &member[index];
}

void * get_function__DebugFrame__roi_polygon_xy_flat(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<int32_t> *>(untyped_member);
  return &member[index];
}

void fetch_function__DebugFrame__roi_polygon_xy_flat(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const int32_t *>(
    get_const_function__DebugFrame__roi_polygon_xy_flat(untyped_member, index));
  auto & value = *reinterpret_cast<int32_t *>(untyped_value);
  value = item;
}

void assign_function__DebugFrame__roi_polygon_xy_flat(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<int32_t *>(
    get_function__DebugFrame__roi_polygon_xy_flat(untyped_member, index));
  const auto & value = *reinterpret_cast<const int32_t *>(untyped_value);
  item = value;
}

void resize_function__DebugFrame__roi_polygon_xy_flat(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<int32_t> *>(untyped_member);
  member->resize(size);
}

size_t size_function__DebugFrame__tiny_roi_xyxy(const void * untyped_member)
{
  (void)untyped_member;
  return 4;
}

const void * get_const_function__DebugFrame__tiny_roi_xyxy(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<int32_t, 4> *>(untyped_member);
  return &member[index];
}

void * get_function__DebugFrame__tiny_roi_xyxy(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<int32_t, 4> *>(untyped_member);
  return &member[index];
}

void fetch_function__DebugFrame__tiny_roi_xyxy(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const int32_t *>(
    get_const_function__DebugFrame__tiny_roi_xyxy(untyped_member, index));
  auto & value = *reinterpret_cast<int32_t *>(untyped_value);
  value = item;
}

void assign_function__DebugFrame__tiny_roi_xyxy(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<int32_t *>(
    get_function__DebugFrame__tiny_roi_xyxy(untyped_member, index));
  const auto & value = *reinterpret_cast<const int32_t *>(untyped_value);
  item = value;
}

size_t size_function__DebugFrame__sam_candidates(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<fp_debug_msgs::msg::DebugCandidate> *>(untyped_member);
  return member->size();
}

const void * get_const_function__DebugFrame__sam_candidates(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<fp_debug_msgs::msg::DebugCandidate> *>(untyped_member);
  return &member[index];
}

void * get_function__DebugFrame__sam_candidates(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<fp_debug_msgs::msg::DebugCandidate> *>(untyped_member);
  return &member[index];
}

void fetch_function__DebugFrame__sam_candidates(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const fp_debug_msgs::msg::DebugCandidate *>(
    get_const_function__DebugFrame__sam_candidates(untyped_member, index));
  auto & value = *reinterpret_cast<fp_debug_msgs::msg::DebugCandidate *>(untyped_value);
  value = item;
}

void assign_function__DebugFrame__sam_candidates(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<fp_debug_msgs::msg::DebugCandidate *>(
    get_function__DebugFrame__sam_candidates(untyped_member, index));
  const auto & value = *reinterpret_cast<const fp_debug_msgs::msg::DebugCandidate *>(untyped_value);
  item = value;
}

void resize_function__DebugFrame__sam_candidates(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<fp_debug_msgs::msg::DebugCandidate> *>(untyped_member);
  member->resize(size);
}

size_t size_function__DebugFrame__dino_ranked_candidates(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<fp_debug_msgs::msg::DebugCandidate> *>(untyped_member);
  return member->size();
}

const void * get_const_function__DebugFrame__dino_ranked_candidates(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<fp_debug_msgs::msg::DebugCandidate> *>(untyped_member);
  return &member[index];
}

void * get_function__DebugFrame__dino_ranked_candidates(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<fp_debug_msgs::msg::DebugCandidate> *>(untyped_member);
  return &member[index];
}

void fetch_function__DebugFrame__dino_ranked_candidates(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const fp_debug_msgs::msg::DebugCandidate *>(
    get_const_function__DebugFrame__dino_ranked_candidates(untyped_member, index));
  auto & value = *reinterpret_cast<fp_debug_msgs::msg::DebugCandidate *>(untyped_value);
  value = item;
}

void assign_function__DebugFrame__dino_ranked_candidates(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<fp_debug_msgs::msg::DebugCandidate *>(
    get_function__DebugFrame__dino_ranked_candidates(untyped_member, index));
  const auto & value = *reinterpret_cast<const fp_debug_msgs::msg::DebugCandidate *>(untyped_value);
  item = value;
}

void resize_function__DebugFrame__dino_ranked_candidates(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<fp_debug_msgs::msg::DebugCandidate> *>(untyped_member);
  member->resize(size);
}

size_t size_function__DebugFrame__pose_items(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<fp_debug_msgs::msg::DebugPoseItem> *>(untyped_member);
  return member->size();
}

const void * get_const_function__DebugFrame__pose_items(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<fp_debug_msgs::msg::DebugPoseItem> *>(untyped_member);
  return &member[index];
}

void * get_function__DebugFrame__pose_items(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<fp_debug_msgs::msg::DebugPoseItem> *>(untyped_member);
  return &member[index];
}

void fetch_function__DebugFrame__pose_items(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const fp_debug_msgs::msg::DebugPoseItem *>(
    get_const_function__DebugFrame__pose_items(untyped_member, index));
  auto & value = *reinterpret_cast<fp_debug_msgs::msg::DebugPoseItem *>(untyped_value);
  value = item;
}

void assign_function__DebugFrame__pose_items(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<fp_debug_msgs::msg::DebugPoseItem *>(
    get_function__DebugFrame__pose_items(untyped_member, index));
  const auto & value = *reinterpret_cast<const fp_debug_msgs::msg::DebugPoseItem *>(untyped_value);
  item = value;
}

void resize_function__DebugFrame__pose_items(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<fp_debug_msgs::msg::DebugPoseItem> *>(untyped_member);
  member->resize(size);
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember DebugFrame_message_member_array[12] = {
  {
    "stamp",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<builtin_interfaces::msg::Time>(),  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(fp_debug_msgs::msg::DebugFrame, stamp),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "cam_id",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(fp_debug_msgs::msg::DebugFrame, cam_id),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "max_candidate_draw",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(fp_debug_msgs::msg::DebugFrame, max_candidate_draw),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "show_axes",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(fp_debug_msgs::msg::DebugFrame, show_axes),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "roi_polygon_xy_flat",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(fp_debug_msgs::msg::DebugFrame, roi_polygon_xy_flat),  // bytes offset in struct
    nullptr,  // default value
    size_function__DebugFrame__roi_polygon_xy_flat,  // size() function pointer
    get_const_function__DebugFrame__roi_polygon_xy_flat,  // get_const(index) function pointer
    get_function__DebugFrame__roi_polygon_xy_flat,  // get(index) function pointer
    fetch_function__DebugFrame__roi_polygon_xy_flat,  // fetch(index, &value) function pointer
    assign_function__DebugFrame__roi_polygon_xy_flat,  // assign(index, value) function pointer
    resize_function__DebugFrame__roi_polygon_xy_flat  // resize(index) function pointer
  },
  {
    "has_tiny_roi",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(fp_debug_msgs::msg::DebugFrame, has_tiny_roi),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "tiny_roi_xyxy",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    4,  // array size
    false,  // is upper bound
    offsetof(fp_debug_msgs::msg::DebugFrame, tiny_roi_xyxy),  // bytes offset in struct
    nullptr,  // default value
    size_function__DebugFrame__tiny_roi_xyxy,  // size() function pointer
    get_const_function__DebugFrame__tiny_roi_xyxy,  // get_const(index) function pointer
    get_function__DebugFrame__tiny_roi_xyxy,  // get(index) function pointer
    fetch_function__DebugFrame__tiny_roi_xyxy,  // fetch(index, &value) function pointer
    assign_function__DebugFrame__tiny_roi_xyxy,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "update_sam",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(fp_debug_msgs::msg::DebugFrame, update_sam),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "update_dino",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(fp_debug_msgs::msg::DebugFrame, update_dino),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "sam_candidates",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<fp_debug_msgs::msg::DebugCandidate>(),  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(fp_debug_msgs::msg::DebugFrame, sam_candidates),  // bytes offset in struct
    nullptr,  // default value
    size_function__DebugFrame__sam_candidates,  // size() function pointer
    get_const_function__DebugFrame__sam_candidates,  // get_const(index) function pointer
    get_function__DebugFrame__sam_candidates,  // get(index) function pointer
    fetch_function__DebugFrame__sam_candidates,  // fetch(index, &value) function pointer
    assign_function__DebugFrame__sam_candidates,  // assign(index, value) function pointer
    resize_function__DebugFrame__sam_candidates  // resize(index) function pointer
  },
  {
    "dino_ranked_candidates",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<fp_debug_msgs::msg::DebugCandidate>(),  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(fp_debug_msgs::msg::DebugFrame, dino_ranked_candidates),  // bytes offset in struct
    nullptr,  // default value
    size_function__DebugFrame__dino_ranked_candidates,  // size() function pointer
    get_const_function__DebugFrame__dino_ranked_candidates,  // get_const(index) function pointer
    get_function__DebugFrame__dino_ranked_candidates,  // get(index) function pointer
    fetch_function__DebugFrame__dino_ranked_candidates,  // fetch(index, &value) function pointer
    assign_function__DebugFrame__dino_ranked_candidates,  // assign(index, value) function pointer
    resize_function__DebugFrame__dino_ranked_candidates  // resize(index) function pointer
  },
  {
    "pose_items",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<fp_debug_msgs::msg::DebugPoseItem>(),  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(fp_debug_msgs::msg::DebugFrame, pose_items),  // bytes offset in struct
    nullptr,  // default value
    size_function__DebugFrame__pose_items,  // size() function pointer
    get_const_function__DebugFrame__pose_items,  // get_const(index) function pointer
    get_function__DebugFrame__pose_items,  // get(index) function pointer
    fetch_function__DebugFrame__pose_items,  // fetch(index, &value) function pointer
    assign_function__DebugFrame__pose_items,  // assign(index, value) function pointer
    resize_function__DebugFrame__pose_items  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers DebugFrame_message_members = {
  "fp_debug_msgs::msg",  // message namespace
  "DebugFrame",  // message name
  12,  // number of fields
  sizeof(fp_debug_msgs::msg::DebugFrame),
  DebugFrame_message_member_array,  // message members
  DebugFrame_init_function,  // function to initialize message memory (memory has to be allocated)
  DebugFrame_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t DebugFrame_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &DebugFrame_message_members,
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
get_message_type_support_handle<fp_debug_msgs::msg::DebugFrame>()
{
  return &::fp_debug_msgs::msg::rosidl_typesupport_introspection_cpp::DebugFrame_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, fp_debug_msgs, msg, DebugFrame)() {
  return &::fp_debug_msgs::msg::rosidl_typesupport_introspection_cpp::DebugFrame_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
