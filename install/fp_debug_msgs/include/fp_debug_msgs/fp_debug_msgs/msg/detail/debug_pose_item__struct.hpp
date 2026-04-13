// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from fp_debug_msgs:msg/DebugPoseItem.idl
// generated code does not contain a copyright notice

#ifndef FP_DEBUG_MSGS__MSG__DETAIL__DEBUG_POSE_ITEM__STRUCT_HPP_
#define FP_DEBUG_MSGS__MSG__DETAIL__DEBUG_POSE_ITEM__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'mask'
#include "fp_debug_msgs/msg/detail/debug_mask_crop__struct.hpp"
// Member 'pose_camera'
// Member 'pose_base'
#include "geometry_msgs/msg/detail/pose__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__fp_debug_msgs__msg__DebugPoseItem __attribute__((deprecated))
#else
# define DEPRECATED__fp_debug_msgs__msg__DebugPoseItem __declspec(deprecated)
#endif

namespace fp_debug_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct DebugPoseItem_
{
  using Type = DebugPoseItem_<ContainerAllocator>;

  explicit DebugPoseItem_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : mask(_init),
    pose_camera(_init),
    pose_base(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->object_id = "";
      this->mode = "";
      this->score = 0.0f;
      this->has_bbox = false;
      std::fill<typename std::array<int32_t, 4>::iterator, int32_t>(this->bbox_xyxy.begin(), this->bbox_xyxy.end(), 0l);
      this->has_mask = false;
      this->axis_len_m = 0.0f;
    }
  }

  explicit DebugPoseItem_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : object_id(_alloc),
    mode(_alloc),
    bbox_xyxy(_alloc),
    mask(_alloc, _init),
    pose_camera(_alloc, _init),
    pose_base(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->object_id = "";
      this->mode = "";
      this->score = 0.0f;
      this->has_bbox = false;
      std::fill<typename std::array<int32_t, 4>::iterator, int32_t>(this->bbox_xyxy.begin(), this->bbox_xyxy.end(), 0l);
      this->has_mask = false;
      this->axis_len_m = 0.0f;
    }
  }

  // field types and members
  using _object_id_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _object_id_type object_id;
  using _mode_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _mode_type mode;
  using _score_type =
    float;
  _score_type score;
  using _has_bbox_type =
    bool;
  _has_bbox_type has_bbox;
  using _bbox_xyxy_type =
    std::array<int32_t, 4>;
  _bbox_xyxy_type bbox_xyxy;
  using _has_mask_type =
    bool;
  _has_mask_type has_mask;
  using _mask_type =
    fp_debug_msgs::msg::DebugMaskCrop_<ContainerAllocator>;
  _mask_type mask;
  using _pose_camera_type =
    geometry_msgs::msg::Pose_<ContainerAllocator>;
  _pose_camera_type pose_camera;
  using _pose_base_type =
    geometry_msgs::msg::Pose_<ContainerAllocator>;
  _pose_base_type pose_base;
  using _axis_len_m_type =
    float;
  _axis_len_m_type axis_len_m;

  // setters for named parameter idiom
  Type & set__object_id(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->object_id = _arg;
    return *this;
  }
  Type & set__mode(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->mode = _arg;
    return *this;
  }
  Type & set__score(
    const float & _arg)
  {
    this->score = _arg;
    return *this;
  }
  Type & set__has_bbox(
    const bool & _arg)
  {
    this->has_bbox = _arg;
    return *this;
  }
  Type & set__bbox_xyxy(
    const std::array<int32_t, 4> & _arg)
  {
    this->bbox_xyxy = _arg;
    return *this;
  }
  Type & set__has_mask(
    const bool & _arg)
  {
    this->has_mask = _arg;
    return *this;
  }
  Type & set__mask(
    const fp_debug_msgs::msg::DebugMaskCrop_<ContainerAllocator> & _arg)
  {
    this->mask = _arg;
    return *this;
  }
  Type & set__pose_camera(
    const geometry_msgs::msg::Pose_<ContainerAllocator> & _arg)
  {
    this->pose_camera = _arg;
    return *this;
  }
  Type & set__pose_base(
    const geometry_msgs::msg::Pose_<ContainerAllocator> & _arg)
  {
    this->pose_base = _arg;
    return *this;
  }
  Type & set__axis_len_m(
    const float & _arg)
  {
    this->axis_len_m = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    fp_debug_msgs::msg::DebugPoseItem_<ContainerAllocator> *;
  using ConstRawPtr =
    const fp_debug_msgs::msg::DebugPoseItem_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<fp_debug_msgs::msg::DebugPoseItem_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<fp_debug_msgs::msg::DebugPoseItem_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      fp_debug_msgs::msg::DebugPoseItem_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<fp_debug_msgs::msg::DebugPoseItem_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      fp_debug_msgs::msg::DebugPoseItem_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<fp_debug_msgs::msg::DebugPoseItem_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<fp_debug_msgs::msg::DebugPoseItem_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<fp_debug_msgs::msg::DebugPoseItem_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__fp_debug_msgs__msg__DebugPoseItem
    std::shared_ptr<fp_debug_msgs::msg::DebugPoseItem_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__fp_debug_msgs__msg__DebugPoseItem
    std::shared_ptr<fp_debug_msgs::msg::DebugPoseItem_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const DebugPoseItem_ & other) const
  {
    if (this->object_id != other.object_id) {
      return false;
    }
    if (this->mode != other.mode) {
      return false;
    }
    if (this->score != other.score) {
      return false;
    }
    if (this->has_bbox != other.has_bbox) {
      return false;
    }
    if (this->bbox_xyxy != other.bbox_xyxy) {
      return false;
    }
    if (this->has_mask != other.has_mask) {
      return false;
    }
    if (this->mask != other.mask) {
      return false;
    }
    if (this->pose_camera != other.pose_camera) {
      return false;
    }
    if (this->pose_base != other.pose_base) {
      return false;
    }
    if (this->axis_len_m != other.axis_len_m) {
      return false;
    }
    return true;
  }
  bool operator!=(const DebugPoseItem_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct DebugPoseItem_

// alias to use template instance with default allocator
using DebugPoseItem =
  fp_debug_msgs::msg::DebugPoseItem_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace fp_debug_msgs

#endif  // FP_DEBUG_MSGS__MSG__DETAIL__DEBUG_POSE_ITEM__STRUCT_HPP_
