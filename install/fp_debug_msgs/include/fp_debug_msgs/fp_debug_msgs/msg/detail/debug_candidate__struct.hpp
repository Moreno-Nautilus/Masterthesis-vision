// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from fp_debug_msgs:msg/DebugCandidate.idl
// generated code does not contain a copyright notice

#ifndef FP_DEBUG_MSGS__MSG__DETAIL__DEBUG_CANDIDATE__STRUCT_HPP_
#define FP_DEBUG_MSGS__MSG__DETAIL__DEBUG_CANDIDATE__STRUCT_HPP_

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

#ifndef _WIN32
# define DEPRECATED__fp_debug_msgs__msg__DebugCandidate __attribute__((deprecated))
#else
# define DEPRECATED__fp_debug_msgs__msg__DebugCandidate __declspec(deprecated)
#endif

namespace fp_debug_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct DebugCandidate_
{
  using Type = DebugCandidate_<ContainerAllocator>;

  explicit DebugCandidate_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : mask(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->object_id = "";
      this->score = 0.0f;
      std::fill<typename std::array<int32_t, 4>::iterator, int32_t>(this->bbox_xyxy.begin(), this->bbox_xyxy.end(), 0l);
      this->has_mask = false;
    }
  }

  explicit DebugCandidate_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : object_id(_alloc),
    bbox_xyxy(_alloc),
    mask(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->object_id = "";
      this->score = 0.0f;
      std::fill<typename std::array<int32_t, 4>::iterator, int32_t>(this->bbox_xyxy.begin(), this->bbox_xyxy.end(), 0l);
      this->has_mask = false;
    }
  }

  // field types and members
  using _object_id_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _object_id_type object_id;
  using _score_type =
    float;
  _score_type score;
  using _bbox_xyxy_type =
    std::array<int32_t, 4>;
  _bbox_xyxy_type bbox_xyxy;
  using _has_mask_type =
    bool;
  _has_mask_type has_mask;
  using _mask_type =
    fp_debug_msgs::msg::DebugMaskCrop_<ContainerAllocator>;
  _mask_type mask;

  // setters for named parameter idiom
  Type & set__object_id(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->object_id = _arg;
    return *this;
  }
  Type & set__score(
    const float & _arg)
  {
    this->score = _arg;
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

  // constant declarations

  // pointer types
  using RawPtr =
    fp_debug_msgs::msg::DebugCandidate_<ContainerAllocator> *;
  using ConstRawPtr =
    const fp_debug_msgs::msg::DebugCandidate_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<fp_debug_msgs::msg::DebugCandidate_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<fp_debug_msgs::msg::DebugCandidate_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      fp_debug_msgs::msg::DebugCandidate_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<fp_debug_msgs::msg::DebugCandidate_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      fp_debug_msgs::msg::DebugCandidate_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<fp_debug_msgs::msg::DebugCandidate_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<fp_debug_msgs::msg::DebugCandidate_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<fp_debug_msgs::msg::DebugCandidate_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__fp_debug_msgs__msg__DebugCandidate
    std::shared_ptr<fp_debug_msgs::msg::DebugCandidate_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__fp_debug_msgs__msg__DebugCandidate
    std::shared_ptr<fp_debug_msgs::msg::DebugCandidate_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const DebugCandidate_ & other) const
  {
    if (this->object_id != other.object_id) {
      return false;
    }
    if (this->score != other.score) {
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
    return true;
  }
  bool operator!=(const DebugCandidate_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct DebugCandidate_

// alias to use template instance with default allocator
using DebugCandidate =
  fp_debug_msgs::msg::DebugCandidate_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace fp_debug_msgs

#endif  // FP_DEBUG_MSGS__MSG__DETAIL__DEBUG_CANDIDATE__STRUCT_HPP_
