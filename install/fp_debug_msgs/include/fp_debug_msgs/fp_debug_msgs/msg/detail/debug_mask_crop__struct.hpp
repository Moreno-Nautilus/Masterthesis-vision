// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from fp_debug_msgs:msg/DebugMaskCrop.idl
// generated code does not contain a copyright notice

#ifndef FP_DEBUG_MSGS__MSG__DETAIL__DEBUG_MASK_CROP__STRUCT_HPP_
#define FP_DEBUG_MSGS__MSG__DETAIL__DEBUG_MASK_CROP__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__fp_debug_msgs__msg__DebugMaskCrop __attribute__((deprecated))
#else
# define DEPRECATED__fp_debug_msgs__msg__DebugMaskCrop __declspec(deprecated)
#endif

namespace fp_debug_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct DebugMaskCrop_
{
  using Type = DebugMaskCrop_<ContainerAllocator>;

  explicit DebugMaskCrop_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->width = 0ul;
      this->height = 0ul;
    }
  }

  explicit DebugMaskCrop_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->width = 0ul;
      this->height = 0ul;
    }
  }

  // field types and members
  using _width_type =
    uint32_t;
  _width_type width;
  using _height_type =
    uint32_t;
  _height_type height;
  using _data_type =
    std::vector<uint8_t, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<uint8_t>>;
  _data_type data;

  // setters for named parameter idiom
  Type & set__width(
    const uint32_t & _arg)
  {
    this->width = _arg;
    return *this;
  }
  Type & set__height(
    const uint32_t & _arg)
  {
    this->height = _arg;
    return *this;
  }
  Type & set__data(
    const std::vector<uint8_t, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<uint8_t>> & _arg)
  {
    this->data = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    fp_debug_msgs::msg::DebugMaskCrop_<ContainerAllocator> *;
  using ConstRawPtr =
    const fp_debug_msgs::msg::DebugMaskCrop_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<fp_debug_msgs::msg::DebugMaskCrop_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<fp_debug_msgs::msg::DebugMaskCrop_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      fp_debug_msgs::msg::DebugMaskCrop_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<fp_debug_msgs::msg::DebugMaskCrop_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      fp_debug_msgs::msg::DebugMaskCrop_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<fp_debug_msgs::msg::DebugMaskCrop_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<fp_debug_msgs::msg::DebugMaskCrop_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<fp_debug_msgs::msg::DebugMaskCrop_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__fp_debug_msgs__msg__DebugMaskCrop
    std::shared_ptr<fp_debug_msgs::msg::DebugMaskCrop_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__fp_debug_msgs__msg__DebugMaskCrop
    std::shared_ptr<fp_debug_msgs::msg::DebugMaskCrop_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const DebugMaskCrop_ & other) const
  {
    if (this->width != other.width) {
      return false;
    }
    if (this->height != other.height) {
      return false;
    }
    if (this->data != other.data) {
      return false;
    }
    return true;
  }
  bool operator!=(const DebugMaskCrop_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct DebugMaskCrop_

// alias to use template instance with default allocator
using DebugMaskCrop =
  fp_debug_msgs::msg::DebugMaskCrop_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace fp_debug_msgs

#endif  // FP_DEBUG_MSGS__MSG__DETAIL__DEBUG_MASK_CROP__STRUCT_HPP_
