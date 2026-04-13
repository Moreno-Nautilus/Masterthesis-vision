// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from fp_debug_msgs:msg/DebugMaskCrop.idl
// generated code does not contain a copyright notice

#ifndef FP_DEBUG_MSGS__MSG__DETAIL__DEBUG_MASK_CROP__BUILDER_HPP_
#define FP_DEBUG_MSGS__MSG__DETAIL__DEBUG_MASK_CROP__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "fp_debug_msgs/msg/detail/debug_mask_crop__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace fp_debug_msgs
{

namespace msg
{

namespace builder
{

class Init_DebugMaskCrop_data
{
public:
  explicit Init_DebugMaskCrop_data(::fp_debug_msgs::msg::DebugMaskCrop & msg)
  : msg_(msg)
  {}
  ::fp_debug_msgs::msg::DebugMaskCrop data(::fp_debug_msgs::msg::DebugMaskCrop::_data_type arg)
  {
    msg_.data = std::move(arg);
    return std::move(msg_);
  }

private:
  ::fp_debug_msgs::msg::DebugMaskCrop msg_;
};

class Init_DebugMaskCrop_height
{
public:
  explicit Init_DebugMaskCrop_height(::fp_debug_msgs::msg::DebugMaskCrop & msg)
  : msg_(msg)
  {}
  Init_DebugMaskCrop_data height(::fp_debug_msgs::msg::DebugMaskCrop::_height_type arg)
  {
    msg_.height = std::move(arg);
    return Init_DebugMaskCrop_data(msg_);
  }

private:
  ::fp_debug_msgs::msg::DebugMaskCrop msg_;
};

class Init_DebugMaskCrop_width
{
public:
  Init_DebugMaskCrop_width()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_DebugMaskCrop_height width(::fp_debug_msgs::msg::DebugMaskCrop::_width_type arg)
  {
    msg_.width = std::move(arg);
    return Init_DebugMaskCrop_height(msg_);
  }

private:
  ::fp_debug_msgs::msg::DebugMaskCrop msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::fp_debug_msgs::msg::DebugMaskCrop>()
{
  return fp_debug_msgs::msg::builder::Init_DebugMaskCrop_width();
}

}  // namespace fp_debug_msgs

#endif  // FP_DEBUG_MSGS__MSG__DETAIL__DEBUG_MASK_CROP__BUILDER_HPP_
