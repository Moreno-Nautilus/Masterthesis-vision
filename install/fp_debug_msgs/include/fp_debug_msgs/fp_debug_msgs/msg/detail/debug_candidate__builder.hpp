// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from fp_debug_msgs:msg/DebugCandidate.idl
// generated code does not contain a copyright notice

#ifndef FP_DEBUG_MSGS__MSG__DETAIL__DEBUG_CANDIDATE__BUILDER_HPP_
#define FP_DEBUG_MSGS__MSG__DETAIL__DEBUG_CANDIDATE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "fp_debug_msgs/msg/detail/debug_candidate__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace fp_debug_msgs
{

namespace msg
{

namespace builder
{

class Init_DebugCandidate_mask
{
public:
  explicit Init_DebugCandidate_mask(::fp_debug_msgs::msg::DebugCandidate & msg)
  : msg_(msg)
  {}
  ::fp_debug_msgs::msg::DebugCandidate mask(::fp_debug_msgs::msg::DebugCandidate::_mask_type arg)
  {
    msg_.mask = std::move(arg);
    return std::move(msg_);
  }

private:
  ::fp_debug_msgs::msg::DebugCandidate msg_;
};

class Init_DebugCandidate_has_mask
{
public:
  explicit Init_DebugCandidate_has_mask(::fp_debug_msgs::msg::DebugCandidate & msg)
  : msg_(msg)
  {}
  Init_DebugCandidate_mask has_mask(::fp_debug_msgs::msg::DebugCandidate::_has_mask_type arg)
  {
    msg_.has_mask = std::move(arg);
    return Init_DebugCandidate_mask(msg_);
  }

private:
  ::fp_debug_msgs::msg::DebugCandidate msg_;
};

class Init_DebugCandidate_bbox_xyxy
{
public:
  explicit Init_DebugCandidate_bbox_xyxy(::fp_debug_msgs::msg::DebugCandidate & msg)
  : msg_(msg)
  {}
  Init_DebugCandidate_has_mask bbox_xyxy(::fp_debug_msgs::msg::DebugCandidate::_bbox_xyxy_type arg)
  {
    msg_.bbox_xyxy = std::move(arg);
    return Init_DebugCandidate_has_mask(msg_);
  }

private:
  ::fp_debug_msgs::msg::DebugCandidate msg_;
};

class Init_DebugCandidate_score
{
public:
  explicit Init_DebugCandidate_score(::fp_debug_msgs::msg::DebugCandidate & msg)
  : msg_(msg)
  {}
  Init_DebugCandidate_bbox_xyxy score(::fp_debug_msgs::msg::DebugCandidate::_score_type arg)
  {
    msg_.score = std::move(arg);
    return Init_DebugCandidate_bbox_xyxy(msg_);
  }

private:
  ::fp_debug_msgs::msg::DebugCandidate msg_;
};

class Init_DebugCandidate_object_id
{
public:
  Init_DebugCandidate_object_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_DebugCandidate_score object_id(::fp_debug_msgs::msg::DebugCandidate::_object_id_type arg)
  {
    msg_.object_id = std::move(arg);
    return Init_DebugCandidate_score(msg_);
  }

private:
  ::fp_debug_msgs::msg::DebugCandidate msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::fp_debug_msgs::msg::DebugCandidate>()
{
  return fp_debug_msgs::msg::builder::Init_DebugCandidate_object_id();
}

}  // namespace fp_debug_msgs

#endif  // FP_DEBUG_MSGS__MSG__DETAIL__DEBUG_CANDIDATE__BUILDER_HPP_
