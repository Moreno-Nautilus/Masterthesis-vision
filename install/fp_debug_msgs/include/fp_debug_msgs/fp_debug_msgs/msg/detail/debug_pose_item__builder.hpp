// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from fp_debug_msgs:msg/DebugPoseItem.idl
// generated code does not contain a copyright notice

#ifndef FP_DEBUG_MSGS__MSG__DETAIL__DEBUG_POSE_ITEM__BUILDER_HPP_
#define FP_DEBUG_MSGS__MSG__DETAIL__DEBUG_POSE_ITEM__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "fp_debug_msgs/msg/detail/debug_pose_item__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace fp_debug_msgs
{

namespace msg
{

namespace builder
{

class Init_DebugPoseItem_axis_len_m
{
public:
  explicit Init_DebugPoseItem_axis_len_m(::fp_debug_msgs::msg::DebugPoseItem & msg)
  : msg_(msg)
  {}
  ::fp_debug_msgs::msg::DebugPoseItem axis_len_m(::fp_debug_msgs::msg::DebugPoseItem::_axis_len_m_type arg)
  {
    msg_.axis_len_m = std::move(arg);
    return std::move(msg_);
  }

private:
  ::fp_debug_msgs::msg::DebugPoseItem msg_;
};

class Init_DebugPoseItem_pose_base
{
public:
  explicit Init_DebugPoseItem_pose_base(::fp_debug_msgs::msg::DebugPoseItem & msg)
  : msg_(msg)
  {}
  Init_DebugPoseItem_axis_len_m pose_base(::fp_debug_msgs::msg::DebugPoseItem::_pose_base_type arg)
  {
    msg_.pose_base = std::move(arg);
    return Init_DebugPoseItem_axis_len_m(msg_);
  }

private:
  ::fp_debug_msgs::msg::DebugPoseItem msg_;
};

class Init_DebugPoseItem_pose_camera
{
public:
  explicit Init_DebugPoseItem_pose_camera(::fp_debug_msgs::msg::DebugPoseItem & msg)
  : msg_(msg)
  {}
  Init_DebugPoseItem_pose_base pose_camera(::fp_debug_msgs::msg::DebugPoseItem::_pose_camera_type arg)
  {
    msg_.pose_camera = std::move(arg);
    return Init_DebugPoseItem_pose_base(msg_);
  }

private:
  ::fp_debug_msgs::msg::DebugPoseItem msg_;
};

class Init_DebugPoseItem_mask
{
public:
  explicit Init_DebugPoseItem_mask(::fp_debug_msgs::msg::DebugPoseItem & msg)
  : msg_(msg)
  {}
  Init_DebugPoseItem_pose_camera mask(::fp_debug_msgs::msg::DebugPoseItem::_mask_type arg)
  {
    msg_.mask = std::move(arg);
    return Init_DebugPoseItem_pose_camera(msg_);
  }

private:
  ::fp_debug_msgs::msg::DebugPoseItem msg_;
};

class Init_DebugPoseItem_has_mask
{
public:
  explicit Init_DebugPoseItem_has_mask(::fp_debug_msgs::msg::DebugPoseItem & msg)
  : msg_(msg)
  {}
  Init_DebugPoseItem_mask has_mask(::fp_debug_msgs::msg::DebugPoseItem::_has_mask_type arg)
  {
    msg_.has_mask = std::move(arg);
    return Init_DebugPoseItem_mask(msg_);
  }

private:
  ::fp_debug_msgs::msg::DebugPoseItem msg_;
};

class Init_DebugPoseItem_bbox_xyxy
{
public:
  explicit Init_DebugPoseItem_bbox_xyxy(::fp_debug_msgs::msg::DebugPoseItem & msg)
  : msg_(msg)
  {}
  Init_DebugPoseItem_has_mask bbox_xyxy(::fp_debug_msgs::msg::DebugPoseItem::_bbox_xyxy_type arg)
  {
    msg_.bbox_xyxy = std::move(arg);
    return Init_DebugPoseItem_has_mask(msg_);
  }

private:
  ::fp_debug_msgs::msg::DebugPoseItem msg_;
};

class Init_DebugPoseItem_has_bbox
{
public:
  explicit Init_DebugPoseItem_has_bbox(::fp_debug_msgs::msg::DebugPoseItem & msg)
  : msg_(msg)
  {}
  Init_DebugPoseItem_bbox_xyxy has_bbox(::fp_debug_msgs::msg::DebugPoseItem::_has_bbox_type arg)
  {
    msg_.has_bbox = std::move(arg);
    return Init_DebugPoseItem_bbox_xyxy(msg_);
  }

private:
  ::fp_debug_msgs::msg::DebugPoseItem msg_;
};

class Init_DebugPoseItem_score
{
public:
  explicit Init_DebugPoseItem_score(::fp_debug_msgs::msg::DebugPoseItem & msg)
  : msg_(msg)
  {}
  Init_DebugPoseItem_has_bbox score(::fp_debug_msgs::msg::DebugPoseItem::_score_type arg)
  {
    msg_.score = std::move(arg);
    return Init_DebugPoseItem_has_bbox(msg_);
  }

private:
  ::fp_debug_msgs::msg::DebugPoseItem msg_;
};

class Init_DebugPoseItem_mode
{
public:
  explicit Init_DebugPoseItem_mode(::fp_debug_msgs::msg::DebugPoseItem & msg)
  : msg_(msg)
  {}
  Init_DebugPoseItem_score mode(::fp_debug_msgs::msg::DebugPoseItem::_mode_type arg)
  {
    msg_.mode = std::move(arg);
    return Init_DebugPoseItem_score(msg_);
  }

private:
  ::fp_debug_msgs::msg::DebugPoseItem msg_;
};

class Init_DebugPoseItem_object_id
{
public:
  Init_DebugPoseItem_object_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_DebugPoseItem_mode object_id(::fp_debug_msgs::msg::DebugPoseItem::_object_id_type arg)
  {
    msg_.object_id = std::move(arg);
    return Init_DebugPoseItem_mode(msg_);
  }

private:
  ::fp_debug_msgs::msg::DebugPoseItem msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::fp_debug_msgs::msg::DebugPoseItem>()
{
  return fp_debug_msgs::msg::builder::Init_DebugPoseItem_object_id();
}

}  // namespace fp_debug_msgs

#endif  // FP_DEBUG_MSGS__MSG__DETAIL__DEBUG_POSE_ITEM__BUILDER_HPP_
