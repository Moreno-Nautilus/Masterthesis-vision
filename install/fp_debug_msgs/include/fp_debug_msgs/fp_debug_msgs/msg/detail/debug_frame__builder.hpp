// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from fp_debug_msgs:msg/DebugFrame.idl
// generated code does not contain a copyright notice

#ifndef FP_DEBUG_MSGS__MSG__DETAIL__DEBUG_FRAME__BUILDER_HPP_
#define FP_DEBUG_MSGS__MSG__DETAIL__DEBUG_FRAME__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "fp_debug_msgs/msg/detail/debug_frame__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace fp_debug_msgs
{

namespace msg
{

namespace builder
{

class Init_DebugFrame_track_icp_rmse_mm
{
public:
  explicit Init_DebugFrame_track_icp_rmse_mm(::fp_debug_msgs::msg::DebugFrame & msg)
  : msg_(msg)
  {}
  ::fp_debug_msgs::msg::DebugFrame track_icp_rmse_mm(::fp_debug_msgs::msg::DebugFrame::_track_icp_rmse_mm_type arg)
  {
    msg_.track_icp_rmse_mm = std::move(arg);
    return std::move(msg_);
  }

private:
  ::fp_debug_msgs::msg::DebugFrame msg_;
};

class Init_DebugFrame_track_icp_fitness
{
public:
  explicit Init_DebugFrame_track_icp_fitness(::fp_debug_msgs::msg::DebugFrame & msg)
  : msg_(msg)
  {}
  Init_DebugFrame_track_icp_rmse_mm track_icp_fitness(::fp_debug_msgs::msg::DebugFrame::_track_icp_fitness_type arg)
  {
    msg_.track_icp_fitness = std::move(arg);
    return Init_DebugFrame_track_icp_rmse_mm(msg_);
  }

private:
  ::fp_debug_msgs::msg::DebugFrame msg_;
};

class Init_DebugFrame_track_object_id
{
public:
  explicit Init_DebugFrame_track_object_id(::fp_debug_msgs::msg::DebugFrame & msg)
  : msg_(msg)
  {}
  Init_DebugFrame_track_icp_fitness track_object_id(::fp_debug_msgs::msg::DebugFrame::_track_object_id_type arg)
  {
    msg_.track_object_id = std::move(arg);
    return Init_DebugFrame_track_icp_fitness(msg_);
  }

private:
  ::fp_debug_msgs::msg::DebugFrame msg_;
};

class Init_DebugFrame_track_mask
{
public:
  explicit Init_DebugFrame_track_mask(::fp_debug_msgs::msg::DebugFrame & msg)
  : msg_(msg)
  {}
  Init_DebugFrame_track_object_id track_mask(::fp_debug_msgs::msg::DebugFrame::_track_mask_type arg)
  {
    msg_.track_mask = std::move(arg);
    return Init_DebugFrame_track_object_id(msg_);
  }

private:
  ::fp_debug_msgs::msg::DebugFrame msg_;
};

class Init_DebugFrame_track_mask_bbox_xyxy
{
public:
  explicit Init_DebugFrame_track_mask_bbox_xyxy(::fp_debug_msgs::msg::DebugFrame & msg)
  : msg_(msg)
  {}
  Init_DebugFrame_track_mask track_mask_bbox_xyxy(::fp_debug_msgs::msg::DebugFrame::_track_mask_bbox_xyxy_type arg)
  {
    msg_.track_mask_bbox_xyxy = std::move(arg);
    return Init_DebugFrame_track_mask(msg_);
  }

private:
  ::fp_debug_msgs::msg::DebugFrame msg_;
};

class Init_DebugFrame_has_track_mask
{
public:
  explicit Init_DebugFrame_has_track_mask(::fp_debug_msgs::msg::DebugFrame & msg)
  : msg_(msg)
  {}
  Init_DebugFrame_track_mask_bbox_xyxy has_track_mask(::fp_debug_msgs::msg::DebugFrame::_has_track_mask_type arg)
  {
    msg_.has_track_mask = std::move(arg);
    return Init_DebugFrame_track_mask_bbox_xyxy(msg_);
  }

private:
  ::fp_debug_msgs::msg::DebugFrame msg_;
};

class Init_DebugFrame_pose_items
{
public:
  explicit Init_DebugFrame_pose_items(::fp_debug_msgs::msg::DebugFrame & msg)
  : msg_(msg)
  {}
  Init_DebugFrame_has_track_mask pose_items(::fp_debug_msgs::msg::DebugFrame::_pose_items_type arg)
  {
    msg_.pose_items = std::move(arg);
    return Init_DebugFrame_has_track_mask(msg_);
  }

private:
  ::fp_debug_msgs::msg::DebugFrame msg_;
};

class Init_DebugFrame_dino_ranked_candidates
{
public:
  explicit Init_DebugFrame_dino_ranked_candidates(::fp_debug_msgs::msg::DebugFrame & msg)
  : msg_(msg)
  {}
  Init_DebugFrame_pose_items dino_ranked_candidates(::fp_debug_msgs::msg::DebugFrame::_dino_ranked_candidates_type arg)
  {
    msg_.dino_ranked_candidates = std::move(arg);
    return Init_DebugFrame_pose_items(msg_);
  }

private:
  ::fp_debug_msgs::msg::DebugFrame msg_;
};

class Init_DebugFrame_sam_candidates
{
public:
  explicit Init_DebugFrame_sam_candidates(::fp_debug_msgs::msg::DebugFrame & msg)
  : msg_(msg)
  {}
  Init_DebugFrame_dino_ranked_candidates sam_candidates(::fp_debug_msgs::msg::DebugFrame::_sam_candidates_type arg)
  {
    msg_.sam_candidates = std::move(arg);
    return Init_DebugFrame_dino_ranked_candidates(msg_);
  }

private:
  ::fp_debug_msgs::msg::DebugFrame msg_;
};

class Init_DebugFrame_update_dino
{
public:
  explicit Init_DebugFrame_update_dino(::fp_debug_msgs::msg::DebugFrame & msg)
  : msg_(msg)
  {}
  Init_DebugFrame_sam_candidates update_dino(::fp_debug_msgs::msg::DebugFrame::_update_dino_type arg)
  {
    msg_.update_dino = std::move(arg);
    return Init_DebugFrame_sam_candidates(msg_);
  }

private:
  ::fp_debug_msgs::msg::DebugFrame msg_;
};

class Init_DebugFrame_update_sam
{
public:
  explicit Init_DebugFrame_update_sam(::fp_debug_msgs::msg::DebugFrame & msg)
  : msg_(msg)
  {}
  Init_DebugFrame_update_dino update_sam(::fp_debug_msgs::msg::DebugFrame::_update_sam_type arg)
  {
    msg_.update_sam = std::move(arg);
    return Init_DebugFrame_update_dino(msg_);
  }

private:
  ::fp_debug_msgs::msg::DebugFrame msg_;
};

class Init_DebugFrame_tiny_roi_xyxy
{
public:
  explicit Init_DebugFrame_tiny_roi_xyxy(::fp_debug_msgs::msg::DebugFrame & msg)
  : msg_(msg)
  {}
  Init_DebugFrame_update_sam tiny_roi_xyxy(::fp_debug_msgs::msg::DebugFrame::_tiny_roi_xyxy_type arg)
  {
    msg_.tiny_roi_xyxy = std::move(arg);
    return Init_DebugFrame_update_sam(msg_);
  }

private:
  ::fp_debug_msgs::msg::DebugFrame msg_;
};

class Init_DebugFrame_has_tiny_roi
{
public:
  explicit Init_DebugFrame_has_tiny_roi(::fp_debug_msgs::msg::DebugFrame & msg)
  : msg_(msg)
  {}
  Init_DebugFrame_tiny_roi_xyxy has_tiny_roi(::fp_debug_msgs::msg::DebugFrame::_has_tiny_roi_type arg)
  {
    msg_.has_tiny_roi = std::move(arg);
    return Init_DebugFrame_tiny_roi_xyxy(msg_);
  }

private:
  ::fp_debug_msgs::msg::DebugFrame msg_;
};

class Init_DebugFrame_roi_polygon_xy_flat
{
public:
  explicit Init_DebugFrame_roi_polygon_xy_flat(::fp_debug_msgs::msg::DebugFrame & msg)
  : msg_(msg)
  {}
  Init_DebugFrame_has_tiny_roi roi_polygon_xy_flat(::fp_debug_msgs::msg::DebugFrame::_roi_polygon_xy_flat_type arg)
  {
    msg_.roi_polygon_xy_flat = std::move(arg);
    return Init_DebugFrame_has_tiny_roi(msg_);
  }

private:
  ::fp_debug_msgs::msg::DebugFrame msg_;
};

class Init_DebugFrame_show_axes
{
public:
  explicit Init_DebugFrame_show_axes(::fp_debug_msgs::msg::DebugFrame & msg)
  : msg_(msg)
  {}
  Init_DebugFrame_roi_polygon_xy_flat show_axes(::fp_debug_msgs::msg::DebugFrame::_show_axes_type arg)
  {
    msg_.show_axes = std::move(arg);
    return Init_DebugFrame_roi_polygon_xy_flat(msg_);
  }

private:
  ::fp_debug_msgs::msg::DebugFrame msg_;
};

class Init_DebugFrame_max_candidate_draw
{
public:
  explicit Init_DebugFrame_max_candidate_draw(::fp_debug_msgs::msg::DebugFrame & msg)
  : msg_(msg)
  {}
  Init_DebugFrame_show_axes max_candidate_draw(::fp_debug_msgs::msg::DebugFrame::_max_candidate_draw_type arg)
  {
    msg_.max_candidate_draw = std::move(arg);
    return Init_DebugFrame_show_axes(msg_);
  }

private:
  ::fp_debug_msgs::msg::DebugFrame msg_;
};

class Init_DebugFrame_cam_id
{
public:
  explicit Init_DebugFrame_cam_id(::fp_debug_msgs::msg::DebugFrame & msg)
  : msg_(msg)
  {}
  Init_DebugFrame_max_candidate_draw cam_id(::fp_debug_msgs::msg::DebugFrame::_cam_id_type arg)
  {
    msg_.cam_id = std::move(arg);
    return Init_DebugFrame_max_candidate_draw(msg_);
  }

private:
  ::fp_debug_msgs::msg::DebugFrame msg_;
};

class Init_DebugFrame_stamp
{
public:
  Init_DebugFrame_stamp()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_DebugFrame_cam_id stamp(::fp_debug_msgs::msg::DebugFrame::_stamp_type arg)
  {
    msg_.stamp = std::move(arg);
    return Init_DebugFrame_cam_id(msg_);
  }

private:
  ::fp_debug_msgs::msg::DebugFrame msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::fp_debug_msgs::msg::DebugFrame>()
{
  return fp_debug_msgs::msg::builder::Init_DebugFrame_stamp();
}

}  // namespace fp_debug_msgs

#endif  // FP_DEBUG_MSGS__MSG__DETAIL__DEBUG_FRAME__BUILDER_HPP_
