// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from fp_debug_msgs:msg/DebugFrame.idl
// generated code does not contain a copyright notice

#ifndef FP_DEBUG_MSGS__MSG__DETAIL__DEBUG_FRAME__STRUCT_HPP_
#define FP_DEBUG_MSGS__MSG__DETAIL__DEBUG_FRAME__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'stamp'
#include "builtin_interfaces/msg/detail/time__struct.hpp"
// Member 'sam_candidates'
// Member 'dino_ranked_candidates'
#include "fp_debug_msgs/msg/detail/debug_candidate__struct.hpp"
// Member 'pose_items'
#include "fp_debug_msgs/msg/detail/debug_pose_item__struct.hpp"
// Member 'track_mask'
#include "fp_debug_msgs/msg/detail/debug_mask_crop__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__fp_debug_msgs__msg__DebugFrame __attribute__((deprecated))
#else
# define DEPRECATED__fp_debug_msgs__msg__DebugFrame __declspec(deprecated)
#endif

namespace fp_debug_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct DebugFrame_
{
  using Type = DebugFrame_<ContainerAllocator>;

  explicit DebugFrame_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : stamp(_init),
    track_mask(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->cam_id = "";
      this->max_candidate_draw = 0l;
      this->show_axes = false;
      this->has_tiny_roi = false;
      std::fill<typename std::array<int32_t, 4>::iterator, int32_t>(this->tiny_roi_xyxy.begin(), this->tiny_roi_xyxy.end(), 0l);
      this->update_sam = false;
      this->update_dino = false;
      this->has_track_mask = false;
      std::fill<typename std::array<int32_t, 4>::iterator, int32_t>(this->track_mask_bbox_xyxy.begin(), this->track_mask_bbox_xyxy.end(), 0l);
      this->track_object_id = "";
      this->track_icp_fitness = 0.0f;
      this->track_icp_rmse_mm = 0.0f;
    }
  }

  explicit DebugFrame_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : stamp(_alloc, _init),
    cam_id(_alloc),
    tiny_roi_xyxy(_alloc),
    track_mask_bbox_xyxy(_alloc),
    track_mask(_alloc, _init),
    track_object_id(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->cam_id = "";
      this->max_candidate_draw = 0l;
      this->show_axes = false;
      this->has_tiny_roi = false;
      std::fill<typename std::array<int32_t, 4>::iterator, int32_t>(this->tiny_roi_xyxy.begin(), this->tiny_roi_xyxy.end(), 0l);
      this->update_sam = false;
      this->update_dino = false;
      this->has_track_mask = false;
      std::fill<typename std::array<int32_t, 4>::iterator, int32_t>(this->track_mask_bbox_xyxy.begin(), this->track_mask_bbox_xyxy.end(), 0l);
      this->track_object_id = "";
      this->track_icp_fitness = 0.0f;
      this->track_icp_rmse_mm = 0.0f;
    }
  }

  // field types and members
  using _stamp_type =
    builtin_interfaces::msg::Time_<ContainerAllocator>;
  _stamp_type stamp;
  using _cam_id_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _cam_id_type cam_id;
  using _max_candidate_draw_type =
    int32_t;
  _max_candidate_draw_type max_candidate_draw;
  using _show_axes_type =
    bool;
  _show_axes_type show_axes;
  using _roi_polygon_xy_flat_type =
    std::vector<int32_t, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<int32_t>>;
  _roi_polygon_xy_flat_type roi_polygon_xy_flat;
  using _has_tiny_roi_type =
    bool;
  _has_tiny_roi_type has_tiny_roi;
  using _tiny_roi_xyxy_type =
    std::array<int32_t, 4>;
  _tiny_roi_xyxy_type tiny_roi_xyxy;
  using _update_sam_type =
    bool;
  _update_sam_type update_sam;
  using _update_dino_type =
    bool;
  _update_dino_type update_dino;
  using _sam_candidates_type =
    std::vector<fp_debug_msgs::msg::DebugCandidate_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<fp_debug_msgs::msg::DebugCandidate_<ContainerAllocator>>>;
  _sam_candidates_type sam_candidates;
  using _dino_ranked_candidates_type =
    std::vector<fp_debug_msgs::msg::DebugCandidate_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<fp_debug_msgs::msg::DebugCandidate_<ContainerAllocator>>>;
  _dino_ranked_candidates_type dino_ranked_candidates;
  using _pose_items_type =
    std::vector<fp_debug_msgs::msg::DebugPoseItem_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<fp_debug_msgs::msg::DebugPoseItem_<ContainerAllocator>>>;
  _pose_items_type pose_items;
  using _has_track_mask_type =
    bool;
  _has_track_mask_type has_track_mask;
  using _track_mask_bbox_xyxy_type =
    std::array<int32_t, 4>;
  _track_mask_bbox_xyxy_type track_mask_bbox_xyxy;
  using _track_mask_type =
    fp_debug_msgs::msg::DebugMaskCrop_<ContainerAllocator>;
  _track_mask_type track_mask;
  using _track_object_id_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _track_object_id_type track_object_id;
  using _track_icp_fitness_type =
    float;
  _track_icp_fitness_type track_icp_fitness;
  using _track_icp_rmse_mm_type =
    float;
  _track_icp_rmse_mm_type track_icp_rmse_mm;

  // setters for named parameter idiom
  Type & set__stamp(
    const builtin_interfaces::msg::Time_<ContainerAllocator> & _arg)
  {
    this->stamp = _arg;
    return *this;
  }
  Type & set__cam_id(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->cam_id = _arg;
    return *this;
  }
  Type & set__max_candidate_draw(
    const int32_t & _arg)
  {
    this->max_candidate_draw = _arg;
    return *this;
  }
  Type & set__show_axes(
    const bool & _arg)
  {
    this->show_axes = _arg;
    return *this;
  }
  Type & set__roi_polygon_xy_flat(
    const std::vector<int32_t, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<int32_t>> & _arg)
  {
    this->roi_polygon_xy_flat = _arg;
    return *this;
  }
  Type & set__has_tiny_roi(
    const bool & _arg)
  {
    this->has_tiny_roi = _arg;
    return *this;
  }
  Type & set__tiny_roi_xyxy(
    const std::array<int32_t, 4> & _arg)
  {
    this->tiny_roi_xyxy = _arg;
    return *this;
  }
  Type & set__update_sam(
    const bool & _arg)
  {
    this->update_sam = _arg;
    return *this;
  }
  Type & set__update_dino(
    const bool & _arg)
  {
    this->update_dino = _arg;
    return *this;
  }
  Type & set__sam_candidates(
    const std::vector<fp_debug_msgs::msg::DebugCandidate_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<fp_debug_msgs::msg::DebugCandidate_<ContainerAllocator>>> & _arg)
  {
    this->sam_candidates = _arg;
    return *this;
  }
  Type & set__dino_ranked_candidates(
    const std::vector<fp_debug_msgs::msg::DebugCandidate_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<fp_debug_msgs::msg::DebugCandidate_<ContainerAllocator>>> & _arg)
  {
    this->dino_ranked_candidates = _arg;
    return *this;
  }
  Type & set__pose_items(
    const std::vector<fp_debug_msgs::msg::DebugPoseItem_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<fp_debug_msgs::msg::DebugPoseItem_<ContainerAllocator>>> & _arg)
  {
    this->pose_items = _arg;
    return *this;
  }
  Type & set__has_track_mask(
    const bool & _arg)
  {
    this->has_track_mask = _arg;
    return *this;
  }
  Type & set__track_mask_bbox_xyxy(
    const std::array<int32_t, 4> & _arg)
  {
    this->track_mask_bbox_xyxy = _arg;
    return *this;
  }
  Type & set__track_mask(
    const fp_debug_msgs::msg::DebugMaskCrop_<ContainerAllocator> & _arg)
  {
    this->track_mask = _arg;
    return *this;
  }
  Type & set__track_object_id(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->track_object_id = _arg;
    return *this;
  }
  Type & set__track_icp_fitness(
    const float & _arg)
  {
    this->track_icp_fitness = _arg;
    return *this;
  }
  Type & set__track_icp_rmse_mm(
    const float & _arg)
  {
    this->track_icp_rmse_mm = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    fp_debug_msgs::msg::DebugFrame_<ContainerAllocator> *;
  using ConstRawPtr =
    const fp_debug_msgs::msg::DebugFrame_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<fp_debug_msgs::msg::DebugFrame_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<fp_debug_msgs::msg::DebugFrame_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      fp_debug_msgs::msg::DebugFrame_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<fp_debug_msgs::msg::DebugFrame_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      fp_debug_msgs::msg::DebugFrame_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<fp_debug_msgs::msg::DebugFrame_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<fp_debug_msgs::msg::DebugFrame_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<fp_debug_msgs::msg::DebugFrame_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__fp_debug_msgs__msg__DebugFrame
    std::shared_ptr<fp_debug_msgs::msg::DebugFrame_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__fp_debug_msgs__msg__DebugFrame
    std::shared_ptr<fp_debug_msgs::msg::DebugFrame_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const DebugFrame_ & other) const
  {
    if (this->stamp != other.stamp) {
      return false;
    }
    if (this->cam_id != other.cam_id) {
      return false;
    }
    if (this->max_candidate_draw != other.max_candidate_draw) {
      return false;
    }
    if (this->show_axes != other.show_axes) {
      return false;
    }
    if (this->roi_polygon_xy_flat != other.roi_polygon_xy_flat) {
      return false;
    }
    if (this->has_tiny_roi != other.has_tiny_roi) {
      return false;
    }
    if (this->tiny_roi_xyxy != other.tiny_roi_xyxy) {
      return false;
    }
    if (this->update_sam != other.update_sam) {
      return false;
    }
    if (this->update_dino != other.update_dino) {
      return false;
    }
    if (this->sam_candidates != other.sam_candidates) {
      return false;
    }
    if (this->dino_ranked_candidates != other.dino_ranked_candidates) {
      return false;
    }
    if (this->pose_items != other.pose_items) {
      return false;
    }
    if (this->has_track_mask != other.has_track_mask) {
      return false;
    }
    if (this->track_mask_bbox_xyxy != other.track_mask_bbox_xyxy) {
      return false;
    }
    if (this->track_mask != other.track_mask) {
      return false;
    }
    if (this->track_object_id != other.track_object_id) {
      return false;
    }
    if (this->track_icp_fitness != other.track_icp_fitness) {
      return false;
    }
    if (this->track_icp_rmse_mm != other.track_icp_rmse_mm) {
      return false;
    }
    return true;
  }
  bool operator!=(const DebugFrame_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct DebugFrame_

// alias to use template instance with default allocator
using DebugFrame =
  fp_debug_msgs::msg::DebugFrame_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace fp_debug_msgs

#endif  // FP_DEBUG_MSGS__MSG__DETAIL__DEBUG_FRAME__STRUCT_HPP_
