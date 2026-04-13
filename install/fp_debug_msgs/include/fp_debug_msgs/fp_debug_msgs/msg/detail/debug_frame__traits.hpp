// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from fp_debug_msgs:msg/DebugFrame.idl
// generated code does not contain a copyright notice

#ifndef FP_DEBUG_MSGS__MSG__DETAIL__DEBUG_FRAME__TRAITS_HPP_
#define FP_DEBUG_MSGS__MSG__DETAIL__DEBUG_FRAME__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "fp_debug_msgs/msg/detail/debug_frame__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'stamp'
#include "builtin_interfaces/msg/detail/time__traits.hpp"
// Member 'sam_candidates'
// Member 'dino_ranked_candidates'
#include "fp_debug_msgs/msg/detail/debug_candidate__traits.hpp"
// Member 'pose_items'
#include "fp_debug_msgs/msg/detail/debug_pose_item__traits.hpp"

namespace fp_debug_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const DebugFrame & msg,
  std::ostream & out)
{
  out << "{";
  // member: stamp
  {
    out << "stamp: ";
    to_flow_style_yaml(msg.stamp, out);
    out << ", ";
  }

  // member: cam_id
  {
    out << "cam_id: ";
    rosidl_generator_traits::value_to_yaml(msg.cam_id, out);
    out << ", ";
  }

  // member: max_candidate_draw
  {
    out << "max_candidate_draw: ";
    rosidl_generator_traits::value_to_yaml(msg.max_candidate_draw, out);
    out << ", ";
  }

  // member: show_axes
  {
    out << "show_axes: ";
    rosidl_generator_traits::value_to_yaml(msg.show_axes, out);
    out << ", ";
  }

  // member: roi_polygon_xy_flat
  {
    if (msg.roi_polygon_xy_flat.size() == 0) {
      out << "roi_polygon_xy_flat: []";
    } else {
      out << "roi_polygon_xy_flat: [";
      size_t pending_items = msg.roi_polygon_xy_flat.size();
      for (auto item : msg.roi_polygon_xy_flat) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: has_tiny_roi
  {
    out << "has_tiny_roi: ";
    rosidl_generator_traits::value_to_yaml(msg.has_tiny_roi, out);
    out << ", ";
  }

  // member: tiny_roi_xyxy
  {
    if (msg.tiny_roi_xyxy.size() == 0) {
      out << "tiny_roi_xyxy: []";
    } else {
      out << "tiny_roi_xyxy: [";
      size_t pending_items = msg.tiny_roi_xyxy.size();
      for (auto item : msg.tiny_roi_xyxy) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: update_sam
  {
    out << "update_sam: ";
    rosidl_generator_traits::value_to_yaml(msg.update_sam, out);
    out << ", ";
  }

  // member: update_dino
  {
    out << "update_dino: ";
    rosidl_generator_traits::value_to_yaml(msg.update_dino, out);
    out << ", ";
  }

  // member: sam_candidates
  {
    if (msg.sam_candidates.size() == 0) {
      out << "sam_candidates: []";
    } else {
      out << "sam_candidates: [";
      size_t pending_items = msg.sam_candidates.size();
      for (auto item : msg.sam_candidates) {
        to_flow_style_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: dino_ranked_candidates
  {
    if (msg.dino_ranked_candidates.size() == 0) {
      out << "dino_ranked_candidates: []";
    } else {
      out << "dino_ranked_candidates: [";
      size_t pending_items = msg.dino_ranked_candidates.size();
      for (auto item : msg.dino_ranked_candidates) {
        to_flow_style_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: pose_items
  {
    if (msg.pose_items.size() == 0) {
      out << "pose_items: []";
    } else {
      out << "pose_items: [";
      size_t pending_items = msg.pose_items.size();
      for (auto item : msg.pose_items) {
        to_flow_style_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const DebugFrame & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: stamp
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "stamp:\n";
    to_block_style_yaml(msg.stamp, out, indentation + 2);
  }

  // member: cam_id
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "cam_id: ";
    rosidl_generator_traits::value_to_yaml(msg.cam_id, out);
    out << "\n";
  }

  // member: max_candidate_draw
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "max_candidate_draw: ";
    rosidl_generator_traits::value_to_yaml(msg.max_candidate_draw, out);
    out << "\n";
  }

  // member: show_axes
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "show_axes: ";
    rosidl_generator_traits::value_to_yaml(msg.show_axes, out);
    out << "\n";
  }

  // member: roi_polygon_xy_flat
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.roi_polygon_xy_flat.size() == 0) {
      out << "roi_polygon_xy_flat: []\n";
    } else {
      out << "roi_polygon_xy_flat:\n";
      for (auto item : msg.roi_polygon_xy_flat) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: has_tiny_roi
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "has_tiny_roi: ";
    rosidl_generator_traits::value_to_yaml(msg.has_tiny_roi, out);
    out << "\n";
  }

  // member: tiny_roi_xyxy
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.tiny_roi_xyxy.size() == 0) {
      out << "tiny_roi_xyxy: []\n";
    } else {
      out << "tiny_roi_xyxy:\n";
      for (auto item : msg.tiny_roi_xyxy) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: update_sam
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "update_sam: ";
    rosidl_generator_traits::value_to_yaml(msg.update_sam, out);
    out << "\n";
  }

  // member: update_dino
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "update_dino: ";
    rosidl_generator_traits::value_to_yaml(msg.update_dino, out);
    out << "\n";
  }

  // member: sam_candidates
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.sam_candidates.size() == 0) {
      out << "sam_candidates: []\n";
    } else {
      out << "sam_candidates:\n";
      for (auto item : msg.sam_candidates) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }

  // member: dino_ranked_candidates
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.dino_ranked_candidates.size() == 0) {
      out << "dino_ranked_candidates: []\n";
    } else {
      out << "dino_ranked_candidates:\n";
      for (auto item : msg.dino_ranked_candidates) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }

  // member: pose_items
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.pose_items.size() == 0) {
      out << "pose_items: []\n";
    } else {
      out << "pose_items:\n";
      for (auto item : msg.pose_items) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const DebugFrame & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace msg

}  // namespace fp_debug_msgs

namespace rosidl_generator_traits
{

[[deprecated("use fp_debug_msgs::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const fp_debug_msgs::msg::DebugFrame & msg,
  std::ostream & out, size_t indentation = 0)
{
  fp_debug_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use fp_debug_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const fp_debug_msgs::msg::DebugFrame & msg)
{
  return fp_debug_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<fp_debug_msgs::msg::DebugFrame>()
{
  return "fp_debug_msgs::msg::DebugFrame";
}

template<>
inline const char * name<fp_debug_msgs::msg::DebugFrame>()
{
  return "fp_debug_msgs/msg/DebugFrame";
}

template<>
struct has_fixed_size<fp_debug_msgs::msg::DebugFrame>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<fp_debug_msgs::msg::DebugFrame>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<fp_debug_msgs::msg::DebugFrame>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // FP_DEBUG_MSGS__MSG__DETAIL__DEBUG_FRAME__TRAITS_HPP_
