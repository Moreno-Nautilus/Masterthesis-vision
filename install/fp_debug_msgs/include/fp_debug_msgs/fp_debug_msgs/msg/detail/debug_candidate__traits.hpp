// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from fp_debug_msgs:msg/DebugCandidate.idl
// generated code does not contain a copyright notice

#ifndef FP_DEBUG_MSGS__MSG__DETAIL__DEBUG_CANDIDATE__TRAITS_HPP_
#define FP_DEBUG_MSGS__MSG__DETAIL__DEBUG_CANDIDATE__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "fp_debug_msgs/msg/detail/debug_candidate__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'mask'
#include "fp_debug_msgs/msg/detail/debug_mask_crop__traits.hpp"

namespace fp_debug_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const DebugCandidate & msg,
  std::ostream & out)
{
  out << "{";
  // member: object_id
  {
    out << "object_id: ";
    rosidl_generator_traits::value_to_yaml(msg.object_id, out);
    out << ", ";
  }

  // member: score
  {
    out << "score: ";
    rosidl_generator_traits::value_to_yaml(msg.score, out);
    out << ", ";
  }

  // member: bbox_xyxy
  {
    if (msg.bbox_xyxy.size() == 0) {
      out << "bbox_xyxy: []";
    } else {
      out << "bbox_xyxy: [";
      size_t pending_items = msg.bbox_xyxy.size();
      for (auto item : msg.bbox_xyxy) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: has_mask
  {
    out << "has_mask: ";
    rosidl_generator_traits::value_to_yaml(msg.has_mask, out);
    out << ", ";
  }

  // member: mask
  {
    out << "mask: ";
    to_flow_style_yaml(msg.mask, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const DebugCandidate & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: object_id
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "object_id: ";
    rosidl_generator_traits::value_to_yaml(msg.object_id, out);
    out << "\n";
  }

  // member: score
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "score: ";
    rosidl_generator_traits::value_to_yaml(msg.score, out);
    out << "\n";
  }

  // member: bbox_xyxy
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.bbox_xyxy.size() == 0) {
      out << "bbox_xyxy: []\n";
    } else {
      out << "bbox_xyxy:\n";
      for (auto item : msg.bbox_xyxy) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: has_mask
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "has_mask: ";
    rosidl_generator_traits::value_to_yaml(msg.has_mask, out);
    out << "\n";
  }

  // member: mask
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "mask:\n";
    to_block_style_yaml(msg.mask, out, indentation + 2);
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const DebugCandidate & msg, bool use_flow_style = false)
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
  const fp_debug_msgs::msg::DebugCandidate & msg,
  std::ostream & out, size_t indentation = 0)
{
  fp_debug_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use fp_debug_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const fp_debug_msgs::msg::DebugCandidate & msg)
{
  return fp_debug_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<fp_debug_msgs::msg::DebugCandidate>()
{
  return "fp_debug_msgs::msg::DebugCandidate";
}

template<>
inline const char * name<fp_debug_msgs::msg::DebugCandidate>()
{
  return "fp_debug_msgs/msg/DebugCandidate";
}

template<>
struct has_fixed_size<fp_debug_msgs::msg::DebugCandidate>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<fp_debug_msgs::msg::DebugCandidate>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<fp_debug_msgs::msg::DebugCandidate>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // FP_DEBUG_MSGS__MSG__DETAIL__DEBUG_CANDIDATE__TRAITS_HPP_
