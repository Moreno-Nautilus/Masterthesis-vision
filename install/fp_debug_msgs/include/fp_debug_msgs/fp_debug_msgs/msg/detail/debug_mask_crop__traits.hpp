// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from fp_debug_msgs:msg/DebugMaskCrop.idl
// generated code does not contain a copyright notice

#ifndef FP_DEBUG_MSGS__MSG__DETAIL__DEBUG_MASK_CROP__TRAITS_HPP_
#define FP_DEBUG_MSGS__MSG__DETAIL__DEBUG_MASK_CROP__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "fp_debug_msgs/msg/detail/debug_mask_crop__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace fp_debug_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const DebugMaskCrop & msg,
  std::ostream & out)
{
  out << "{";
  // member: width
  {
    out << "width: ";
    rosidl_generator_traits::value_to_yaml(msg.width, out);
    out << ", ";
  }

  // member: height
  {
    out << "height: ";
    rosidl_generator_traits::value_to_yaml(msg.height, out);
    out << ", ";
  }

  // member: data
  {
    if (msg.data.size() == 0) {
      out << "data: []";
    } else {
      out << "data: [";
      size_t pending_items = msg.data.size();
      for (auto item : msg.data) {
        rosidl_generator_traits::value_to_yaml(item, out);
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
  const DebugMaskCrop & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: width
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "width: ";
    rosidl_generator_traits::value_to_yaml(msg.width, out);
    out << "\n";
  }

  // member: height
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "height: ";
    rosidl_generator_traits::value_to_yaml(msg.height, out);
    out << "\n";
  }

  // member: data
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.data.size() == 0) {
      out << "data: []\n";
    } else {
      out << "data:\n";
      for (auto item : msg.data) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const DebugMaskCrop & msg, bool use_flow_style = false)
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
  const fp_debug_msgs::msg::DebugMaskCrop & msg,
  std::ostream & out, size_t indentation = 0)
{
  fp_debug_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use fp_debug_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const fp_debug_msgs::msg::DebugMaskCrop & msg)
{
  return fp_debug_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<fp_debug_msgs::msg::DebugMaskCrop>()
{
  return "fp_debug_msgs::msg::DebugMaskCrop";
}

template<>
inline const char * name<fp_debug_msgs::msg::DebugMaskCrop>()
{
  return "fp_debug_msgs/msg/DebugMaskCrop";
}

template<>
struct has_fixed_size<fp_debug_msgs::msg::DebugMaskCrop>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<fp_debug_msgs::msg::DebugMaskCrop>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<fp_debug_msgs::msg::DebugMaskCrop>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // FP_DEBUG_MSGS__MSG__DETAIL__DEBUG_MASK_CROP__TRAITS_HPP_
