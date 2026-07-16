cmake_minimum_required(VERSION 3.8)
project(fp_debug_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(action_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/DebugMaskCrop.msg"
  "msg/DebugCandidate.msg"
  "msg/DebugPoseItem.msg"
  "msg/DebugFrame.msg"
  # assembly-cell interfaces (state machine <-> grasp node / insertion node),
  # hosted here per team agreement: all shared definitions live in this package
  "msg/AssemblyProgress.msg"
  "action/GraspAssembly.action"
  "action/InsertPart.action"
  DEPENDENCIES builtin_interfaces geometry_msgs action_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
