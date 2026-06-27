cmake_minimum_required(VERSION 3.20)
project(ghost_annotation)

# --- Dependencies -------------------------------------------------------------
find_package(rosidl_default_generators REQUIRED)
find_package(rosidl_typesupport_c REQUIRED)
find_package(rosidl_typesupport_cpp REQUIRED)
find_package(rosidl_typesupport_interface REQUIRED)
find_package(rosidl_generator_py REQUIRED)

find_package(ament_cmake)
find_package(rclpy REQUIRED)
find_package(rclcpp REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(std_msgs REQUIRED)

# --- Presets ------------------------------------------------------------------
# Default to C99
if(NOT CMAKE_C_STANDARD)
    set(CMAKE_C_STANDARD 99)
endif()

# Default to C++20
if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 20)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

if(NOT MSVC)
    if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE Release)
    endif()
endif()

# --- Install ------------------------------------------------------------------
install(PROGRAMS
    "src/annotator_node.py"
    "src/grid_utils.py"
    DESTINATION lib/${PROJECT_NAME})

ament_export_dependencies(
    rclpy
    rclcpp
    builtin_interfaces )
ament_package()
