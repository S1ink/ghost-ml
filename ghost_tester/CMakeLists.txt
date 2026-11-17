cmake_minimum_required(VERSION 3.20)
project(ghost_tester VERSION 0.0.1)

# --- Dependencies -------------------------------------------------------------
find_package(ament_cmake REQUIRED)
find_package(csm_utils REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)

# --- Presets ------------------------------------------------------------------
# standard_compile_options() and auto_target_dependencies() (the patch for
# Lyrical) come with csm_utils.
standard_compile_options()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wno-psabi)
endif()

include_directories(SYSTEM)
include_directories(${PCL_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)

link_directories(${PCL_LIBRARY_DIRS})

# --- Package ------------------------------------------------------------------
add_executable(test_node
    "src/main.cpp" )

auto_target_dependencies(test_node
    csm_utils
    rclcpp
    std_msgs
    sensor_msgs
    geometry_msgs
    pcl_conversions
    tf2_ros
    tf2_sensor_msgs
    tf2_geometry_msgs )
target_link_libraries(test_node
    Eigen3::Eigen
    ${PCL_LIBRARIES} )

# Lidar artifact filter example. include/artifact_filter/artifact_filter.hpp
# is self-contained (no ROS, no PCL); OpenMP only makes it faster.
find_package(OpenMP)

add_executable(artifact_filter_node
    "src/artifact_filter_node.cpp" )
auto_target_dependencies(artifact_filter_node
    rclcpp
    sensor_msgs )

# Compares the C++ against Python on a check file (no ROS).
add_executable(artifact_filter_check
    "src/artifact_filter_check.cpp" )

if(OpenMP_CXX_FOUND)
    target_link_libraries(artifact_filter_node OpenMP::OpenMP_CXX)
    target_link_libraries(artifact_filter_check OpenMP::OpenMP_CXX)
endif()

# --- Export -------------------------------------------------------------------
install(TARGETS
    test_node
    artifact_filter_node
    artifact_filter_check
    DESTINATION lib/${PROJECT_NAME} )

install(DIRECTORY
    launch
    config
    DESTINATION share/${PROJECT_NAME} )

ament_export_dependencies(
    rclcpp
    std_msgs
    sensor_msgs
    geometry_msgs
    pcl_ros
    tf2_ros
    tf2_sensor_msgs
    tf2_geometry_msgs )

ament_package()
