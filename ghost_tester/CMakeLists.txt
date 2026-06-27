cmake_minimum_required(VERSION 3.20)
project(ghost_tester VERSION 0.0.1)

# --- Dependencies -------------------------------------------------------------
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)

# --- Presets ------------------------------------------------------------------
if(NOT CMAKE_C_STANDARD)
    set(CMAKE_C_STANDARD 99)
endif()

if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 20)
endif()

if(NOT MSVC)
    if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE Release)
    endif()
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -Wno-psabi)
endif()

include_directories(SYSTEM)
include_directories(${PCL_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)

link_directories(${PCL_LIBRARY_DIRS})

# --- Distro detection -----------------------------------------------------
# Jazzy's (and lower) pcl_ros/tf2 packages don't export namespaced CMake
# targets for every dependency, so fall back to ament_target_dependencies there.
set(NEEDS_AMENT_TARGET_DEPS TRUE)
if(DEFINED ENV{ROS_DISTRO} AND "$ENV{ROS_DISTRO}" STREQUAL "lyrical")
    set(NEEDS_AMENT_TARGET_DEPS FALSE)
endif()

# --- Package ------------------------------------------------------------------
add_executable(test_node
    "src/main.cpp" )

if(NEEDS_AMENT_TARGET_DEPS)
    ament_target_dependencies(test_node
        rclcpp
        std_msgs
        sensor_msgs
        geometry_msgs
        pcl_ros
        tf2_ros
        tf2_sensor_msgs
        tf2_geometry_msgs )
    target_link_libraries(test_node
        Eigen3::Eigen
        ${PCL_LIBRARIES} )
else()
    target_link_libraries(test_node PUBLIC
        rclcpp::rclcpp
        pcl_conversions::pcl_conversions
        std_msgs::std_msgs
        sensor_msgs::sensor_msgs
        geometry_msgs::geometry_msgs
        tf2_ros::tf2_ros
        tf2_sensor_msgs::tf2_sensor_msgs
        tf2_geometry_msgs::tf2_geometry_msgs
        Eigen3::Eigen
        ${PCL_LIBRARIES} )
endif()

# --- Export -------------------------------------------------------------------
install(TARGETS
    test_node
    DESTINATION lib/${PROJECT_NAME} )

if(NEEDS_AMENT_TARGET_DEPS)
    ament_export_dependencies(
        rclcpp
        std_msgs
        sensor_msgs
        geometry_msgs
        pcl_ros
        tf2_ros
        tf2_sensor_msgs
        tf2_geometry_msgs )
endif()

ament_package()
