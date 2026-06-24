cmake_minimum_required(VERSION 3.20)
project(ghost_buster VERSION 0.0.1)

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
# Default to C99
if(NOT CMAKE_C_STANDARD)
    set(CMAKE_C_STANDARD 99)
endif()

# Default to C++17
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

# --- Package ------------------------------------------------------------------
add_executable(main_node
    "src/main.cpp" )
target_link_libraries(main_node PUBLIC
    rclcpp::rclcpp
    # pcl_ros::pcl_ros
    pcl_conversions::pcl_conversions
    std_msgs::std_msgs
    sensor_msgs::sensor_msgs
    geometry_msgs::geometry_msgs
    tf2_ros::tf2_ros
    tf2_sensor_msgs::tf2_sensor_msgs
    tf2_geometry_msgs::tf2_geometry_msgs
    Eigen3::Eigen
    ${PCL_LIBRARIES} )

# --- Export -------------------------------------------------------------------
install(TARGETS
    main_node
    DESTINATION lib/${PROJECT_NAME} )

ament_package()
