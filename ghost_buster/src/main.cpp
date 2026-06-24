#include <memory>
#include <iostream>

#include <Eigen/Core>

#include <pcl/point_cloud.h>
#include <pcl/filters/shadowpoints.h>

#include <pcl_conversions/pcl_conversions.h>

#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>

#include "util/pub_map.hpp"
#include "util/ros_utils.hpp"


#define NODE_TOPIC(x) "/ghost_buster" x

#define PC_SUB_TOPIC "/multiscan/lidar_scan"

#define SENSOR_QOS           \
    rclcpp::SensorDataQoS {}


class MainNode : public rclcpp::Node, util::UsingRosAliases
{
    using PointCloudMsg = sensor_msgs::msg::PointCloud2;

    using PointXYZ = pcl::PointXYZ;
    using PointCloudXYZ = pcl::PointCloud<PointXYZ>;

public:
    MainNode();

public:
    void scanCallback(const PointCloudMsg::ConstSharedPtr&);

protected:
    util::GenericPubMap pub_map;

    RclSubPtr<PointCloudMsg> pc_sub;
};


MainNode::MainNode() :
    Node("ghost_buster_node"),
    pub_map{*this, NODE_TOPIC(), SENSOR_QOS},
    pc_sub{this->create_subscription<PointCloudMsg>(
        PC_SUB_TOPIC,
        SENSOR_QOS,
        [this](const PointCloudMsg::ConstSharedPtr& msg)
        { this->scanCallback(msg); })}
{
}

void MainNode::scanCallback(const PointCloudMsg::ConstSharedPtr& msg)
{
    PointCloudXYZ raw_cloud;
    pcl::fromROSMsg(*msg, raw_cloud);

    
}


int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MainNode>());
    rclcpp::shutdown();

    return 0;
}
