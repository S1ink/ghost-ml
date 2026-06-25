#include <memory>
#include <iostream>

#include <Eigen/Core>

#include <pcl/point_cloud.h>
#include <pcl/filters/shadowpoints.h>
#include <pcl/features/normal_3d.h>

#include <pcl_conversions/pcl_conversions.h>

#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>

#include "util/pub_map.hpp"
#include "util/cloud_ops.hpp"
#include "util/ros_utils.hpp"
#include "util/std_utils.hpp"


#define NODE_TOPIC(x) "/ghost_buster" x

#define PC_SUB_TOPIC "/multiscan/lidar_scan"

#define SENSOR_QOS           \
    rclcpp::SensorDataQoS {}


class MainNode : public rclcpp::Node, util::UsingRosAliases
{
    using PointCloudMsg = sensor_msgs::msg::PointCloud2;

    using PointXYZ = pcl::PointXYZ;
    using PointCloudXYZ = pcl::PointCloud<PointXYZ>;
    using PointCloudNormal = pcl::PointCloud<pcl::Normal>;

public:
    MainNode();

public:
    void scanCallback(const PointCloudMsg::ConstSharedPtr&);

protected:
    util::GenericPubMap pub_map;

    RclSubPtr<PointCloudMsg> pc_sub;

    pcl::NormalEstimation<PointXYZ, pcl::Normal> normal_est;
    pcl::ShadowPoints<PointXYZ, pcl::Normal> shadow_point_filter;
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

    std::shared_ptr<PointCloudXYZ> raw_cloud_ptr =
        util::wrapUnmanaged(raw_cloud);
    this->normal_est.setInputCloud(raw_cloud_ptr);

    PointCloudNormal normals;
    this->normal_est.compute(normals);

    std::shared_ptr<PointCloudNormal> normals_ptr =
        util::wrapUnmanaged(normals);
    this->shadow_point_filter.setInputCloud(raw_cloud_ptr);
    this->shadow_point_filter.setNormals(normals_ptr);
    this->shadow_point_filter.setKSearch(10);

    pcl::Indices filtered_indices;
    this->shadow_point_filter.filter(filtered_indices);

    util::removeSelection(raw_cloud, filtered_indices);

    PointCloudMsg out_msg;
    pcl::toROSMsg(raw_cloud, out_msg);
    this->pub_map.publish("filtered_cloud", out_msg);
}


int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MainNode>());
    rclcpp::shutdown();

    return 0;
}
