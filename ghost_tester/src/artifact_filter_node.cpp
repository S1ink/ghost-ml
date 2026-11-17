/*
 * artifact_filter_node
 *
 * Example of running the lidar artifact classifier on a live point cloud:
 * every incoming scan is classified with a model exported from annotator_app
 * and republished with the artifact points removed.
 *
 * Parameters
 *   model_file          (required) artifact_<variant>.model.txt from the
 *                       project's models folder. It must be a realtime model.
 *   input_topic         PointCloud2 from the lidar, in the lidar's own frame
 *                       (default /multiscan/lidar_scan)
 *   threshold           flag points with probability >= this; a negative
 *                       value (default) uses the one stored in the model
 *   early_exit          stop evaluating the trees for points that are clearly
 *                       clean (default true). Several times faster and flags
 *                       practically the same points, but artifact_prob of
 *                       points under the threshold is then approximate. Set
 *                       false for exact probabilities.
 *   keep_near           keep points nearer than the model's minimum range,
 *                       which are never classified (default false)
 *   publish_artifacts   also publish the removed points (default true)
 *   publish_classified  also publish every point with an extra float32 field
 *                       `artifact_prob`, -1 where not classified (default false)
 *
 * Topics (under the node's name)
 *   ~/filtered     the scan without artifact points, same fields as the input
 *   ~/artifacts    only the artifact points
 *   ~/classified   every point plus artifact_prob
 *
 *   ros2 run ghost_tester artifact_filter_node --ros-args \
 *       -p model_file:=/path/to/models/artifact_xyz.model.txt
 */
#include <chrono>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>

#include "artifact_filter/artifact_filter.hpp"


class ArtifactFilterNode : public rclcpp::Node
{
    using PointCloudMsg = sensor_msgs::msg::PointCloud2;
    using PointField = sensor_msgs::msg::PointField;

public:
    ArtifactFilterNode() : Node("artifact_filter")
    {
        const std::string model_file = this->declare_parameter<std::string>("model_file", "");
        const std::string input_topic =
            this->declare_parameter<std::string>("input_topic", "/multiscan/lidar_scan");
        this->threshold = this->declare_parameter<double>("threshold", -1.);
        this->keep_near = this->declare_parameter<bool>("keep_near", false);
        const bool early_exit = this->declare_parameter<bool>("early_exit", true);
        const bool pub_artifacts = this->declare_parameter<bool>("publish_artifacts", true);
        const bool pub_classified = this->declare_parameter<bool>("publish_classified", false);

        if (model_file.empty())
        {
            throw std::runtime_error(
                "parameter 'model_file' is not set: give the path of an exported "
                "model, e.g. -p model_file:=<project>/models/artifact_xyz.model.txt");
        }
        this->filter = std::make_unique<artifact_filter::Filter>(
            artifact_filter::Model::load(model_file));
        const artifact_filter::Model& m = this->filter->model();
        if (this->threshold < 0.) { this->threshold = m.threshold; }
        this->filter->early_exit(early_exit);
        if (early_exit && m.exit_after.empty())
        {
            RCLCPP_WARN(
                this->get_logger(),
                "the model file has no early exit checkpoints (trained before they "
                "were added): every tree is evaluated for every point. Retrain to get them.");
        }
        else if (early_exit && this->threshold < 0.05)
        {
            RCLCPP_WARN(
                this->get_logger(),
                "early exit is calibrated for thresholds of 0.05 and up; with %.3f "
                "some points may be missed. Set early_exit:=false.", this->threshold);
        }

        RCLCPP_INFO(
            this->get_logger(),
            "model %s (%s, trained %s): %zu trees, %d features, grid %d x %d, "
            "%s, threshold %.2f. Listening on %s",
            model_file.c_str(), m.variant.c_str(), m.trained.c_str(),
            m.tree_root.size(), m.n_features, m.rings, m.cols,
            m.use_intensity ? ("uses intensity field '" + m.intensity_field + "'").c_str()
                            : "range only",
            this->threshold, input_topic.c_str());

        const rclcpp::SensorDataQoS qos;
        this->filtered_pub = this->create_publisher<PointCloudMsg>("~/filtered", qos);
        if (pub_artifacts)
        {
            this->artifacts_pub = this->create_publisher<PointCloudMsg>("~/artifacts", qos);
        }
        if (pub_classified)
        {
            this->classified_pub = this->create_publisher<PointCloudMsg>("~/classified", qos);
        }
        this->sub = this->create_subscription<PointCloudMsg>(
            input_topic, qos,
            [this](const PointCloudMsg::ConstSharedPtr& msg) { this->scanCallback(*msg); });
    }

private:
    static const PointField* findField(const PointCloudMsg& msg, const std::string& name)
    {
        for (const PointField& f : msg.fields)
        {
            if (f.name == name) { return &f; }
        }
        return nullptr;
    }

    /** One value of any PointField type as a float */
    static float readAs(const uint8_t* p, uint8_t datatype)
    {
        switch (datatype)
        {
            case PointField::INT8: { int8_t v; std::memcpy(&v, p, 1); return v; }
            case PointField::UINT8: { uint8_t v; std::memcpy(&v, p, 1); return v; }
            case PointField::INT16: { int16_t v; std::memcpy(&v, p, 2); return v; }
            case PointField::UINT16: { uint16_t v; std::memcpy(&v, p, 2); return v; }
            case PointField::INT32: { int32_t v; std::memcpy(&v, p, 4); return static_cast<float>(v); }
            case PointField::UINT32: { uint32_t v; std::memcpy(&v, p, 4); return static_cast<float>(v); }
            case PointField::FLOAT32: { float v; std::memcpy(&v, p, 4); return v; }
            case PointField::FLOAT64: { double v; std::memcpy(&v, p, 8); return static_cast<float>(v); }
            default: return 0.f;
        }
    }

    /** A cloud holding the selected points of `src`, bytes copied as they are */
    static void copySelected(
        const PointCloudMsg& src, const std::vector<uint8_t>& take, PointCloudMsg& out)
    {
        out.header = src.header;
        out.fields = src.fields;
        out.is_bigendian = src.is_bigendian;
        out.point_step = src.point_step;
        out.is_dense = src.is_dense;
        out.height = 1;
        out.data.clear();
        out.data.reserve(src.data.size());
        uint32_t n = 0;
        for (size_t i = 0; i < take.size(); i++)
        {
            if (!take[i]) { continue; }
            const uint8_t* p = src.data.data() + i * src.point_step;
            out.data.insert(out.data.end(), p, p + src.point_step);
            n++;
        }
        out.width = n;
        out.row_step = n * src.point_step;
    }

    void scanCallback(const PointCloudMsg& msg)
    {
        const auto t0 = std::chrono::steady_clock::now();
        const artifact_filter::Model& m = this->filter->model();

        const PointField* fx = findField(msg, "x");
        const PointField* fy = findField(msg, "y");
        const PointField* fz = findField(msg, "z");
        const PointField* fi =
            m.use_intensity ? findField(msg, m.intensity_field) : nullptr;
        if (!fx || !fy || !fz ||
            fx->datatype != PointField::FLOAT32 ||
            fy->datatype != PointField::FLOAT32 ||
            fz->datatype != PointField::FLOAT32 ||
            msg.is_bigendian)
        {
            RCLCPP_ERROR_THROTTLE(
                this->get_logger(), *this->get_clock(), 5000,
                "the cloud needs little-endian float32 x, y, z fields");
            return;
        }
        if (m.use_intensity && !fi)
        {
            RCLCPP_ERROR_THROTTLE(
                this->get_logger(), *this->get_clock(), 5000,
                "this model needs the '%s' field, which the cloud does not have: "
                "use the range-only model (artifact_xyz.model.txt)",
                m.intensity_field.c_str());
            return;
        }

        // The classifier takes plain arrays.
        const size_t n = static_cast<size_t>(msg.width) * msg.height;
        this->xyz.resize(n * 3);
        this->intensity.resize(fi ? n : 0);
        for (size_t i = 0; i < n; i++)
        {
            const uint8_t* p = msg.data.data() + i * msg.point_step;
            std::memcpy(&this->xyz[3 * i + 0], p + fx->offset, 4);
            std::memcpy(&this->xyz[3 * i + 1], p + fy->offset, 4);
            std::memcpy(&this->xyz[3 * i + 2], p + fz->offset, 4);
            if (fi) { this->intensity[i] = readAs(p + fi->offset, fi->datatype); }
        }
        const int64_t stamp =
            static_cast<int64_t>(msg.header.stamp.sec) * 1000000000LL + msg.header.stamp.nanosec;

        const size_t flagged = this->filter->process(
            this->xyz.data(), n, fi ? this->intensity.data() : nullptr, stamp,
            this->prob, this->threshold);

        // prob < 0: not classified. Those are invalid points (dropped), points
        // under the minimum range (kept only with keep_near) and the farther
        // returns of a multi-echo beam (kept).
        std::vector<uint8_t> keep(n), bad(n);
        for (size_t i = 0; i < n; i++)
        {
            const float x = this->xyz[3 * i], y = this->xyz[3 * i + 1], z = this->xyz[3 * i + 2];
            const double rng = std::hypot(std::hypot(static_cast<double>(x), static_cast<double>(y)), static_cast<double>(z));
            const bool valid = std::isfinite(rng) && rng > 1e-4;
            const bool near = rng < m.min_range_m;
            bad[i] = this->prob[i] >= this->threshold;
            keep[i] = valid && !bad[i] && (!near || this->keep_near);
        }

        PointCloudMsg out;
        copySelected(msg, keep, out);
        this->filtered_pub->publish(out);
        if (this->artifacts_pub)
        {
            copySelected(msg, bad, out);
            this->artifacts_pub->publish(out);
        }
        if (this->classified_pub)
        {
            PointCloudMsg c;
            c.header = msg.header;
            c.height = 1;
            c.width = static_cast<uint32_t>(n);
            c.is_bigendian = false;
            c.is_dense = msg.is_dense;
            c.fields = msg.fields;
            PointField pf;
            pf.name = "artifact_prob";
            pf.offset = msg.point_step;
            pf.datatype = PointField::FLOAT32;
            pf.count = 1;
            c.fields.push_back(pf);
            c.point_step = msg.point_step + 4;
            c.row_step = c.point_step * c.width;
            c.data.resize(static_cast<size_t>(c.row_step));
            for (size_t i = 0; i < n; i++)
            {
                uint8_t* d = c.data.data() + i * c.point_step;
                std::memcpy(d, msg.data.data() + i * msg.point_step, msg.point_step);
                std::memcpy(d + msg.point_step, &this->prob[i], 4);
            }
            this->classified_pub->publish(c);
        }

        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
        RCLCPP_INFO_THROTTLE(
            this->get_logger(), *this->get_clock(), 5000,
            "%zu points, %zu flagged, %.1f ms per scan", n, flagged, ms);
    }

private:
    std::unique_ptr<artifact_filter::Filter> filter;
    double threshold = -1.;
    bool keep_near = false;

    rclcpp::Subscription<PointCloudMsg>::SharedPtr sub;
    rclcpp::Publisher<PointCloudMsg>::SharedPtr filtered_pub, artifacts_pub, classified_pub;

    std::vector<float> xyz, intensity, prob;
};


int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    try
    {
        rclcpp::spin(std::make_shared<ArtifactFilterNode>());
    }
    catch (const std::exception& e)
    {
        RCLCPP_FATAL(rclcpp::get_logger("artifact_filter"), "%s", e.what());
        rclcpp::shutdown();
        return 1;
    }
    rclcpp::shutdown();
    return 0;
}
