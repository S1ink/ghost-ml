/*******************************************************************************
*   Copyright (C) 2024-2026 Cardinal Space Mining Club                         *
*                                                                              *
*                                 ;xxxxxxx:                                    *
*                                ;$$$$$$$$$       ...::..                      *
*                                $$$$$$$$$$x   .:::::::::::..                  *
*                             x$$$$$$$$$$$$$$::::::::::::::::.                 *
*                         :$$$$$&X;      .xX:::::::::::::.::...                *
*                 .$$Xx++$$$$+  :::.     :;:   .::::::.  ....  :               *
*                :$$$$$$$$$  ;:      ;xXXXXXXXx  .::.  .::::. .:.              *
*               :$$$$$$$$: ;      ;xXXXXXXXXXXXXx: ..::::::  .::.              *
*              ;$$$$$$$$ ::   :;XXXXXXXXXXXXXXXXXX+ .::::.  .:::               *
*               X$$$$$X : +XXXXXXXXXXXXXXXXXXXXXXXX; .::  .::::.               *
*                .$$$$ :xXXXXXXXXXXXXXXXXXXXXXXXXXXX.   .:::::.                *
*                 X$$X XXXXXXXXXXXXXXXXXXXXXXXXXXXXx:  .::::.                  *
*                 $$$:.XXXXXXXXXXXXXXXXXXXXXXXXXXX  ;; ..:.                    *
*                 $$& :XXXXXXXXXXXXXXXXXXXXXXXX;  +XX; X$$;                    *
*                 $$$: XXXXXXXXXXXXXXXXXXXXXX; :XXXXX; X$$;                    *
*                 X$$X XXXXXXXXXXXXXXXXXXX; .+XXXXXXX; $$$                     *
*                 $$$$ ;XXXXXXXXXXXXXXX+  +XXXXXXXXx+ X$$$+                    *
*               x$$$$$X ;XXXXXXXXXXX+ :xXXXXXXXX+   .;$$$$$$                   *
*              +$$$$$$$$ ;XXXXXXx;;+XXXXXXXXX+    : +$$$$$$$$                  *
*               +$$$$$$$$: xXXXXXXXXXXXXXX+      ; X$$$$$$$$                   *
*                :$$$$$$$$$. +XXXXXXXXX;      ;: x$$$$$$$$$                    *
*                ;x$$$$XX$$$$+ .;+X+      :;: :$$$$$xX$$$X                     *
*               ;;;;;;;;;;X$$$$$$$+      :X$$$$$$&.                            *
*               ;;;;;;;:;;;;;x$$$$$$$$$$$$$$$$x.                               *
*               :;;;;;;;;;;;;.  :$$$$$$$$$$X                                   *
*                .;;;;;;;;:;;    +$$$$$$$$$                                    *
*                  .;;;;;;.       X$$$$$$$:                                    *
*                                                                              *
*   Unless required by applicable law or agreed to in writing, software        *
*   distributed under the License is distributed on an "AS IS" BASIS,          *
*   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
*   See the License for the specific language governing permissions and        *
*   limitations under the License.                                             *
*                                                                              *
*******************************************************************************/

/*  artifact_filter_node: an example of running the lidar artifact classifier
 *  on a live point cloud. Every incoming scan is classified with a model
 *  exported from annotator_app and republished with the artifact points
 *  removed.
 *
 *  The classifier is multiscan_driver's (include/multiscan_driver/
 *  artifact_filter/), the same code the driver filters with.
 *
 *  Parameters
 *    model_file          (required) artifact_<variant>.model.txt from the
 *                        project's models folder. It must be a realtime
 *                        model.
 *    input_topic         PointCloud2 from the lidar, in the lidar's own frame
 *                        (default /multiscan/lidar_scan)
 *    threshold           flag points with probability >= this; a negative
 *                        value (default) uses the one stored in the model
 *    early_exit          stop evaluating the trees for points that are
 *                        clearly clean (default true). Several times faster
 *                        and flags practically the same points, but
 *                        artifact_prob of points under the threshold is then
 *                        approximate. Set false for exact probabilities.
 *    keep_near           keep points nearer than the model's minimum range,
 *                        which are never classified (default false)
 *    publish_artifacts   also publish the removed points (default true)
 *    publish_classified  also publish every point with an extra float32
 *                        field `artifact_prob`, -1 where not classified
 *                        (default false)
 *
 *  Topics (under the node's name)
 *    ~/filtered     the scan without artifact points, same fields as the input
 *    ~/artifacts    only the artifact points
 *    ~/classified   every point plus artifact_prob
 *
 *    ros2 run ghost_tester artifact_filter_node --ros-args \
 *        -p model_file:=/path/to/models/artifact_xyz.model.txt
 */
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>

#include <multiscan_driver/artifact_filter/filter.hpp>


class ArtifactFilterNode : public rclcpp::Node
{
    using PointCloudMsg = sensor_msgs::msg::PointCloud2;
    using PointFieldMsg = sensor_msgs::msg::PointField;

public:
    ArtifactFilterNode();

protected:
    /* The input fields the classifier reads. */
    struct InputFields
    {
        const PointFieldMsg* x{nullptr};
        const PointFieldMsg* y{nullptr};
        const PointFieldMsg* z{nullptr};
        const PointFieldMsg* intensity{nullptr};  // only for full models
    };

protected:
    static const PointFieldMsg* findField(
        const PointCloudMsg& cloud,
        const std::string& name);
    /* One value of any PointField type as a float. */
    static float readAsFloat(const uint8_t* value, uint8_t datatype);
    /* A cloud holding the selected points of `source`, bytes copied as they
     * are. */
    static void copySelected(
        const PointCloudMsg& source,
        const std::vector<uint8_t>& selected,
        PointCloudMsg& out);

protected:
    void scanCallback(const PointCloudMsg& cloud);
    bool findInputFields(const PointCloudMsg& cloud, InputFields& fields);
    void extractPoints(const PointCloudMsg& cloud, const InputFields& fields);
    void selectPoints(
        size_t num_points,
        std::vector<uint8_t>& keep,
        std::vector<uint8_t>& artifacts) const;
    void publishClassified(const PointCloudMsg& cloud, size_t num_points);

protected:
    std::unique_ptr<artifact_filter::Filter> filter;
    double threshold{-1.};
    bool keep_near{false};

    rclcpp::Subscription<PointCloudMsg>::SharedPtr scan_subscription;
    rclcpp::Publisher<PointCloudMsg>::SharedPtr filtered_publisher;
    rclcpp::Publisher<PointCloudMsg>::SharedPtr artifacts_publisher;
    rclcpp::Publisher<PointCloudMsg>::SharedPtr classified_publisher;

    // per scan, kept to reuse their memory
    std::vector<float> xyz;
    std::vector<float> intensity;
    std::vector<float> probability;
};



// --- Setup -------------------------------------------------------------------

ArtifactFilterNode::ArtifactFilterNode() : Node("artifact_filter")
{
    const std::string model_file =
        this->declare_parameter<std::string>("model_file", "");
    const std::string input_topic = this->declare_parameter<std::string>(
        "input_topic",
        "/multiscan/lidar_scan");
    this->threshold = this->declare_parameter<double>("threshold", -1.);
    this->keep_near = this->declare_parameter<bool>("keep_near", false);
    const bool early_exit = this->declare_parameter<bool>("early_exit", true);
    const bool publish_artifacts =
        this->declare_parameter<bool>("publish_artifacts", true);
    const bool publish_classified =
        this->declare_parameter<bool>("publish_classified", false);

    if (model_file.empty())
    {
        throw std::runtime_error(
            "parameter 'model_file' is not set: give the path of an exported "
            "model, e.g. -p model_file:=<project>/models/artifact_xyz.model.txt");
    }
    this->filter = std::make_unique<artifact_filter::Filter>(
        artifact_filter::Model::load(model_file));
    const artifact_filter::Model& model = this->filter->model();
    if (this->threshold < 0.)
    {
        this->threshold = model.threshold;
    }
    this->filter->setEarlyExit(early_exit);
    if (early_exit && !model.hasEarlyExit())
    {
        RCLCPP_WARN(
            this->get_logger(),
            "the model file has no early exit checkpoints (trained before "
            "they were added): every tree is evaluated for every point. "
            "Retrain to get them.");
    }
    else if (early_exit && this->threshold < 0.05)
    {
        RCLCPP_WARN(
            this->get_logger(),
            "early exit is calibrated for thresholds of 0.05 and up; with "
            "%.3f some points may be missed. Set early_exit:=false.",
            this->threshold);
    }

    const std::string inputs =
        model.use_intensity
            ? "uses intensity field '" + model.intensity_field + "'"
            : std::string("range only");
    RCLCPP_INFO(
        this->get_logger(),
        "model %s (%s, trained %s): %zu trees, %d features, grid %d x %d, "
        "%s, threshold %.2f. Listening on %s",
        model_file.c_str(),
        model.variant.c_str(),
        model.trained.c_str(),
        model.tree_roots.size(),
        model.num_features,
        model.num_rings,
        model.num_columns,
        inputs.c_str(),
        this->threshold,
        input_topic.c_str());

    const rclcpp::SensorDataQoS qos;
    this->filtered_publisher =
        this->create_publisher<PointCloudMsg>("~/filtered", qos);
    if (publish_artifacts)
    {
        this->artifacts_publisher =
            this->create_publisher<PointCloudMsg>("~/artifacts", qos);
    }
    if (publish_classified)
    {
        this->classified_publisher =
            this->create_publisher<PointCloudMsg>("~/classified", qos);
    }
    this->scan_subscription = this->create_subscription<PointCloudMsg>(
        input_topic,
        qos,
        [this](const PointCloudMsg::ConstSharedPtr& cloud)
        { this->scanCallback(*cloud); });
}



// --- Point cloud helpers -----------------------------------------------------

const ArtifactFilterNode::PointFieldMsg* ArtifactFilterNode::findField(
    const PointCloudMsg& cloud,
    const std::string& name)
{
    for (const PointFieldMsg& field : cloud.fields)
    {
        if (field.name == name)
        {
            return &field;
        }
    }
    return nullptr;
}

float ArtifactFilterNode::readAsFloat(const uint8_t* value, uint8_t datatype)
{
    const auto read = [value](auto typed)
    {
        std::memcpy(&typed, value, sizeof(typed));
        return static_cast<float>(typed);
    };
    switch (datatype)
    {
        case PointFieldMsg::INT8:
            return read(int8_t{});
        case PointFieldMsg::UINT8:
            return read(uint8_t{});
        case PointFieldMsg::INT16:
            return read(int16_t{});
        case PointFieldMsg::UINT16:
            return read(uint16_t{});
        case PointFieldMsg::INT32:
            return read(int32_t{});
        case PointFieldMsg::UINT32:
            return read(uint32_t{});
        case PointFieldMsg::FLOAT32:
            return read(float{});
        case PointFieldMsg::FLOAT64:
            return read(double{});
        default:
            return 0.f;
    }
}

void ArtifactFilterNode::copySelected(
    const PointCloudMsg& source,
    const std::vector<uint8_t>& selected,
    PointCloudMsg& out)
{
    out.header = source.header;
    out.fields = source.fields;
    out.is_bigendian = source.is_bigendian;
    out.point_step = source.point_step;
    out.is_dense = source.is_dense;
    out.height = 1;
    out.data.clear();
    out.data.reserve(source.data.size());
    uint32_t num_selected = 0;
    for (size_t point = 0; point < selected.size(); point++)
    {
        if (!selected[point])
        {
            continue;
        }
        const uint8_t* values = source.data.data() + point * source.point_step;
        out.data.insert(out.data.end(), values, values + source.point_step);
        num_selected++;
    }
    out.width = num_selected;
    out.row_step = num_selected * source.point_step;
}



// --- Scans -------------------------------------------------------------------

void ArtifactFilterNode::scanCallback(const PointCloudMsg& cloud)
{
    const auto start = std::chrono::steady_clock::now();
    InputFields fields;
    if (!this->findInputFields(cloud, fields))
    {
        return;
    }

    // the classifier takes plain arrays
    const size_t num_points = static_cast<size_t>(cloud.width) * cloud.height;
    this->extractPoints(cloud, fields);
    const int64_t stamp_ns =
        static_cast<int64_t>(cloud.header.stamp.sec) * 1000000000LL +
        cloud.header.stamp.nanosec;
    const size_t flagged = this->filter->process(
        this->xyz.data(),
        num_points,
        fields.intensity ? this->intensity.data() : nullptr,
        stamp_ns,
        this->probability,
        this->threshold);

    std::vector<uint8_t> keep;
    std::vector<uint8_t> artifacts;
    this->selectPoints(num_points, keep, artifacts);
    PointCloudMsg out;
    copySelected(cloud, keep, out);
    this->filtered_publisher->publish(out);
    if (this->artifacts_publisher)
    {
        copySelected(cloud, artifacts, out);
        this->artifacts_publisher->publish(out);
    }
    if (this->classified_publisher)
    {
        this->publishClassified(cloud, num_points);
    }

    const double elapsed_ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - start)
                                  .count();
    RCLCPP_INFO_THROTTLE(
        this->get_logger(),
        *this->get_clock(),
        5000,
        "%zu points, %zu flagged, %.1f ms per scan",
        num_points,
        flagged,
        elapsed_ms);
}

bool ArtifactFilterNode::findInputFields(
    const PointCloudMsg& cloud,
    InputFields& fields)
{
    const artifact_filter::Model& model = this->filter->model();
    fields.x = findField(cloud, "x");
    fields.y = findField(cloud, "y");
    fields.z = findField(cloud, "z");
    fields.intensity =
        model.use_intensity ? findField(cloud, model.intensity_field) : nullptr;

    const auto isFloat32 = [](const PointFieldMsg* field)
    { return field && field->datatype == PointFieldMsg::FLOAT32; };
    if (!isFloat32(fields.x) || !isFloat32(fields.y) || !isFloat32(fields.z) ||
        cloud.is_bigendian)
    {
        RCLCPP_ERROR_THROTTLE(
            this->get_logger(),
            *this->get_clock(),
            5000,
            "the cloud needs little-endian float32 x, y, z fields");
        return false;
    }
    if (model.use_intensity && !fields.intensity)
    {
        RCLCPP_ERROR_THROTTLE(
            this->get_logger(),
            *this->get_clock(),
            5000,
            "this model needs the '%s' field, which the cloud does not have: "
            "use the range-only model (artifact_xyz.model.txt)",
            model.intensity_field.c_str());
        return false;
    }
    return true;
}

void ArtifactFilterNode::extractPoints(
    const PointCloudMsg& cloud,
    const InputFields& fields)
{
    const size_t num_points = static_cast<size_t>(cloud.width) * cloud.height;
    this->xyz.resize(num_points * 3);
    this->intensity.resize(fields.intensity ? num_points : 0);
    for (size_t point = 0; point < num_points; point++)
    {
        const uint8_t* values = cloud.data.data() + point * cloud.point_step;
        std::memcpy(&this->xyz[3 * point + 0], values + fields.x->offset, 4);
        std::memcpy(&this->xyz[3 * point + 1], values + fields.y->offset, 4);
        std::memcpy(&this->xyz[3 * point + 2], values + fields.z->offset, 4);
        if (fields.intensity)
        {
            this->intensity[point] = readAsFloat(
                values + fields.intensity->offset,
                fields.intensity->datatype);
        }
    }
}

/* Probability < 0: not classified. Those are invalid points (dropped),
 * points under the minimum range (kept only with keep_near) and the farther
 * returns of a multi-echo beam (kept). */
void ArtifactFilterNode::selectPoints(
    size_t num_points,
    std::vector<uint8_t>& keep,
    std::vector<uint8_t>& artifacts) const
{
    const double min_range_m = this->filter->model().min_range_m;
    keep.assign(num_points, 0);
    artifacts.assign(num_points, 0);
    for (size_t point = 0; point < num_points; point++)
    {
        const double x = this->xyz[3 * point];
        const double y = this->xyz[3 * point + 1];
        const double z = this->xyz[3 * point + 2];
        const double range = std::hypot(std::hypot(x, y), z);
        const bool valid = std::isfinite(range) && range > 1e-4;
        const bool near = range < min_range_m;
        artifacts[point] = this->probability[point] >= this->threshold;
        keep[point] = valid && !artifacts[point] && (!near || this->keep_near);
    }
}

/* Every point with an extra float32 field artifact_prob. */
void ArtifactFilterNode::publishClassified(
    const PointCloudMsg& cloud,
    size_t num_points)
{
    PointCloudMsg classified;
    classified.header = cloud.header;
    classified.height = 1;
    classified.width = static_cast<uint32_t>(num_points);
    classified.is_bigendian = false;
    classified.is_dense = cloud.is_dense;
    classified.fields = cloud.fields;
    PointFieldMsg probability_field;
    probability_field.name = "artifact_prob";
    probability_field.offset = cloud.point_step;
    probability_field.datatype = PointFieldMsg::FLOAT32;
    probability_field.count = 1;
    classified.fields.push_back(probability_field);
    classified.point_step = cloud.point_step + 4;
    classified.row_step = classified.point_step * classified.width;
    classified.data.resize(static_cast<size_t>(classified.row_step));
    for (size_t point = 0; point < num_points; point++)
    {
        uint8_t* out = classified.data.data() + point * classified.point_step;
        std::memcpy(
            out,
            cloud.data.data() + point * cloud.point_step,
            cloud.point_step);
        std::memcpy(out + cloud.point_step, &this->probability[point], 4);
    }
    this->classified_publisher->publish(classified);
}


int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    try
    {
        rclcpp::spin(std::make_shared<ArtifactFilterNode>());
    }
    catch (const std::exception& error)
    {
        RCLCPP_FATAL(rclcpp::get_logger("artifact_filter"), "%s", error.what());
        rclcpp::shutdown();
        return 1;
    }
    rclcpp::shutdown();
    return 0;
}
