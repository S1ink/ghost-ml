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

/*  artifact_filter_check <model.txt> <check.bin>
 *
 *  Feeds the scans of a check file (annotator_app/export_model.py
 *  --check-file) through multiscan_driver's artifact filter and compares the
 *  result with the probabilities Python computed for the same scans. Also
 *  reports the time per scan. No ROS needed. Exit code 0 when the two agree.
 */
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include <multiscan_driver/artifact_filter/filter.hpp>


namespace
{

using Clock = std::chrono::steady_clock;

constexpr double TOLERANCE = 1e-3;
constexpr int EXAMPLES_SHOWN = 6;

template<typename T> bool readInto(std::ifstream& in, T* out, size_t count)
{
    in.read(
        reinterpret_cast<char*>(out),
        static_cast<std::streamsize>(count * sizeof(T)));
    return static_cast<bool>(in);
}

double millisecondsSince(Clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - start)
        .count();
}

struct CheckFileHeader
{
    int num_scans;
    int num_rings;
    int num_columns;
    bool has_intensity;
};

/* One scan of the check file, with Python's answers. */
struct CheckScan
{
    int64_t stamp_ns;
    std::vector<float> xyz;
    std::vector<float> intensity;
    std::vector<float> expected_cell_probability;
    std::vector<float> expected_point_probability;
    std::vector<float> cell_probability;  // C++, filled in by the comparison
};

/* What differs between C++ and Python. */
struct Comparison
{
    size_t cells_compared{0};
    size_t cells_off{0};
    size_t points_compared{0};
    size_t points_off{0};
    size_t flagged_python{0};
    size_t flagged_cpp{0};
    size_t flags_differing{0};
    double largest_difference{0.};
    double total_ms{0.};
    int examples_shown{0};
};


// --- Reading -----------------------------------------------------------------

bool readHeader(std::ifstream& in, CheckFileHeader& header)
{
    char magic[8];
    int32_t values[4];
    if (!readInto(in, magic, 8) || std::memcmp(magic, "LACHECK1", 8) != 0 ||
        !readInto(in, values, 4))
    {
        return false;
    }
    header = {values[0], values[1], values[2], values[3] != 0};
    return true;
}

/* False at the end of the file; throws when it is truncated. */
bool readScan(
    std::ifstream& in,
    const CheckFileHeader& header,
    size_t num_cells,
    CheckScan& scan)
{
    int32_t num_points;
    if (!readInto(in, &scan.stamp_ns, 1) || !readInto(in, &num_points, 1))
    {
        return false;
    }
    const size_t points = static_cast<size_t>(num_points);
    scan.xyz.resize(points * 3);
    scan.intensity.resize(header.has_intensity ? points : 0);
    scan.expected_cell_probability.resize(num_cells);
    scan.expected_point_probability.resize(points);
    if (!readInto(in, scan.xyz.data(), scan.xyz.size()) ||
        (header.has_intensity &&
         !readInto(in, scan.intensity.data(), scan.intensity.size())) ||
        !readInto(in, scan.expected_cell_probability.data(), num_cells) ||
        !readInto(in, scan.expected_point_probability.data(), points))
    {
        throw std::runtime_error("check file is truncated");
    }
    return true;
}


// --- Comparing ---------------------------------------------------------------

void compareScan(
    int scan_index,
    CheckScan& scan,
    artifact_filter::Filter& filter,
    Comparison& comparison)
{
    const double threshold = filter.model().threshold;
    const size_t num_points = scan.xyz.size() / 3;
    std::vector<float> point_probability;
    const Clock::time_point start = Clock::now();
    filter.process(
        scan.xyz.data(),
        num_points,
        scan.intensity.empty() ? nullptr : scan.intensity.data(),
        scan.stamp_ns,
        point_probability);
    comparison.total_ms += millisecondsSince(start);
    scan.cell_probability = filter.cellProbabilities();

    for (size_t cell = 0; cell < scan.cell_probability.size(); cell++)
    {
        const double expected = scan.expected_cell_probability[cell];
        const double computed = scan.cell_probability[cell];
        const double difference = std::abs(computed - expected);
        comparison.largest_difference =
            std::max(comparison.largest_difference, difference);
        comparison.cells_compared++;
        comparison.cells_off += (difference > TOLERANCE);
        const bool python_flags = expected >= threshold;
        const bool cpp_flags = computed >= threshold;
        comparison.flagged_python += python_flags;
        comparison.flagged_cpp += cpp_flags;
        comparison.flags_differing += (python_flags != cpp_flags);
    }

    for (size_t point = 0; point < num_points; point++)
    {
        const float expected = scan.expected_point_probability[point];
        const float computed = point_probability[point];
        const bool off =
            std::abs(static_cast<double>(computed) - expected) > TOLERANCE;
        comparison.points_compared++;
        comparison.points_off += off;
        if (off && comparison.examples_shown < EXAMPLES_SHOWN)
        {
            comparison.examples_shown++;
            std::printf(
                "  e.g. scan %d point %zu (%.3f, %.3f, %.3f): Python %.4f, "
                "C++ %.4f\n",
                scan_index,
                point,
                scan.xyz[3 * point],
                scan.xyz[3 * point + 1],
                scan.xyz[3 * point + 2],
                expected,
                computed);
        }
    }
}

/* The classifier inputs of the last scan, one by one. They must match
 * exactly: see artifact_features._log. */
void compareFeatures(
    std::ifstream& in,
    const artifact_filter::Filter& filter,
    size_t num_cells)
{
    const int num_features = filter.model().num_features;
    int32_t stored_features = 0;
    if (!readInto(in, &stored_features, 1) || stored_features != num_features)
    {
        return;
    }
    std::vector<float> expected(num_cells * static_cast<size_t>(num_features));
    if (!readInto(in, expected.data(), expected.size()))
    {
        return;
    }

    const std::vector<float>& computed = filter.lastFeatures();
    std::vector<size_t> cells_differing(static_cast<size_t>(num_features), 0);
    std::vector<double> largest(static_cast<size_t>(num_features), 0.);
    size_t total_differing = 0;
    for (size_t i = 0; i < expected.size(); i++)
    {
        const float python = expected[i];
        const float cpp = computed[i];
        const bool one_missing = std::isnan(python) != std::isnan(cpp);
        if (!one_missing && (std::isnan(python) || python == cpp))
        {
            continue;
        }
        const size_t feature = i % static_cast<size_t>(num_features);
        cells_differing[feature]++;
        total_differing++;
        largest[feature] = std::max(
            largest[feature],
            one_missing ? 1e9 : std::abs(static_cast<double>(python) - cpp));
    }

    std::printf(
        "  classifier inputs of the last scan: %zu of %zu differ\n",
        total_differing,
        expected.size());
    for (size_t feature = 0; feature < cells_differing.size(); feature++)
    {
        if (!cells_differing[feature])
        {
            continue;
        }
        const bool missing = largest[feature] > 1e8;
        std::printf(
            "    input %zu: %zu cells, largest difference %s%.3g\n",
            feature,
            cells_differing[feature],
            missing ? "(missing in one) " : "",
            missing ? 0. : largest[feature]);
    }
}

/* A second pass with early exit: how much faster, and does it change which
 * cells are flagged (at the model's threshold and at 0.5)? */
void compareEarlyExit(
    const char* model_file,
    const std::vector<CheckScan>& scans)
{
    artifact_filter::Filter fast{artifact_filter::Model::load(model_file)};
    if (!fast.model().hasEarlyExit())
    {
        std::puts("  (the model file has no early exit checkpoints)");
        return;
    }
    fast.setEarlyExit(true);
    const double threshold = fast.model().threshold;

    double total_ms = 0.;
    size_t flagged = 0;
    size_t changed = 0;
    size_t changed_at_half = 0;
    std::vector<float> point_probability;
    for (const CheckScan& scan : scans)
    {
        const Clock::time_point start = Clock::now();
        fast.process(
            scan.xyz.data(),
            scan.xyz.size() / 3,
            scan.intensity.empty() ? nullptr : scan.intensity.data(),
            scan.stamp_ns,
            point_probability);
        total_ms += millisecondsSince(start);

        const std::vector<float>& fast_probability = fast.cellProbabilities();
        for (size_t cell = 0; cell < fast_probability.size(); cell++)
        {
            const float exact = scan.cell_probability[cell];
            flagged += (exact >= threshold);
            changed +=
                ((exact >= threshold) != (fast_probability[cell] >= threshold));
            changed_at_half +=
                ((exact >= 0.5f) != (fast_probability[cell] >= 0.5f));
        }
    }
    std::printf(
        "  with early exit                    %.2f ms per scan; of %zu "
        "flagged cells, %zu change at threshold %.2f and %zu at 0.50\n",
        total_ms / std::max<size_t>(scans.size(), 1),
        flagged,
        changed,
        threshold,
        changed_at_half);
}

void printSummary(
    int num_scans,
    const artifact_filter::Model& model,
    const Comparison& comparison)
{
    std::printf(
        "%d scans, %zu grid cells, %zu points compared with Python\n"
        "  largest probability difference   %.2e\n"
        "  cells differing by more than 1e-3  %zu\n"
        "  points differing by more than 1e-3 %zu\n"
        "  flagged at threshold %.2f          Python %zu, C++ %zu, disagree "
        "on %zu\n"
        "  time per scan                      %.2f ms (%zu trees, %d "
        "features)\n",
        num_scans,
        comparison.cells_compared,
        comparison.points_compared,
        comparison.largest_difference,
        comparison.cells_off,
        comparison.points_off,
        model.threshold,
        comparison.flagged_python,
        comparison.flagged_cpp,
        comparison.flags_differing,
        comparison.total_ms / std::max(num_scans, 1),
        model.tree_roots.size(),
        model.num_features);
}

int check(const char* model_file, const char* check_file)
{
    artifact_filter::Filter filter{artifact_filter::Model::load(model_file)};
    const artifact_filter::Model& model = filter.model();

    std::ifstream in(check_file, std::ios::binary);
    CheckFileHeader header;
    if (!readHeader(in, header))
    {
        std::cerr << check_file << " is not a check file\n";
        return 2;
    }
    if (header.num_rings != model.num_rings ||
        header.num_columns != model.num_columns ||
        header.has_intensity != model.use_intensity)
    {
        std::cerr << "the check file was made for another model\n";
        return 2;
    }
    const size_t num_cells = static_cast<size_t>(header.num_rings) *
                             static_cast<size_t>(header.num_columns);

    // exact probabilities first, keeping the scans for the early exit pass
    filter.setKeepFeatures(true);
    Comparison comparison;
    std::vector<CheckScan> scans;
    for (int index = 0; index < header.num_scans; index++)
    {
        CheckScan scan;
        if (!readScan(in, header, num_cells, scan))
        {
            break;
        }
        compareScan(index, scan, filter, comparison);
        scans.push_back(std::move(scan));
    }
    compareFeatures(in, filter, num_cells);
    printSummary(header.num_scans, model, comparison);
    compareEarlyExit(model_file, scans);

    // The two are built to give the same numbers; only the last digits of
    // the final sum may differ.
    const bool agree = comparison.cells_off == 0 &&
                       comparison.points_off == 0 &&
                       comparison.flags_differing == 0;
    std::puts(agree ? "OK: C++ and Python agree" : "MISMATCH");
    return agree ? 0 : 1;
}

}  // namespace


int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cerr << "usage: artifact_filter_check <model.txt> <check.bin>\n";
        return 2;
    }
    try
    {
        return check(argv[1], argv[2]);
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << "\n";
        return 2;
    }
}
