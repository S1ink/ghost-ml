/*
 * artifact_filter_check <model.txt> <check.bin>
 *
 * Feeds the scans of a check file (annotator_app/export_model.py --check-file)
 * through artifact_filter.hpp and compares the result with the probabilities
 * Python computed for the same scans. Also reports the time per scan.
 * No ROS needed. Exit code 0 when the two agree.
 */
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include "artifact_filter/artifact_filter.hpp"

template<typename T>
static bool readInto(std::ifstream& in, T* out, size_t n)
{
    in.read(reinterpret_cast<char*>(out), static_cast<std::streamsize>(n * sizeof(T)));
    return static_cast<bool>(in);
}

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cerr << "usage: artifact_filter_check <model.txt> <check.bin>\n";
        return 2;
    }
    try
    {
        artifact_filter::Filter filter(artifact_filter::Model::load(argv[1]));
        const artifact_filter::Model& m = filter.model();

        std::ifstream in(argv[2], std::ios::binary);
        char magic[8];
        int32_t head[4];
        if (!readInto(in, magic, 8) || std::memcmp(magic, "LACHECK1", 8) != 0 ||
            !readInto(in, head, 4))
        {
            std::cerr << argv[2] << " is not a check file\n";
            return 2;
        }
        const int n_scans = head[0], rings = head[1], cols = head[2];
        const bool has_int = head[3] != 0;
        if (rings != m.rings || cols != m.cols || has_int != m.use_intensity)
        {
            std::cerr << "the check file was made for another model\n";
            return 2;
        }
        const size_t n_cells = static_cast<size_t>(rings) * static_cast<size_t>(cols);

        size_t n_cmp = 0, n_off = 0, n_flag_py = 0, n_flag_cpp = 0, n_flag_differ = 0;
        size_t n_pt = 0, n_pt_off = 0;
        double max_diff = 0., total_ms = 0.;
        std::vector<float> xyz, inten, want, want_pt, got_pt;
        filter.keep_features(true);
        int shown = 0;
        // kept for the second pass (early exit)
        struct Kept { int64_t stamp; std::vector<float> xyz, inten, exact; };
        std::vector<Kept> kept;
        for (int s = 0; s < n_scans; s++)
        {
            int64_t stamp;
            int32_t n;
            if (!readInto(in, &stamp, 1) || !readInto(in, &n, 1)) { break; }
            xyz.resize(static_cast<size_t>(n) * 3);
            inten.resize(has_int ? static_cast<size_t>(n) : 0);
            want.resize(n_cells);
            want_pt.resize(static_cast<size_t>(n));
            if (!readInto(in, xyz.data(), xyz.size()) ||
                (has_int && !readInto(in, inten.data(), inten.size())) ||
                !readInto(in, want.data(), n_cells) ||
                !readInto(in, want_pt.data(), want_pt.size()))
            {
                std::cerr << "check file is truncated\n";
                return 2;
            }

            const auto t0 = std::chrono::steady_clock::now();
            filter.process(
                xyz.data(), static_cast<size_t>(n), has_int ? inten.data() : nullptr,
                stamp, got_pt);
            total_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count();

            const std::vector<float>& got = filter.cell_probabilities();
            kept.push_back({stamp, xyz, inten, got});
            for (size_t c = 0; c < n_cells; c++)
            {
                const double d = std::abs(static_cast<double>(got[c]) - want[c]);
                max_diff = std::max(max_diff, d);
                n_cmp++;
                n_off += (d > 1e-3);
                const bool a = want[c] >= m.threshold, b = got[c] >= m.threshold;
                n_flag_py += a;
                n_flag_cpp += b;
                n_flag_differ += (a != b);
            }
            for (size_t i = 0; i < static_cast<size_t>(n); i++)
            {
                n_pt++;
                const bool off = std::abs(static_cast<double>(got_pt[i]) - want_pt[i]) > 1e-3;
                n_pt_off += off;
                if (off && shown < 6)
                {
                    shown++;
                    std::printf(
                        "  e.g. scan %d point %zu (%.3f, %.3f, %.3f): Python %.4f, C++ %.4f\n",
                        s, i, xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2], want_pt[i], got_pt[i]);
                }
            }
        }

        // The classifier inputs of the last scan, one by one.
        int32_t n_feat = 0;
        if (readInto(in, &n_feat, 1) && n_feat == m.n_features)
        {
            std::vector<float> want_f(n_cells * static_cast<size_t>(n_feat));
            if (readInto(in, want_f.data(), want_f.size()))
            {
                const std::vector<float>& got_f = filter.last_features();
                std::vector<size_t> bad(static_cast<size_t>(n_feat), 0);
                std::vector<double> worst(static_cast<size_t>(n_feat), 0.);
                size_t n_bad = 0;
                for (size_t j = 0; j < want_f.size(); j++)
                {
                    const float a = want_f[j], b = got_f[j];
                    const bool differ = std::isnan(a) != std::isnan(b) ||
                        (!std::isnan(a) && a != b);      // exactly: see artifact_features._log
                    if (differ)
                    {
                        const size_t f = j % static_cast<size_t>(n_feat);
                        bad[f]++;
                        n_bad++;
                        worst[f] = std::max(
                            worst[f], std::isnan(a) || std::isnan(b) ? 1e9 : std::abs(static_cast<double>(a) - b));
                    }
                }
                std::printf("  classifier inputs of the last scan: %zu of %zu differ\n", n_bad, want_f.size());
                for (int f = 0; f < n_feat; f++)
                {
                    if (bad[static_cast<size_t>(f)])
                    {
                        std::printf("    input %d: %zu cells, largest difference %s%.3g\n", f,
                                    bad[static_cast<size_t>(f)],
                                    worst[static_cast<size_t>(f)] > 1e8 ? "(missing in one) " : "",
                                    worst[static_cast<size_t>(f)] > 1e8 ? 0. : worst[static_cast<size_t>(f)]);
                    }
                }
            }
        }

        std::printf(
            "%d scans, %zu grid cells, %zu points compared with Python\n"
            "  largest probability difference   %.2e\n"
            "  cells differing by more than 1e-3  %zu\n"
            "  points differing by more than 1e-3 %zu\n"
            "  flagged at threshold %.2f          Python %zu, C++ %zu, disagree on %zu\n"
            "  time per scan                      %.2f ms (%zu trees, %d features)\n",
            n_scans, n_cmp, n_pt, max_diff, n_off, n_pt_off, m.threshold,
            n_flag_py, n_flag_cpp, n_flag_differ, total_ms / std::max(n_scans, 1),
            m.tree_root.size(), m.n_features);

        // Second pass with early exit: how much faster, and does it change
        // which cells are flagged (at the model's threshold and at 0.5)?
        if (!m.exit_after.empty())
        {
            artifact_filter::Filter fast(artifact_filter::Model::load(argv[1]));
            fast.early_exit(true);
            double fast_ms = 0.;
            size_t changed = 0, changed_half = 0, flagged = 0;
            for (const Kept& k : kept)
            {
                const auto t0 = std::chrono::steady_clock::now();
                fast.process(
                    k.xyz.data(), k.xyz.size() / 3, has_int ? k.inten.data() : nullptr,
                    k.stamp, got_pt);
                fast_ms += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0).count();
                const std::vector<float>& got = fast.cell_probabilities();
                for (size_t c = 0; c < n_cells; c++)
                {
                    flagged += (k.exact[c] >= m.threshold);
                    changed += ((k.exact[c] >= m.threshold) != (got[c] >= m.threshold));
                    changed_half += ((k.exact[c] >= 0.5f) != (got[c] >= 0.5f));
                }
            }
            std::printf(
                "  with early exit                    %.2f ms per scan; of %zu flagged cells, "
                "%zu change at threshold %.2f and %zu at 0.50\n",
                fast_ms / std::max<size_t>(kept.size(), 1), flagged, changed, m.threshold,
                changed_half);
        }
        else
        {
            std::puts("  (the model file has no early exit checkpoints)");
        }

        // The two are built to give the same numbers; only the last digits
        // of the final sum may differ.
        const bool ok = n_off == 0 && n_pt_off == 0 && n_flag_differ == 0;
        std::puts(ok ? "OK: C++ and Python agree" : "MISMATCH");
        return ok ? 0 : 1;
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << "\n";
        return 2;
    }
}
