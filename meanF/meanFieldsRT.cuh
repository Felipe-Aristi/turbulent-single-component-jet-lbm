#ifndef MEANFIELDSRT_CUH
#define MEANFIELDSRT_CUH

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>

#include "../utilities/cudaUtilities.cuh"
#include "../constants.cuh"
#include "memoryMeanFields.cuh"

struct MeanFieldsFiles
{
    FILE *tke_total = nullptr;
    FILE *tke_avg = nullptr;
    FILE *tke_diff = nullptr;
    FILE *radial_sum_uy = nullptr;
    FILE *radial_sum_uy2 = nullptr;
    FILE *radial_sum_ur = nullptr;
    FILE *radial_sum_ur2 = nullptr;
    FILE *radial_sum_uruy = nullptr;
    FILE *radial_count = nullptr;
};

struct MeanFieldsRuntime
{
    MeanFieldsDevice device{};
    MeanFieldsHost host{};
    MeanFieldsState state{};
    MeanFieldsFiles files{};

    std::string case_dir{};
    std::string mean_dir{};
    std::string vti_dir{};
};

// ------------- Create directory ---------------
inline std::string make_case_folder_name()
{
    std::ostringstream oss;
    oss << "./JET_VTK/"
        << "Re" << static_cast<int>(std::round(Re));
    return oss.str();
}

inline bool ensure_case_directories(MeanFieldsRuntime &mf)
{
    mf.case_dir = make_case_folder_name();
    mf.mean_dir = mf.case_dir + "/mean_profiles";
    mf.vti_dir = mf.case_dir + "/vti_slices";

    std::error_code ec;

    std::filesystem::create_directories(mf.mean_dir, ec);
    if (ec)
    {
        std::fprintf(stderr, "Failed to create mean_profiles directory: %s\n", mf.mean_dir.c_str());
        return false;
    }

    ec.clear();
    std::filesystem::create_directories(mf.vti_dir, ec);
    if (ec)
    {
        std::fprintf(stderr, "Failed to create vti_data directory: %s\n", mf.vti_dir.c_str());
        return false;
    }

    return true;
}

// ---------------------------------------------------

inline bool open_mean_fields_files(MeanFieldsFiles &f, const std::string &mean_dir)
{
    const std::string path_total = mean_dir + "/tke_total.bin";
    const std::string path_avg = mean_dir + "/tke_avg.bin";
    const std::string path_diff = mean_dir + "/tke_diff.bin";
    const std::string path_radial_sum_uy = mean_dir + "/radial_sum_uy.bin";
    const std::string path_radial_sum_uy2 = mean_dir + "/radial_sum_uy2.bin";
    const std::string path_radial_sum_ur = mean_dir + "/radial_sum_ur.bin";
    const std::string path_radial_sum_ur2 = mean_dir + "/radial_sum_ur2.bin";
    const std::string path_radial_sum_uruy = mean_dir + "/radial_sum_uruy.bin";
    const std::string path_radial_count = mean_dir + "/radial_count.bin";

    f.tke_total = std::fopen(path_total.c_str(), "wb");
    f.tke_avg = std::fopen(path_avg.c_str(), "wb");
    f.tke_diff = std::fopen(path_diff.c_str(), "wb");
    f.radial_sum_uy = std::fopen(path_radial_sum_uy.c_str(), "wb");
    f.radial_sum_uy2 = std::fopen(path_radial_sum_uy2.c_str(), "wb");
    f.radial_sum_ur = std::fopen(path_radial_sum_ur.c_str(), "wb");
    f.radial_sum_ur2 = std::fopen(path_radial_sum_ur2.c_str(), "wb");
    f.radial_sum_uruy = std::fopen(path_radial_sum_uruy.c_str(), "wb");
    f.radial_count = std::fopen(path_radial_count.c_str(), "wb");

    if (f.tke_total == nullptr || f.tke_avg == nullptr ||
        f.tke_diff == nullptr ||
        f.radial_sum_uy == nullptr || f.radial_sum_uy2 == nullptr ||
        f.radial_sum_ur == nullptr || f.radial_sum_ur2 == nullptr ||
        f.radial_sum_uruy == nullptr || f.radial_count == nullptr)
    {
        std::fprintf(stderr, "Error opening one or more files in: %s\n", mean_dir.c_str());
        return false;
    }

    return true;
}

inline void close_mean_fields_files(MeanFieldsFiles &f)
{
    if (f.tke_total != nullptr)
        std::fclose(f.tke_total);
    if (f.tke_avg != nullptr)
        std::fclose(f.tke_avg);
    if (f.tke_diff != nullptr)
        std::fclose(f.tke_diff);
    if (f.radial_sum_uy != nullptr)
        std::fclose(f.radial_sum_uy);
    if (f.radial_sum_uy2 != nullptr)
        std::fclose(f.radial_sum_uy2);
    if (f.radial_sum_ur != nullptr)
        std::fclose(f.radial_sum_ur);
    if (f.radial_sum_ur2 != nullptr)
        std::fclose(f.radial_sum_ur2);
    if (f.radial_sum_uruy != nullptr)
        std::fclose(f.radial_sum_uruy);
    if (f.radial_count != nullptr)
        std::fclose(f.radial_count);

    f = MeanFieldsFiles{};
}

inline bool initialize_mean_fields_runtime(const int deviceID,
                                           MeanFieldsRuntime &mf)
{
    CUDA_CHECK(cudaSetDevice(deviceID));

    mf.device = allocate_mean_fields_device();
    mf.host = allocate_mean_fields_host();
    mf.state = MeanFieldsState{};

    if (!ensure_case_directories(mf))
    {
        free_mean_fields_device(mf.device);
        free_mean_fields_host(mf.host);
        return false;
    }

    if (!open_mean_fields_files(mf.files, mf.mean_dir))
    {
        std::fprintf(stderr, "Error opening one or more mean-field output files.\n");

        close_mean_fields_files(mf.files);
        free_mean_fields_device(mf.device);
        free_mean_fields_host(mf.host);

        return false;
    }

    return true;
}

inline void free_mean_fields_runtime(MeanFieldsRuntime &mf)
{
    close_mean_fields_files(mf.files);
    free_mean_fields_device(mf.device);
    free_mean_fields_host(mf.host);

    mf = MeanFieldsRuntime{};
}

inline void update_tke_state(MeanFieldsRuntime &mf, const int step)
{
    if (mf.state.first_tke_sample)
    {
        mf.host.delta = static_cast<real_t>(0);
        mf.host.abs_delta = static_cast<real_t>(0);
        mf.state.first_tke_sample = false;
    }
    else
    {
        mf.host.delta = mf.host.tke_total - mf.host.tke_total_prev;

        real_t denom = std::abs(mf.host.tke_total_prev);
        if (denom < static_cast<real_t>(1e-30))
        {
            denom = static_cast<real_t>(1e-30);
        }

        mf.host.abs_delta = std::abs(mf.host.delta) / denom;

        if (mf.host.abs_delta < MeanFieldsState::tke_tolerance)
        {
            ++mf.state.stable_energy_samples;
        }
        else
        {
            mf.state.stable_energy_samples = 0;
        }

        if (!mf.state.start_uy_average &&
            mf.state.stable_energy_samples >= MeanFieldsState::stable_energy_samples_required)
        {
            mf.state.start_uy_average = true;
            mf.state.step_uy_avg_start = static_cast<unsigned int>(step);

            std::cout << "Starting radial profile averages at step "
                      << mf.state.step_uy_avg_start
                      << " | relative_energy_delta = " << mf.host.abs_delta
                      << "\n";
        }
    }

    std::cout << std::scientific
              << "step " << step
              << " | tke_total = " << mf.host.tke_total
              << " | tke_avg = " << mf.host.tke_avg
              << " | relative_energy_delta = " << mf.host.abs_delta
              << " | stable_samples = " << mf.state.stable_energy_samples
              << "\n";

    mf.host.tke_total_prev = mf.host.tke_total;
    mf.host.tke_avg_prev = mf.host.tke_avg;
}

inline void write_tke_outputs(MeanFieldsRuntime &mf)
{
    std::fwrite(&mf.host.tke_total, sizeof(real_t), 1, mf.files.tke_total);
    std::fwrite(&mf.host.tke_avg, sizeof(real_t), 1, mf.files.tke_avg);
    std::fwrite(&mf.host.abs_delta, sizeof(real_t), 1, mf.files.tke_diff);

    std::fflush(mf.files.tke_total);
    std::fflush(mf.files.tke_avg);
    std::fflush(mf.files.tke_diff);
}

inline void copy_radial_profile_outputs(MeanFieldsRuntime &mf)
{
    CUDA_CHECK(cudaMemcpy(mf.host.sum_uy,
                          mf.device.sum_uy,
                          profileStatBytes,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(mf.host.sum_uy2,
                          mf.device.sum_uy2,
                          profileStatBytes,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(mf.host.sum_ur,
                          mf.device.sum_ur,
                          profileStatBytes,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(mf.host.sum_ur2,
                          mf.device.sum_ur2,
                          profileStatBytes,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(mf.host.sum_uruy,
                          mf.device.sum_uruy,
                          profileStatBytes,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(mf.host.count,
                          mf.device.count,
                          profileCountBytes,
                          cudaMemcpyDeviceToHost));
}

inline void write_radial_profile_metadata(const MeanFieldsRuntime &mf)
{
    const std::string metadata_path = mf.mean_dir + "/radial_profile_metadata.txt";
    std::ofstream out(metadata_path);

    out << "NX " << NX << "\n";
    out << "NY " << NY << "\n";
    out << "NZ " << NZ << "\n";
    out << "NR_BINS " << NR_BINS << "\n";
    out << "NradialProfileCells " << NradialProfileCells << "\n";
    out << "NSTATS_SAMPLE " << NSTATS_SAMPLE << "\n";
    out << "NPROFILE_OUTPUT " << NPROFILE_OUTPUT << "\n";
    out << "start_step " << mf.state.step_uy_avg_start << "\n";
    out << "radial_sample_count " << mf.state.radial_sample_count << "\n";
    out << "layout y_major_index_equals_y_times_NR_BINS_plus_rbin\n";
    out << "stat_dtype float64\n";
    out << "count_dtype uint64\n";
}

inline void write_radial_profile_outputs(MeanFieldsRuntime &mf)
{
    copy_radial_profile_outputs(mf);

    std::rewind(mf.files.radial_sum_uy);
    std::rewind(mf.files.radial_sum_uy2);
    std::rewind(mf.files.radial_sum_ur);
    std::rewind(mf.files.radial_sum_ur2);
    std::rewind(mf.files.radial_sum_uruy);
    std::rewind(mf.files.radial_count);

    std::fwrite(mf.host.sum_uy, sizeof(profile_stat_t), NradialProfileCells, mf.files.radial_sum_uy);
    std::fwrite(mf.host.sum_uy2, sizeof(profile_stat_t), NradialProfileCells, mf.files.radial_sum_uy2);
    std::fwrite(mf.host.sum_ur, sizeof(profile_stat_t), NradialProfileCells, mf.files.radial_sum_ur);
    std::fwrite(mf.host.sum_ur2, sizeof(profile_stat_t), NradialProfileCells, mf.files.radial_sum_ur2);
    std::fwrite(mf.host.sum_uruy, sizeof(profile_stat_t), NradialProfileCells, mf.files.radial_sum_uruy);
    std::fwrite(mf.host.count, sizeof(profile_count_t), NradialProfileCells, mf.files.radial_count);

    std::fflush(mf.files.radial_sum_uy);
    std::fflush(mf.files.radial_sum_uy2);
    std::fflush(mf.files.radial_sum_ur);
    std::fflush(mf.files.radial_sum_ur2);
    std::fflush(mf.files.radial_sum_uruy);
    std::fflush(mf.files.radial_count);

    write_radial_profile_metadata(mf);
}

inline void process_tke_sample(MeanFieldsRuntime &mf, const int step)
{
    update_tke_state(mf, step);
    write_tke_outputs(mf);
}

#endif
