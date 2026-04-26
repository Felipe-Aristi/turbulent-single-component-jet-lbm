#include <iostream>

#include "utilities/cudaUtilities.cuh"
#include "utilities/cudaConfig.cuh"
#include "utilities/countmlups.cuh"

#include "meanF/meanFieldsRT.cuh"

#include "constants.cuh"
#include "memory.cuh"
#include "launch.cuh"
#include "io/save_data.cuh"

constexpr int deviceID = 0;

int main()
{
    MeanFieldsRuntime mf{};

    if (!initialize_mean_fields_runtime(deviceID, mf))
    {
        CUDA_CHECK(cudaDeviceReset());
        return EXIT_FAILURE;
    }
    write_radial_profile_outputs(mf);

    CudaConfig cfg = print_device_and_make_config(deviceID);

    LbmDevice d = allocate_device_memory();

    launch_jetDensity(cfg, d);
    launch_Injet(cfg, d);

    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t evStart, evStop;
    CUDA_CHECK(cudaEventCreate(&evStart));
    CUDA_CHECK(cudaEventCreate(&evStop));

    MlupsStats perf{};
    CUDA_CHECK(cudaEventRecord(evStart));

    for (int step = 0; step < NSTEP; ++step)
    {
        launch_Macros(cfg, d);

        if (mf.state.start_uy_average && step % NSTATS_SAMPLE == 0)
        {
            launch_accumulate_radial_moments(cfg, d, mf.device);
            ++mf.state.radial_sample_count;
        }

        launch_collistream(cfg, d);
        launch_inlet_bc(cfg, d);
        launch_neumann_bc(cfg, d);

        if (update_mlups_stats(evStart, evStop, step, perf, 100))
        {
            print_mlups_stats(perf, step);
        }

        if (step % NOUTPUT == 0)
        {
            CUDA_CHECK(cudaMemset(mf.device.tke_total, 0, sizeof(real_t)));

            launch_compute_total_tke(cfg, d, mf.device.tke_total);
            launch_update_tke_average(mf.device.tke_avg, mf.device.tke_total, step, 0);

            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaMemcpy(&mf.host.tke_total,
                                  mf.device.tke_total,
                                  sizeof(real_t),
                                  cudaMemcpyDeviceToHost));

            CUDA_CHECK(cudaMemcpy(&mf.host.tke_avg,
                                  mf.device.tke_avg,
                                  sizeof(real_t),
                                  cudaMemcpyDeviceToHost));

            process_tke_sample(mf, step);

            // write_vti_step_device(step, d, h);
        }

        if (step % NPROFILE_OUTPUT == 0)
        {
            write_radial_profile_outputs(mf);
        }

        if (step % NSLICE_OUTPUT == 0)
        {
            write_midplane_velocity_profile_vti_step_device(step, d, mf.vti_dir);
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    write_radial_profile_outputs(mf);

    CUDA_CHECK(cudaEventDestroy(evStart));
    CUDA_CHECK(cudaEventDestroy(evStop));

    free_mean_fields_runtime(mf);

    free_device_memory(d);

    CUDA_CHECK(cudaDeviceReset());
    return 0;
}
