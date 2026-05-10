#ifndef CONSTANTS_CUH
#define CONSTANTS_CUH

#include <array>
#include <cstddef>
#include "utilities/types.cuh"

// Steps
inline constexpr int NSTEP = 5000000;
inline constexpr int NOUTPUT = 2000;
inline constexpr int NSTATS_SAMPLE = 20;

// Grid
inline constexpr label_t NX = static_cast<label_t>(128);
inline constexpr label_t NZ = static_cast<label_t>(128);
inline constexpr label_t NY = static_cast<label_t>(400);
inline constexpr label_t sponge_cells = static_cast<label_t>(33);

inline constexpr label_t Ncells = NX * NY * NZ;
inline constexpr label_t NR_BINS = (NX < NZ ? NX : NZ) / static_cast<label_t>(2);
inline constexpr std::size_t NradialProfileCells = static_cast<std::size_t>(NY) * static_cast<std::size_t>(NR_BINS);

// VELOCITY SET D2Q27 definition
inline constexpr label_t Q = 27;

// Memory sizes
inline constexpr std::size_t bytesCell = std::size_t(Ncells) * sizeof(real_t);
inline constexpr std::size_t fSize = std::size_t(Ncells) * (std::size_t)Q;
inline constexpr std::size_t bytesF = fSize * sizeof(pop_t);

// Jet parameters
inline constexpr real_t jet_radius = static_cast<real_t>(5.0);
inline constexpr real_t jet_x0 = static_cast<real_t>(NX - 1) / static_cast<real_t>(2);
inline constexpr real_t jet_z0 = static_cast<real_t>(NZ - 1) / static_cast<real_t>(2);
inline constexpr real_t jet_velocity = static_cast<real_t>(0.05);

// Some useful constans
inline constexpr real_t cs2 = static_cast<real_t>(1.0 / 3.0);
inline constexpr real_t inv_cs2 = static_cast<real_t>(1) / cs2;
inline constexpr real_t cs4 = cs2 * cs2;
inline constexpr real_t inv_2cs2 = static_cast<real_t>(1) / (static_cast<real_t>(2) * cs2);
inline constexpr real_t inv_2cs4 = static_cast<real_t>(1) / (static_cast<real_t>(2) * cs4);

// Fluidparameters
inline constexpr real_t rho0 = static_cast<real_t>(1);

inline constexpr real_t Re = static_cast<real_t>(5000);
inline constexpr real_t nu = (static_cast<real_t>(2) * jet_radius * jet_velocity) / Re;

inline constexpr real_t tau = static_cast<real_t>(0.5) + nu / (cs2); // static_cast<real_t>(0.6)

inline constexpr real_t omega = static_cast<real_t>(1) / tau;

#endif
