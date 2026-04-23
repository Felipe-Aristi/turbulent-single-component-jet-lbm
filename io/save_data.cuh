#ifndef SAVE_DATA_CUH
#define SAVE_DATA_CUH

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include <string>

#include "../constants.cuh"
#include "../utilities/types.cuh"
#include "../utilities/cudaUtilities.cuh"
#include "../memory.cuh"

inline std::filesystem::path default_out_dir()
{
    std::ostringstream folder_name;
    folder_name << "Re"
                << static_cast<int>(std::round(Re))
                << "_vtifiles";

    return std::filesystem::current_path() / "JET_VTK" / folder_name.str();
}

inline const char *vtk_real_type()
{
    if constexpr (std::is_same_v<real_t, float>)
        return "Float32";
    else
        return "Float64";
}

// -------------------- Base64 --------------------

inline std::string base64_encode(const unsigned char *data, std::size_t len)
{
    static constexpr char table[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789+/";

    std::string out;
    out.reserve(((len + 2) / 3) * 4);

    for (std::size_t i = 0; i < len; i += 3)
    {
        const std::uint32_t b0 = data[i];
        const std::uint32_t b1 = (i + 1 < len) ? data[i + 1] : 0;
        const std::uint32_t b2 = (i + 2 < len) ? data[i + 2] : 0;

        const std::uint32_t triple = (b0 << 16) | (b1 << 8) | b2;

        out.push_back(table[(triple >> 18) & 0x3F]);
        out.push_back(table[(triple >> 12) & 0x3F]);

        if (i + 1 < len)
            out.push_back(table[(triple >> 6) & 0x3F]);
        else
            out.push_back('=');

        if (i + 2 < len)
            out.push_back(table[triple & 0x3F]);
        else
            out.push_back('=');
    }

    return out;
}

// -------------------- Pack arrays for VTK XML binary --------------------
// VTK binary XML expects:
// [UInt64 byte_count][raw bytes...]
// and then the whole thing base64-encoded.

inline std::string encode_scalar_array_binary(const real_t *data, std::size_t nvals)
{
    const std::uint64_t nbytes =
        static_cast<std::uint64_t>(nvals) * static_cast<std::uint64_t>(sizeof(real_t));

    std::vector<unsigned char> buffer(sizeof(std::uint64_t) + static_cast<std::size_t>(nbytes));

    std::memcpy(buffer.data(), &nbytes, sizeof(std::uint64_t));
    std::memcpy(buffer.data() + sizeof(std::uint64_t), data, static_cast<std::size_t>(nbytes));

    return base64_encode(buffer.data(), buffer.size());
}

inline std::string encode_vec3_array_binary(const real_t *ux,
                                            const real_t *uy,
                                            const real_t *uz,
                                            std::size_t npts)
{
    const std::uint64_t nbytes =
        static_cast<std::uint64_t>(3) *
        static_cast<std::uint64_t>(npts) *
        static_cast<std::uint64_t>(sizeof(real_t));

    std::vector<unsigned char> buffer(sizeof(std::uint64_t) + static_cast<std::size_t>(nbytes));
    std::memcpy(buffer.data(), &nbytes, sizeof(std::uint64_t));

    real_t *payload = reinterpret_cast<real_t *>(buffer.data() + sizeof(std::uint64_t));
    for (std::size_t i = 0; i < npts; ++i)
    {
        payload[3 * i + 0] = ux[i];
        payload[3 * i + 1] = uy[i];
        payload[3 * i + 2] = uz[i];
    }

    return base64_encode(buffer.data(), buffer.size());
}

// -------------------- VTI writer --------------------

inline void write_vti(const std::filesystem::path &filename,
                      const real_t *rho,
                      const real_t *ux,
                      const real_t *uy,
                      const real_t *uz)
{
    std::ofstream out(filename, std::ios::binary);
    if (!out)
        throw std::runtime_error("Cannot open VTI file for writing: " + filename.string());

    constexpr std::size_t npts = static_cast<std::size_t>(Ncells);

    const std::string enc_rho = encode_scalar_array_binary(rho, npts);
    const std::string enc_u = encode_vec3_array_binary(ux, uy, uz, npts);

    out << "<?xml version=\"1.0\"?>\n";
    out << "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n";
    out << "  <ImageData WholeExtent=\"0 " << (NX - 1)
        << " 0 " << (NY - 1)
        << " 0 " << (NZ - 1)
        << "\" Origin=\"0 0 0\" Spacing=\"1 1 1\">\n";
    out << "    <Piece Extent=\"0 " << (NX - 1)
        << " 0 " << (NY - 1)
        << " 0 " << (NZ - 1) << "\">\n";

    out << "      <PointData Scalars=\"rho\" Vectors=\"u\">\n";

    out << "        <DataArray type=\"" << vtk_real_type()
        << "\" Name=\"rho\" format=\"binary\">\n";
    out << enc_rho << "\n";
    out << "        </DataArray>\n";

    out << "        <DataArray type=\"" << vtk_real_type()
        << "\" Name=\"u\" NumberOfComponents=\"3\" format=\"binary\">\n";
    out << enc_u << "\n";
    out << "        </DataArray>\n";

    out << "      </PointData>\n";
    out << "      <CellData>\n";
    out << "      </CellData>\n";
    out << "    </Piece>\n";
    out << "  </ImageData>\n";
    out << "</VTKFile>\n";

    if (!out)
        throw std::runtime_error("Error while finalizing VTI file: " + filename.string());
}

// Copies D->H and writes one file for "step"
inline void write_vti_step_device(int step, const LbmDevice &d, LbmHost &h)
{
    CUDA_CHECK(cudaDeviceSynchronize());
    copy_out_D2H(h, d);

    const auto out_dir = default_out_dir();
    std::filesystem::create_directories(out_dir);

    std::ostringstream name;
    name << "lbm_" << std::setw(8) << std::setfill('0') << step << ".vti";

    write_vti(out_dir / name.str(), h.rho, h.ux, h.uy, h.uz);
}

#endif