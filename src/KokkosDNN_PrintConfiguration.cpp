#include <Kokkos_Core.hpp>
#include <iostream>
#include "KokkosDNN_config.h"

namespace KokkosDNN {

void print_configuration(std::ostream& os) {
  os << "KokkosDNN " << KOKKOSDNN_VERSION_MAJOR << "."
     << KOKKOSDNN_VERSION_MINOR << "." << KOKKOSDNN_VERSION_PATCH << "\n";
#ifdef KOKKOSDNN_ENABLE_CUBLAS
  os << "  cuBLAS dispatch: ON\n";
#else
  os << "  cuBLAS dispatch: OFF (native Kokkos GEMM)\n";
#endif
#ifdef KOKKOSDNN_ENABLE_CUDNN
  os << "  cuDNN SDPA dispatch: ON\n";
#else
  os << "  cuDNN SDPA dispatch: OFF (native Kokkos attention)\n";
#endif
#ifdef KOKKOSDNN_ENABLE_HIPBLAS
  os << "  hipBLAS dispatch: ON\n";
#else
  os << "  hipBLAS dispatch: OFF (native Kokkos GEMM)\n";
#endif
#ifdef KOKKOSDNN_ENABLE_MIOPEN
  os << "  MIOpen dispatch: ON\n";
#else
  os << "  MIOpen dispatch: OFF (native Kokkos conv/attention)\n";
#endif
  os << "  Kokkos backends:";
#ifdef KOKKOS_ENABLE_SERIAL
  os << " Serial";
#endif
#ifdef KOKKOS_ENABLE_OPENMP
  os << " OpenMP";
#endif
#ifdef KOKKOS_ENABLE_CUDA
  os << " CUDA";
#endif
#ifdef KOKKOS_ENABLE_HIP
  os << " HIP";
#endif
#ifdef KOKKOS_ENABLE_SYCL
  os << " SYCL";
#endif
  os << "\n";
}

}  // namespace KokkosDNN
