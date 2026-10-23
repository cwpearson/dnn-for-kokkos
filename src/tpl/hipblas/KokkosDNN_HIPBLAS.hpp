// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_HIPBLAS_HPP
#define KOKKOSDNN_HIPBLAS_HPP

// hipBLAS dispatch for KokkosDNN::gemm/gemv. Included into
// src/KokkosDNN_Gemm.hpp (and KokkosDNN_Handle.hpp for the handle helpers)
// only when KOKKOSDNN_ENABLE_HIPBLAS is defined. All entry points take the
// caller's cached hipblasHandle_t (owned by DNNHandle, created once per
// handle), never creating a handle per call. The stream is (re-)selected
// to the Kokkos HIP stream on every call so the TPL work stays ordered
// with the surrounding Kokkos kernels.
#include "tpl/hipblas/KokkosDNN_HIPBLAS_Handle.hpp"
#include "tpl/hipblas/KokkosDNN_HIPBLAS_Stream.hpp"
#include "tpl/hipblas/KokkosDNN_HIPBLAS_GemmFloat.hpp"
#include "tpl/hipblas/KokkosDNN_HIPBLAS_GemmBF16.hpp"

#endif
