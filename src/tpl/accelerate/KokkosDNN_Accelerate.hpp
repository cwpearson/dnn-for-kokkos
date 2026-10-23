// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
#ifndef KOKKOSDNN_ACCELERATE_HPP
#define KOKKOSDNN_ACCELERATE_HPP

// Apple Accelerate dispatch for KokkosDNN::gemm/gemv on host execution
// spaces. Included into src/KokkosDNN_Gemm.hpp only when
// KOKKOSDNN_ENABLE_ACCELERATE is defined. Stateless (cblas_sgemm takes no
// handle); the caller's DNNHandle only records use_accelerate.
#include "tpl/accelerate/KokkosDNN_Accelerate_GemmFloat.hpp"

#endif
