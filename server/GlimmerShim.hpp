// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
// server/GlimmerShim.hpp: Glimmer header selection for the server backend.
//
// TODO(models/glimmer): a sibling workstream promotes examples/glimmer/ to
// models/glimmer (installable KokkosDNN::glimmer target, same header names
// and API — the promotion is mechanical, mirroring models/lfm2_5). Once it
// lands, KOKKOSDNN_SERVER_HAVE_GLIMMER is defined by server/CMakeLists.txt
// (KokkosDNN::glimmer target exists) and the includes below resolve to the
// promoted headers with no backend changes. Until then the backend compiles
// in stub mode (BackendGlimmer.cpp #else branch) and this header is unused.
#ifndef KOKKOSDNN_SERVER_GLIMMER_SHIM_HPP
#define KOKKOSDNN_SERVER_GLIMMER_SHIM_HPP

#include <KokkosDNN_GlimmerConfig.hpp>
#include <KokkosDNN_GlimmerModel.hpp>
#include <KokkosDNN_GlimmerTokenizer.hpp>
#include <KokkosDNN_GlimmerWeights.hpp>

#endif
