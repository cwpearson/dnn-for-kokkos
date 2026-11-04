# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright Contributors to the KokkosDNN project
# models/common: shared model-support utilities (INTERFACE, header-only).
# KokkosDNN_Common.hpp umbrella + KokkosDNN_Detokenizer.hpp (streaming
# detokenizer with stop-string handling) + KokkosDNN_Usage.hpp (usage
# counters / finish reasons) + KokkosDNN_Generate.hpp (shared
# prefill+decode driver) + KokkosDNN_SessionTypes.hpp (shared turn
# vocabulary) + KokkosDNN_PrefixCacheCore.hpp (shared longest-prefix
# lookup) + KokkosDNN_ModelConcepts.hpp (generic Config/Model/Session/
# PrefixCache concepts). Future home of further code extracted from
# models/*: minimal JSON helpers, BPE tokenizers, chat templates,
# samplers. See README.md.
# Every model lib links KokkosDNN::common; nothing here links a model lib.

kokkosdnn_add_header_only_model_library(kokkosdnn_common
  EXPORT_NAME common
  LINK_LIBRARIES KokkosDNN::kokkosdnn
)

install(FILES KokkosDNN_Common.hpp KokkosDNN_Detokenizer.hpp KokkosDNN_Usage.hpp
  KokkosDNN_Generate.hpp KokkosDNN_SessionTypes.hpp KokkosDNN_PrefixCacheCore.hpp
  KokkosDNN_ModelConcepts.hpp KokkosDNN_JsonLite.hpp KokkosDNN_BpeCore.hpp
  KokkosDNN_ChatStop.hpp
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

if(KokkosDNN_ENABLE_TESTS)
  add_subdirectory(tests)
endif()
