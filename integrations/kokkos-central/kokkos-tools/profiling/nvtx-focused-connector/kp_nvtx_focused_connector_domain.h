//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#ifndef KOKKOSP_KERNEL_NVTX_CONNECTOR_H
#define KOKKOSP_KERNEL_NVTX_CONNECTOR_H

#include <stdio.h>
#include <sys/time.h>
#include <cstring>

#include "nvtx3/nvToolsExt.h"

namespace KokkosTools {
namespace NVTXFocusedConnector {

enum KernelExecutionType {
  PARALLEL_FOR    = 0,
  PARALLEL_REDUCE = 1,
  PARALLEL_SCAN   = 2
};

class KernelNVTXFocusedConnectorInfo {
 public:
  KernelNVTXFocusedConnectorInfo(std::string kName,
                                 KernelExecutionType kernelType) {
    domainNameHandle = kName;
    char* domainName = (char*)malloc(sizeof(char*) * (32 + kName.size()));

    if (kernelType == PARALLEL_FOR) {
      sprintf(domainName, "ParallelFor.%s", kName.c_str());
    } else if (kernelType == PARALLEL_REDUCE) {
      sprintf(domainName, "ParallelReduce.%s", kName.c_str());
    } else if (kernelType == PARALLEL_SCAN) {
      sprintf(domainName, "ParallelScan.%s", kName.c_str());
    } else {
      sprintf(domainName, "Kernel.%s", kName.c_str());
    }

    domain       = nvtxDomainCreateA(domainName);
    currentRange = 0;
  }

  nvtxRangeId_t startRange() {
    nvtxEventAttributes_t eventAttrib = {};
    eventAttrib.version               = NVTX_VERSION;
    eventAttrib.size                  = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.messageType           = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.message.ascii         = "my range";
    currentRange = nvtxDomainRangeStartEx(domain, &eventAttrib);
    return currentRange;
  }

  nvtxRangeId_t getCurrentRange() { return currentRange; }

  void endRange() { nvtxDomainRangeEnd(domain, currentRange); }

  nvtxDomainHandle_t getDomain() { return domain; }

  std::string getDomainNameHandle() { return domainNameHandle; }

  ~KernelNVTXFocusedConnectorInfo() { nvtxDomainDestroy(domain); }

 private:
  std::string domainNameHandle;
  nvtxRangeId_t currentRange;
  nvtxDomainHandle_t domain;
};

#endif
}
}  // KokkosTools::NVTXFocusedConnector
