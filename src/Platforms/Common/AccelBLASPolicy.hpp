//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//
// File developed by: Youngjun Lee, leey@anl.gov, Argonne National Laboratory
//
// File created by: Youngjun Lee, leey@anl.gov, Argonne National Laboratory
//////////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_ACCELBLAS_POLICY_H
#define QMCPLUSPLUS_ACCELBLAS_POLICY_H

#include "PlatformKinds.hpp"

namespace qmcplusplus
{
namespace compute
{

enum class FP64EmulationMode
{
  NATIVE,
  FIXED_POINT,
};

template<PlatformKind PL>
struct BLASPolicy;

} // namespace compute
} // namespace qmcplusplus

#endif
