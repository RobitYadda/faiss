/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once


#include <iostream>

#include <thrust/inner_product.h>
#include <thrust/execution_policy.h>

#include "../utils/Float16.cuh"
#include "../utils/Tensor.cuh"

namespace faiss { namespace gpu {

void runL2Norm(Tensor<float, 2, true>& input,
               Tensor<float, 1, true>& output,
               bool normSquared,
               cudaStream_t stream);

#ifdef FAISS_USE_FLOAT16
void runL2Norm(Tensor<half, 2, true>& input,
               Tensor<half, 1, true>& output,
               bool normSquared,
               cudaStream_t stream);

void runL2Norm(Tensor<float, 2, true>& input, Tensor<float, 1, true>& output, bool normSquard, int numVecs, GpuResources * const resources);

#endif

} } // namespace
