/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

namespace faiss { namespace gpu {

        enum GPU_DATA_TYPE{
            IFLOAT,
            IFLOAT16,
            IINT8
        };

/// How user vector index data is stored on the GPU
enum IndicesOptions {
  /// The user indices are only stored on the CPU; the GPU returns
  /// (inverted list, offset) to the CPU which is then translated to
  /// the real user index.
  INDICES_CPU = 0,
  /// The indices are not stored at all, on either the CPU or
  /// GPU. Only (inverted list, offset) is returned to the user as the
  /// index.
  INDICES_IVF = 1,
  /// Indices are stored as 32 bit integers on the GPU, but returned
  /// as 64 bit integers
  INDICES_32_BIT = 2,
  /// Indices are stored as 64 bit integers on the GPU
  INDICES_64_BIT = 3,
};

} } // namespace
