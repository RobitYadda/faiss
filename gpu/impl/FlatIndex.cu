/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.

#include "FlatIndex.cuh"
#include "Distance.cuh"
#include "L2Norm.cuh"
#include "../utils/CopyUtils.cuh"
#include "../utils/DeviceUtils.h"
#include "../utils/Transpose.cuh"

namespace faiss { namespace gpu {

FlatIndex::FlatIndex(GpuResources* res,
                     int dim,
                     bool l2Distance,
                     GPU_DATA_TYPE useFloat16,
                     bool useFloat16Accumulator,
                     bool storeTransposed,
                     MemorySpace space) :
    resources_(res),
    dim_(dim),
    useFloat16_(useFloat16),
    useFloat16Accumulator_(useFloat16Accumulator),
    storeTransposed_(storeTransposed),
    l2Distance_(l2Distance),
    space_(space),
    num_(0),
    rawData_(space),
    error_(Faiss_Error_OK),
    max_size_(SIZE_MAX){
#ifndef FAISS_USE_FLOAT16
  FAISS_ASSERT(useFloat16_==GPU_DATA_TYPE::IFLOAT);
#endif
}

        GPU_DATA_TYPE
FlatIndex::getUseFloat16() const {
  return useFloat16_;
}

/// Returns the number of vectors we contain
int FlatIndex::getSize() const {
#ifdef FAISS_USE_FLOAT16
  if (useFloat16_==GPU_DATA_TYPE::IFLOAT16) {
    return vectorsHalf_.getSize(0);
  }
if (useFloat16_==GPU_DATA_TYPE::IINT8) {
    return vectorsInt8_.getSize(0);
}
#endif

  return vectors_.getSize(0);
}

int FlatIndex::getDim() const {
#ifdef FAISS_USE_FLOAT16
  if (useFloat16_==GPU_DATA_TYPE::IFLOAT16) {
    return vectorsHalf_.getSize(1);
  }
    if (useFloat16_==GPU_DATA_TYPE::IINT8) {
        return vectorsInt8_.getSize(1);
    }
#endif

  return vectors_.getSize(1);
}

ErrorTypes FlatIndex::error()
{
    auto ret = error_;
    error_ = Faiss_Error_OK;
    return ret;
}

void FlatIndex::setMaxSize(size_t new_size)
{
    max_size_ = new_size;
    if (useFloat16_ == GPU_DATA_TYPE::IFLOAT)
    {
        new_size *= dim_ * sizeof(float);
    } else if (useFloat16_ == GPU_DATA_TYPE::IFLOAT16)
    {
        new_size *= dim_ * sizeof(half);
    } else
    {
        new_size *= dim_ * sizeof(int8_t);
    }
    rawData_.set_max_size(new_size);
}

size_t FlatIndex::getMaxSize() const
{
    return max_size_;
}

void
FlatIndex::reserve(size_t numVecs, cudaStream_t stream) {
    if (useFloat16_==GPU_DATA_TYPE::IINT8) {
        rawData_.reserve(numVecs * dim_ * sizeof(int8_t), stream);//TODO:int8 //mochang
        error_ = rawData_.error();
    }else if (useFloat16_==GPU_DATA_TYPE::IFLOAT16) {
#ifdef FAISS_USE_FLOAT16
    rawData_.reserve(numVecs * dim_ * sizeof(half), stream);

#endif
  } else {
    rawData_.reserve(numVecs * dim_ * sizeof(float), stream);
  }
}

Tensor<float, 2, true>&
FlatIndex::getVectorsFloat32Ref() {
  return vectors_;
}

#ifdef FAISS_USE_FLOAT16
Tensor<half, 2, true>&
FlatIndex::getVectorsFloat16Ref() {
  return vectorsHalf_;
}

Tensor<int8_t , 2, true>&
FlatIndex::getVectorsInt8Ref() {
  return vectorsInt8_;
}

Tensor<float, 1, true>&
FlatIndex::getNormsInt8Ref()
{
    return normsInt8_;
}
#endif

DeviceTensor<float, 2, true>
FlatIndex::getVectorsFloat32Copy(cudaStream_t stream) {
  return getVectorsFloat32Copy(0, num_, stream);
}

DeviceTensor<float, 2, true>
FlatIndex::getVectorsFloat32Copy(int from, int num, cudaStream_t stream) {
  DeviceTensor<float, 2, true> vecFloat32({num, dim_}, space_);

  if(useFloat16_==GPU_DATA_TYPE::IINT8){
      runConvertToFloat32(vecFloat32.data(),
                          vectorsInt8_[from].data(),
                          num * dim_, stream);
  } else if (useFloat16_==GPU_DATA_TYPE::IFLOAT16) {
#ifdef FAISS_USE_FLOAT16
    runConvertToFloat32(vecFloat32.data(),
                        vectorsHalf_[from].data(),
                        num * dim_, stream);
#endif
  } else {
    vectors_.copyTo(vecFloat32, stream);
  }

  return vecFloat32;
}

Tensor<float, 1, true> &FlatIndex::getQueryNormsRef()
{
    return queryNorms_;
}

void
FlatIndex::query(Tensor<float, 2, true>& input,
                 int k,
                 Tensor<float, 2, true>& outDistances,
                 Tensor<int, 2, true>& outIndices,
                 bool exactDistance) {
  auto stream = resources_->getDefaultStreamCurrentDevice();
  auto& mem = resources_->getMemoryManagerCurrentDevice();

    if(useFloat16_==GPU_DATA_TYPE::IINT8){

//        if (use_int8_norms_)
        {
            DeviceTensor<float, 1, true> queryNorms({(int) input.getSize(0)});
            runL2Norm(input, queryNorms, true, input.getSize(0), resources_, stream);
            queryNorms_ = std::move(queryNorms);
        }

        auto inputInt8 = toInt8<2>(resources_, stream, input);

//        DeviceTensor<float, 2, true> outDistancesFloat(
//                mem, {outDistances.getSize(0), outDistances.getSize(1)}, stream);

        query(inputInt8, k, outDistances, outIndices, exactDistance);

//        if (exactDistance) {//TODO:int8 /mochang?
//            // Convert outDistances back
//            fromHalf<2>(stream, outDistancesHalf, outDistances);
//        }
    } else if (useFloat16_==GPU_DATA_TYPE::IFLOAT16) {
    // We need to convert to float16
#ifdef FAISS_USE_FLOAT16
    auto inputHalf = toHalf<2>(resources_, stream, input);

    DeviceTensor<half, 2, true> outDistancesHalf(
      mem, {outDistances.getSize(0), outDistances.getSize(1)}, stream);

    query(inputHalf, k, outDistancesHalf, outIndices, exactDistance);

    if (exactDistance) {
      // Convert outDistances back
      fromHalf<2>(stream, outDistancesHalf, outDistances);
    }
#endif
  } else {
    if (l2Distance_) {
      runL2Distance(resources_,
                    vectors_,
                    storeTransposed_ ? &vectorsTransposed_ : nullptr,
                    &norms_,
                    input,
                    k,
                    outDistances,
                    outIndices,
                    // FIXME
                    !exactDistance);
    } else {
      runIPDistance(resources_,
                    vectors_,
                    storeTransposed_ ? &vectorsTransposed_ : nullptr,
                    input,
                    k,
                    outDistances,
                    outIndices);
    }
  }
}

#ifdef FAISS_USE_FLOAT16
void
FlatIndex::query(Tensor<half, 2, true>& input,
                 int k,
                 Tensor<half, 2, true>& outDistances,
                 Tensor<int, 2, true>& outIndices,
                 bool exactDistance) {
  FAISS_ASSERT(useFloat16_);

  if (l2Distance_) {
    runL2Distance(resources_,
                  vectorsHalf_,
                  storeTransposed_ ? &vectorsHalfTransposed_ : nullptr,
                  &normsHalf_,
                  input,
                  k,
                  outDistances,
                  outIndices,
                  useFloat16Accumulator_,
                  // FIXME
                  !exactDistance);
  } else {
    runIPDistance(resources_,
                  vectorsHalf_,
                  storeTransposed_ ? &vectorsHalfTransposed_ : nullptr,
                  input,
                  k,
                  outDistances,
                  outIndices,
                  useFloat16Accumulator_);
  }
}
void
FlatIndex::query(Tensor<int8_t, 2, true>& input,
                 int k,
                 Tensor<float, 2, true>& outDistances,
                 Tensor<int, 2, true>& outIndices,
                 bool exactDistance) {
    FAISS_ASSERT(!l2Distance_);

    if (l2Distance_) {
//        runL2Distance(resources_,
//                      vectorsHalf_,
//                      storeTransposed_ ? &vectorsHalfTransposed_ : nullptr,
//                      &normsHalf_,
//                      input,
//                      k,
//                      outDistances,
//                      outIndices,
//                      useFloat16Accumulator_,
//                // FIXME
//                      !exactDistance);
    } else {
        runIPDistance(resources_,
                      vectorsInt8_,
                      storeTransposed_ ? &vectorsInt8Transposed_ : nullptr,
                      input,
                      k,
                      outDistances,
                      outIndices,
                      normsInt8_,
                      queryNorms_,
                      use_int8_norms_,
                      useFloat16Accumulator_);
    }
}
#endif

void FlatIndex::del(const long inputIndex, cudaStream_t stream){
    FAISS_ASSERT(!l2Distance_);
    if (useFloat16_==GPU_DATA_TYPE::IFLOAT16)
    {
        FAISS_THROW_MSG ("del not implemented for useFloat16_");
    }
    if(useFloat16_==GPU_DATA_TYPE::IFLOAT){
        int numVecs = 1;
        if(num_!=1){
            //不释放以前申请的
            CUDA_VERIFY(cudaMemcpy(
                    ((char*)rawData_.data())+inputIndex*dim_*sizeof(float),
                    ((char*)rawData_.data())+dim_*(num_-1)*sizeof(float),
                    numVecs*dim_*sizeof(float), //In bytes
                    cudaMemcpyDeviceToDevice
            ));
        }

        num_-=1;
        rawData_.resize(num_ * dim_ * sizeof(float),stream);

        {
            DeviceTensor<float, 2, true> vectors(
                    (float*) rawData_.data(), {(int) num_, dim_}, space_);
            vectors_ = std::move(vectors);
        }

        if (storeTransposed_) {
            {
                vectorsTransposed_ =
                        std::move(DeviceTensor<float, 2, true>({dim_, (int) num_}, space_));
                runTransposeAny(vectors_, 0, 1, vectorsTransposed_, stream);
            }
        }
    }else if(useFloat16_==GPU_DATA_TYPE::IINT8){
        int numVecs = 1;
        if(num_!=1){
            //不释放以前申请的
            CUDA_VERIFY(cudaMemcpy(
                    ((char*)rawData_.data())+inputIndex*dim_*sizeof(int8_t),
                    ((char*)rawData_.data())+dim_*(num_-1)*sizeof(int8_t),
                    numVecs*dim_*sizeof(int8_t), //In bytes
                    cudaMemcpyDeviceToDevice
            ));
//            if (use_int8_norms_)
            {
                CUDA_VERIFY(cudaMemcpy(
                        ((char*) normsInt8_.data()) + inputIndex * sizeof(float),
                        ((char*) normsInt8_.data()) + (num_ - 1) * sizeof(float),
                        numVecs * sizeof(float), //In bytes
                        cudaMemcpyDeviceToDevice
                ));
                normsInt8_.setSize(0, num_ - 1);
            }
        }

        num_-=1;
        rawData_.resize(num_ * dim_ * sizeof(int8_t),stream);
        error_ = rawData_.error();


        {
            DeviceTensor<int8_t , 2, true> vectors(
                    (int8_t*) rawData_.data(), {(int) num_, dim_}, space_);
            vectorsInt8_ = std::move(vectors);
        }

        if (storeTransposed_) {
            {
                vectorsInt8Transposed_ =
                        std::move(DeviceTensor<int8_t, 2, true>({dim_, (int) num_}, space_));
                runTransposeAny(vectorsInt8_, 0, 1, vectorsInt8Transposed_, stream);
            }
        }

    }

}

void
FlatIndex::add(const float* data, int numVecs, cudaStream_t stream) {
    if (numVecs == 0) {
      return;
    }
    if (useFloat16_==GPU_DATA_TYPE::IINT8){
        //fen pian
        // Make sure that `data` is on our device; we'll run the
        // conversion on our device
        const int MAX_ADD_BATCH = 1024*1024;
        int batches = (numVecs/MAX_ADD_BATCH);
        int left = (numVecs % MAX_ADD_BATCH);
        int iBatches = 0;
        unsigned long long num_count = num_;
        for (; iBatches < batches; ++iBatches) {
            auto devData = toDevice<float, 2>(resources_,
                                              getCurrentDevice(),
                                              ((float*) data+ iBatches*MAX_ADD_BATCH*dim_),
                                              stream,
                                              {MAX_ADD_BATCH, dim_});

//            if (use_int8_norms_)
            {
                DeviceTensor<float, 1, true> normsInt8({(int) num_count + MAX_ADD_BATCH});
                if (normsInt8_.getSize(0) != 0)
                    CUDA_VERIFY(cudaMemcpy(normsInt8.data(), normsInt8_.data(), normsInt8_.getSize(0) * sizeof(float), cudaMemcpyDefault));
                runL2Norm(devData, normsInt8, true, MAX_ADD_BATCH, resources_, stream);
                normsInt8_ = std::move(normsInt8);
                num_count += MAX_ADD_BATCH;
            }

            auto devDataInt8 = toInt8<2>(resources_, stream, devData);

            rawData_.append((char*) devDataInt8.data(),
                            devDataInt8.getSizeInBytes(),
                            stream,
                            true /* reserve exactly */);
            error_ = rawData_.error();
            if (error_ != Faiss_Error_OK)
            {
                return;
            }
        }
        if(left>0) {
            auto devData = toDevice<float, 2>(resources_,
                                              getCurrentDevice(),
                                              ((float*) data+ iBatches*MAX_ADD_BATCH*dim_),
                                              stream,
                                              {left, dim_});

//            if (use_int8_norms_)
            {
                DeviceTensor<float, 1, true> normsInt8({(int) num_count + left});
                if (normsInt8_.getSize(0) != 0)
                    CUDA_VERIFY(cudaMemcpy(normsInt8.data(), normsInt8_.data(), normsInt8_.getSize(0) * sizeof(float), cudaMemcpyDefault));
                runL2Norm(devData, normsInt8, true, left, resources_, stream);
                normsInt8_ = std::move(normsInt8);
            }

            auto devDataInt8 = toInt8<2>(resources_, stream, devData);

            rawData_.append((char*) devDataInt8.data(),
                            devDataInt8.getSizeInBytes(),
                            stream,
                            true /* reserve exactly */);
            error_ = rawData_.error();
            if (error_ != Faiss_Error_OK)
            {
                return;
            }
        }
    } else if (useFloat16_==GPU_DATA_TYPE::IFLOAT16) {
#ifdef FAISS_USE_FLOAT16
    // Make sure that `data` is on our device; we'll run the
    // conversion on our device

        auto devData = toDevice<float, 2>(resources_,
                                          getCurrentDevice(),
                                          (float*) data,
                                          stream,
                                          {numVecs, dim_});

        auto devDataHalf = toHalf<2>(resources_, stream, devData);

        rawData_.append((char*) devDataHalf.data(),
                        devDataHalf.getSizeInBytes(),
                        stream,
                        true /* reserve exactly */);
#endif
  } else {
    rawData_.append((char*) data,
                    (size_t) dim_ * numVecs * sizeof(float),
                    stream,
                    true /* reserve exactly */);
  }

  num_ += numVecs;

    if (useFloat16_==GPU_DATA_TYPE::IINT8){
        DeviceTensor<int8_t , 2, true> vectorsInt8(
                (int8_t*) rawData_.data(), {(int) num_, dim_}, space_);
        vectorsInt8_ = std::move(vectorsInt8);
    } else if (useFloat16_==GPU_DATA_TYPE::IFLOAT16) {
#ifdef FAISS_USE_FLOAT16
    DeviceTensor<half, 2, true> vectorsHalf(
      (half*) rawData_.data(), {(int) num_, dim_}, space_);
    vectorsHalf_ = std::move(vectorsHalf);
#endif
  } else {
    DeviceTensor<float, 2, true> vectors(
      (float*) rawData_.data(), {(int) num_, dim_}, space_);
    vectors_ = std::move(vectors);
  }

  if (storeTransposed_) {
      if (useFloat16_==GPU_DATA_TYPE::IINT8){
          vectorsInt8Transposed_ =
                  std::move(DeviceTensor<int8_t , 2, true>({dim_, (int) num_}, space_));
          runTransposeAny(vectorsInt8_, 0, 1, vectorsInt8Transposed_, stream);
      } else if (useFloat16_==GPU_DATA_TYPE::IFLOAT16) {
#ifdef FAISS_USE_FLOAT16
      vectorsHalfTransposed_ =
        std::move(DeviceTensor<half, 2, true>({dim_, (int) num_}, space_));
      runTransposeAny(vectorsHalf_, 0, 1, vectorsHalfTransposed_, stream);
#endif
    } else {
      vectorsTransposed_ =
        std::move(DeviceTensor<float, 2, true>({dim_, (int) num_}, space_));
      runTransposeAny(vectors_, 0, 1, vectorsTransposed_, stream);
    }
  }

  if (l2Distance_) {
    if (useFloat16_==GPU_DATA_TYPE::IFLOAT16) {
#ifdef FAISS_USE_FLOAT16
      DeviceTensor<half, 1, true> normsHalf({(int) num_}, space_);
      runL2Norm(vectorsHalf_, normsHalf, true, stream);
      normsHalf_ = std::move(normsHalf);
#endif
    } else {
      DeviceTensor<float, 1, true> norms({(int) num_}, space_);
      runL2Norm(vectors_, norms, true, stream);
      norms_ = std::move(norms);
    }
  }
}

void
FlatIndex::reset() {
  rawData_.clear();
  vectors_ = std::move(DeviceTensor<float, 2, true>());
    vectorsHalf_ = std::move(DeviceTensor<half, 2, true>());
    vectorsInt8_ = std::move(DeviceTensor<int8_t , 2, true>());
  norms_ = std::move(DeviceTensor<float, 1, true>());
  num_ = 0;
}

} }
