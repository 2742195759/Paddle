// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/infermeta/unary.h"
#include "paddle/phi/kernels/empty_kernel.h"

namespace phi {
template <typename T, typename Context>
void SumRawKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const std::vector<int64_t>& dims,
                  bool keep_dim,
                  bool reduce_all,
                  DataType out_dtype,
                  DenseTensor* out);

template <typename T, typename Context>
void MeanRawKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const std::vector<int64_t>& dims,
                   bool keep_dim,
                   bool reduce_all,
                   DenseTensor* out);

template <typename T, typename Context>
void ProdRawKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const std::vector<int64_t>& dims,
                   bool keep_dim,
                   bool reduce_all,
                   DenseTensor* out);

template <typename T, typename Context>
void MaxRawKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const std::vector<int64_t>& dims,
                  bool keep_dim,
                  bool reduce_all,
                  DenseTensor* out);

template <typename T, typename Context>
void MinRawKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const std::vector<int64_t>& dims,
                  bool keep_dim,
                  bool reduce_all,
                  DenseTensor* out);

template <typename T, typename Context>
void AnyRawKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const std::vector<int64_t>& dims,
                  bool keep_dim,
                  bool reduce_all,
                  DenseTensor* out);

template <typename T, typename Context>
void AllRawKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const std::vector<int64_t>& dims,
                  bool keep_dim,
                  bool reduce_all,
                  DenseTensor* out);

template <typename T, typename Context>
void SumKernel(const Context& dev_ctx,
               const DenseTensor& x,
               const std::vector<int64_t>& dims,
               DataType out_dtype,
               bool keep_dim,
               DenseTensor* out);

template <typename T, typename Context>
void MeanKernel(const Context& dev_ctx,
                const DenseTensor& x,
                const std::vector<int64_t>& dims,
                bool keep_dim,
                DenseTensor* out);

template <typename T, typename Context>
void ProdKernel(const Context& dev_ctx,
                const DenseTensor& x,
                const std::vector<int64_t>& dims,
                bool keep_dim,
                DenseTensor* out);

template <typename T, typename Context>
void MaxKernel(const Context& dev_ctx,
               const DenseTensor& x,
               const std::vector<int64_t>& dims,
               bool keep_dim,
               DenseTensor* out);

template <typename T, typename Context>
void MinKernel(const Context& dev_ctx,
               const DenseTensor& x,
               const std::vector<int64_t>& dims,
               bool keep_dim,
               DenseTensor* out);

template <typename T, typename Context>
void AnyKernel(const Context& dev_ctx,
               const DenseTensor& x,
               const std::vector<int64_t>& dims,
               bool keep_dim,
               DenseTensor* out);

template <typename T, typename Context>
void AllKernel(const Context& dev_ctx,
               const DenseTensor& x,
               const std::vector<int64_t>& dims,
               bool keep_dim,
               DenseTensor* out);

template <typename T, typename Context>
DenseTensor Mean(const Context& dev_ctx,
                 const DenseTensor& x,
                 const std::vector<int64_t>& axis,
                 bool keep_dim) {
  DenseTensor dense_out;
  MetaTensor meta_out(&dense_out);
  SumRawInferMeta(x, axis, keep_dim, false, x.dtype(), &meta_out);
  MeanKernel<T, Context>(dev_ctx, x, axis, keep_dim, &dense_out);
  return dense_out;
}

template <typename T, typename Context>
DenseTensor Sum(const Context& dev_ctx,
                const DenseTensor& x,
                const std::vector<int64_t>& axis,
                DataType dtype,
                bool keep_dim) {
  DenseTensor dense_out;
  MetaTensor meta_out(&dense_out);
  SumInferMeta(x, axis, dtype, keep_dim, &meta_out);
  SumKernel<T, Context>(dev_ctx, x, axis, dtype, keep_dim, &dense_out);
  return dense_out;
}

}  // namespace phi
