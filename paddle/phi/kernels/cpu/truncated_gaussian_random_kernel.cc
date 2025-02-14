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

#include "paddle/phi/kernels/truncated_gaussian_random_kernel.h"

#include <limits>
#include <random>
#include <vector>

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

#include "paddle/fluid/framework/generator.h"

namespace phi {

template <typename T, typename Context>
void TruncatedGaussianRandomKernel(const Context& dev_ctx,
                                   const std::vector<int>& shape,
                                   float mean,
                                   float std,
                                   int seed,
                                   DataType dtype,
                                   DenseTensor* out) {
  auto tensor = out;

  T* data = dev_ctx.template Alloc<T>(tensor);

  auto normal_cdf = [](float x) {
    return (1.0 + std::erf(x / std::sqrt(2.0))) / 2.0;
  };
  float a_normal_cdf = normal_cdf((-2.0 - mean) / std);
  float b_normal_cdf = normal_cdf((2.0 - mean) / std);
  std::uniform_real_distribution<float> dist(2.0 * a_normal_cdf - 1.0,
                                             2.0 * b_normal_cdf - 1.0);
  TruncatedNormal<T> truncated_normal(mean, std);
  int64_t size = tensor->numel();

  auto engine = paddle::framework::GetCPURandomEngine(seed);
  for (int64_t i = 0; i < size; ++i) {
    data[i] = truncated_normal(dist(*engine));
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(truncated_gaussian_random,
                   CPU,
                   ALL_LAYOUT,
                   phi::TruncatedGaussianRandomKernel,
                   float) {}
