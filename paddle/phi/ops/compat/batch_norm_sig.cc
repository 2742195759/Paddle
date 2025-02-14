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

#include "paddle/phi/core/compat/op_utils.h"

namespace phi {

KernelSignature BatchNormOpArgumentMapping(const ArgumentMappingContext& ctx) {
  return KernelSignature("batch_norm",
                         {"X", "Scale", "Bias", "Mean", "Variance"},
                         {"momentum",
                          "epsilon",
                          "data_layout",
                          "is_test",
                          "use_global_stats",
                          "trainable_statistics",
                          "fuse_with_relu"},
                         {"Y",
                          "MeanOut",
                          "VarianceOut",
                          "SavedMean",
                          "SavedVariance",
                          "ReserveSpace"});
}

KernelSignature BatchNormGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature(
      "batch_norm_grad",
      {GradVarName("Y"),
       "X",
       "Scale",
       "Bias",
       "SavedMean",
       "SavedVariance",
       "ReserveSpace",
       "Mean",
       "Variance"},
      {"momentum",
       "epsilon",
       "data_layout",
       "is_test",
       "use_global_stats",
       "trainable_statistics",
       "fuse_with_relu"},
      {GradVarName("X"), GradVarName("Scale"), GradVarName("Bias")});
}

KernelSignature BatchNormGradGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature("batch_norm_grad_grad",
                         {"DDX",
                          "DDScale",
                          "DDBias",
                          "DY",
                          "X",
                          "Scale",
                          "SavedMean",
                          "SavedVariance",
                          "Mean",
                          "Variance"},
                         {"momentum",
                          "epsilon",
                          "data_layout",
                          "is_test",
                          "use_global_stats",
                          "trainable_statistics",
                          "fuse_with_relu"},
                         {"DX", "DScale", "DDY"});
}

}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(batch_norm, phi::BatchNormOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(batch_norm_grad,
                           phi::BatchNormGradOpArgumentMapping);
PD_REGISTER_ARG_MAPPING_FN(batch_norm_grad_grad,
                           phi::BatchNormGradGradOpArgumentMapping);
