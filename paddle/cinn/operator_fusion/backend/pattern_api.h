// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/cinn/operator_fusion/pattern.h"
#include "paddle/cinn/operator_fusion/pattern_api.h"
#include "paddle/cinn/operator_fusion/backend/pattern.h"

namespace cinn::fusion {

// extern template to don't allow compiler specialize the following code.

template <> StmtPattern<BackendStage> ConvertToStmtPattern(const PatternContent<BackendStage>& content) ;

template <>
StmtPattern<BackendStage> RT_x_RT(const ReduceTreePattern<BackendStage>& first,
                       const ReduceTreePattern<BackendStage>& second) ;

template <>
StmtPattern<BackendStage> RT_x_Trivial(const ReduceTreePattern<BackendStage>& first,
                            const TrivialPattern<BackendStage>& second) ;

template <>
StmtPattern<BackendStage> Trivial_x_Reduce(const TrivialPattern<BackendStage>& first,
                            const ReducePattern<BackendStage>& second) ;

template <>
StmtPattern<BackendStage> Trivial_x_Trivial(const TrivialPattern<BackendStage>& first,
                            const TrivialPattern<BackendStage>& second) ;

template <>
StmtPattern<BackendStage> H_x_H(const HorizontalFusionPattern<BackendStage>& first,
                     const HorizontalFusionPattern<BackendStage>& second) ;

std::vector<ir::Expr> GetExprFromPattern (const StmtPattern<BackendStage>& pattern);

}
