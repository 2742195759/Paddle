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
#include "paddle/cinn/operator_fusion/frontend/pattern.h"

namespace cinn::fusion {

// extern template to don't allow compiler specialize the following code.

template <> StmtPattern<FrontendStage> ConvertToStmtPattern(const PatternContent<FrontendStage>& content) ;

template <>
StmtPattern<FrontendStage> RT_x_RT(const ReduceTreePattern<FrontendStage>& first,
                       const ReduceTreePattern<FrontendStage>& second) ;

template <>
StmtPattern<FrontendStage> RT_x_Trivial(const ReduceTreePattern<FrontendStage>& first,
                            const TrivialPattern<FrontendStage>& second) ;

template <>
StmtPattern<FrontendStage> Trivial_x_Reduce(const TrivialPattern<FrontendStage>& first,
                            const ReducePattern<FrontendStage>& second) ;

template <>
StmtPattern<FrontendStage> Trivial_x_Trivial(const TrivialPattern<FrontendStage>& first,
                            const TrivialPattern<FrontendStage>& second) ;

template <>
StmtPattern<FrontendStage> H_x_H(const HorizontalFusionPattern<FrontendStage>& first,
                     const HorizontalFusionPattern<FrontendStage>& second) ;

}