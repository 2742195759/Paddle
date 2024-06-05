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

#include <algorithm>
#include <atomic>
#include <memory>
#include <optional>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include "glog/logging.h"

#include "paddle/cinn/adt/adt.h"
#include "paddle/cinn/hlir/framework/op.h"
#include "paddle/cinn/operator_fusion/pattern.h"
#include "paddle/cinn/operator_fusion/utils.h"

// This file is the protocol of the pattern fuser. Please implement
// ConvertToStmtPattern and MergePatternImpl in the specializations.

namespace cinn::fusion {

StmtPattern ConvertToStmtPattern(const PatternContent& content) {
  const auto& kind = GetOpPatternKind(content.op);
  if (kind == hlir::framework::kReduction) {
    auto result =
        ReducePattern({content.op}, std::make_shared<FusionTracker>());
    result.tracker_->append(std::make_shared<InitPatternInstr>(content.op));
    return result;
  } else if (kind == hlir::framework::kElementWise ||
             kind == hlir::framework::kBroadcast ||
             kind == hlir::framework::kInjective) {
    auto result = TrivialPattern(
        {content.op}, content.op, std::make_shared<FusionTracker>());
    result.tracker_->append(std::make_shared<InitPatternInstr>(content.op));
    return result;
  } else {
    auto result =
        UnsupportPattern({content.op}, std::make_shared<FusionTracker>());
    result.tracker_->append(std::make_shared<InitPatternInstr>(content.op));
    return result;
  }
}

// Trivial x other

StmtPattern MergePatternImpl(const TrivialPattern& first,
                             const TrivialPattern& second) {
  const auto& contents =
      UniqueConcatVector(GetOpsInPattern(first), GetOpsInPattern(second));
  return TrivialPattern(
      contents,
      second.sink_op(),
      std::make_shared<FusionTracker>(first.tracker_, second.tracker_));
}

StmtPattern MergePatternImpl(const TrivialPattern& first,
                             const ReducePattern& second) {
  const auto& contents =
      UniqueConcatVector(GetOpsInPattern(first), GetOpsInPattern(second));
  return ReducePattern(
      contents,
      std::make_shared<FusionTracker>(first.tracker_, second.tracker_));
}

template <typename A, typename B>
B FusePatternIfConnected(A up_pattern,
                         B down_pattern,
                         std::vector<pir::Operation*> connect_ops) {
  if (AnyTargetInCandidate(connect_ops, down_pattern.ops())) {
    return std::get<B>(MergePatternImpl(up_pattern, down_pattern));
  } else {
    return down_pattern;
  }
}

StmtPattern MergePatternImpl(const TrivialPattern& first,
                             const ReduceTreePattern& second) {
  auto connect_ops = FindDownstreamOps(first.sink_op());

  auto old_childs = second.childs();
  std::vector<ReduceTreePattern> new_childs;
  for (const auto& old_child : old_childs) {
    new_childs.emplace_back(
        FusePatternIfConnected(first, old_child, connect_ops));
  }

  return ReduceTreePattern(
      new_childs,
      FusePatternIfConnected(first, second.GetRootPattern(), connect_ops),
      std::make_shared<FusionTracker>(first.tracker_, second.tracker_));
}

StmtPattern MergePatternImpl(const TrivialPattern& first,
                             const ReduceTreePlusTrivialPattern& second) {
  auto connect_ops = FindDownstreamOps(first.sink_op());
  return ReduceTreePlusTrivialPattern(
      FusePatternIfConnected(first, second.tree, connect_ops),
      FusePatternIfConnected(first, second.sink_trivial, connect_ops),
      std::make_shared<FusionTracker>(first.tracker_, second.tracker_));
}

StmtPattern MergePatternImpl(const TrivialPattern& first,
                             const AnchorPattern& second) {
  return AnchorPattern(
      UniqueConcatVector(GetOpsInPattern(first), GetOpsInPattern(second)),
      second.anchor(),
      second.anchor_state,
      std::make_shared<FusionTracker>(first.tracker_, second.tracker_));
}

// RR & RT

int InsertDownstreamIntoTree(const ReduceTreePattern& upstream,
                             ReduceTreePattern& downstream) {  // NOLINT
  if (IsDirectUpstream(upstream.GetRootPattern().GetReduceOp(),
                       downstream.GetRootPattern().GetReduceOp())) {
    downstream.InsertChild(upstream);
    return 1;
  }
  int insert_num = 0;
  for (auto& child : downstream.childs()) {
    insert_num += InsertDownstreamIntoTree(upstream, child);
  }
  return insert_num;
}

StmtPattern MergePatternImpl(const ReduceTreePattern& upstream,
                             const ReduceTreePattern& downstream) {
  ReduceTreePattern result = ReduceTreePattern(
      downstream.childs(),
      downstream.GetRootPattern(),
      std::make_shared<FusionTracker>(upstream.tracker_,
                                      downstream.tracker_));  // copy first.
  int insert_num = InsertDownstreamIntoTree(upstream, result);
  CHECK(insert_num == 1) << "Must insert only once, but insert " << insert_num;
  return result;
}

StmtPattern MergePatternImpl(const ReduceTreePattern& first,
                             const TrivialPattern& second) {
  return ReduceTreePlusTrivialPattern(
      first,
      second,
      std::make_shared<FusionTracker>(first.tracker_, second.tracker_));
}

// Anchor Fusion
ExprPromise InitExprPromiseImpl(const TrivialPattern& pattern,
                                pir::Value anchor) {
  return ExprPromise(anchor);
}

ExprPromise InitExprPromiseImpl(const ReducePattern& pattern,
                                pir::Value anchor) {
  return ExprPromise(anchor);
}

ExprPromise InitExprPromiseImpl(const ReduceTreePattern& pattern,
                                pir::Value anchor) {
  return InitExprPromiseImpl(pattern.GetRootPattern(), anchor);
}

template <typename PATTERN>
ExprPromise InitExprPromiseImpl(const PATTERN& pattern, pir::Value anchor) {
  PADDLE_THROW("Can not Init ExprPromise");
}

ExprPromise InitExprPromise(const StmtPattern& pattern, pir::Value anchor) {
  return std::visit(
      [anchor](const auto& arg) { return InitExprPromiseImpl(arg, anchor); },
      pattern.variant());
}

StmtPattern MergePatternImpl(const AnchorPattern& source,
                             const AnchorPattern& dest) {
  const auto& contents =
      UniqueConcatVector(GetOpsInPattern(source), GetOpsInPattern(dest));
  return AnchorPattern(
      contents,
      source.anchor(),
      AnchorState({}),
      std::make_shared<FusionTracker>(source.tracker_, dest.tracker_));
}

TrivialPattern RecoverAnchorPatternToTrivial(
    const AnchorPattern& anchor_pattern) {
  PADDLE_ENFORCE_EQ(anchor_pattern.anchor_state.promise.size(),
                    1,
                    phi::errors::PreconditionNotMet(
                        "Can only recover AnchorPattern whose anchor_state "
                        "size is 1 (exact %d)",
                        anchor_pattern.anchor_state.promise.size()));

  return TrivialPattern(anchor_pattern.ops(),
                        anchor_pattern.anchor().defining_op(),
                        anchor_pattern.tracker_);
}

AnchorState GetAnchorState(const AnchorPattern& pattern) {
  return pattern.anchor_state;
}

AnchorState ApplyAnchorTransformRoute(const AnchorState& anchor_state,
                                      const AnchorTransformRoute& route) {
  AnchorState result = anchor_state;
  for (auto promise : result.promise) {
    promise.update(route);
  }
  return result;
}

// Horizontal

using LoopFramework = std::vector<symbol::DimExpr>;

// std::optional({}) means not sure.
// std::optional will cause SegmentFault, TODO: fix this.
using MaybeLoopFramework = LoopFramework;

MaybeLoopFramework GetLoopFramework(const StmtPattern& pattern);

static MaybeLoopFramework SqueezeLoopFramework(
    const MaybeLoopFramework& loop_framework) {
  MaybeLoopFramework result;
  for (int i = 0; i < loop_framework.size(); i++) {
    if (loop_framework[i] == 1) {
      continue;  // skip 1
    } else {
      result.push_back(loop_framework[i]);
    }
  }
  return result;
}

bool IsLoopFrameworkEqual(const StmtPattern& lhs, const StmtPattern& rhs) {
  auto lhs_loop = GetLoopFramework(lhs);
  auto rhs_loop = GetLoopFramework(rhs);
  VLOG(4) << "lhs loop range is:" << utils::Join(lhs_loop, ",");
  VLOG(4) << "rhs loop range is:" << utils::Join(rhs_loop, ",");
  return SqueezeLoopFramework(lhs_loop) == SqueezeLoopFramework(rhs_loop);
}

struct LoopFrameworkVisitor {
  MaybeLoopFramework operator()(const ReducePattern& pattern) {
    pir::Operation* reduce_op = pattern.GetReduceOp();
    const auto& flatten_loops = GetDimExprsFromValue(reduce_op->result(0));
    const auto& reduce_axes = GetReduceAxisIdx(reduce_op);
    const auto& reduce_loops = GatherVector(
        GetDimExprsFromValue(reduce_op->operand(0).source()), reduce_axes);
    return ConcatVector(flatten_loops, reduce_loops);
  }

  MaybeLoopFramework operator()(const ReduceTreePattern& pattern) {
    return GetLoopFramework(StmtPattern(pattern.GetRootPattern()));
  }

  MaybeLoopFramework operator()(const TrivialPattern& pattern) {
    pir::Operation* t_op = pattern.sink_op();
    const auto& exprs = GetDimExprsFromValue(t_op->result(0));
    return exprs;
  }

  MaybeLoopFramework operator()(const HorizontalFusionPattern& pattern) {
    // Horizontal Fusion must have the same loop framework.
    VLOG(4) << "Get horizontal fusion pattern for loop framework.";
    const auto& base_exprs =
        GetLoopFramework(StmtPattern(pattern.padding_patterns_.back().pattern));
    const auto& padding_vector = pattern.padding_patterns_.back().padding_pos;
    std::vector<symbol::DimExpr> exprs(
        base_exprs.size() + padding_vector.size(), 1);
    int pointer = 0;
    for (int i = 0; i < exprs.size(); i++) {
      if (std::find(padding_vector.begin(), padding_vector.end(), i) ==
          padding_vector.end()) {
        exprs[i] = base_exprs[pointer++];
      }
    }
    return exprs;
  }

  MaybeLoopFramework operator()(const ReduceTreePlusTrivialPattern& pattern) {
    const auto& sink_trivial = pattern.sink_trivial;
    const auto& trivial_loop =
        GetLoopFramework(StmtPattern(pattern.sink_trivial));
    if (pattern.fake_reduce_iter_idx.empty()) {
      // we add reduce loop to the end;
      int reduce_axes_len =
          GetReduceAxisIdx(pattern.tree.GetRootPattern().GetReduceOp()).size();
      const auto& reduce_loop =
          GetLoopFramework(StmtPattern(pattern.tree.GetRootPattern()));
      return ConcatVector(
          trivial_loop,
          SliceVector(reduce_loop, -reduce_axes_len, reduce_loop.size()));
    } else {
      // we always put fake into the end to make the loop framework consistent.
      const auto& non_fake = GatherVector(
          trivial_loop,
          ExcludeIndex(trivial_loop.size(), pattern.fake_reduce_iter_idx));
      const auto& fake =
          GatherVector(trivial_loop, pattern.fake_reduce_iter_idx);
      return ConcatVector(non_fake, fake);
    }
  }

  MaybeLoopFramework operator()(const UnsupportPattern& pattern) {
    PADDLE_ENFORCE(false, "Not support GetLoopRange.");
  }

  MaybeLoopFramework operator()(const AnchorPattern& pattern) {
    const auto& exprs = GetDimExprsFromValue(pattern.anchor());
    return exprs;
  }
};

MaybeLoopFramework GetLoopFramework(const StmtPattern& pattern) {
  return std::visit(LoopFrameworkVisitor(), pattern.variant());
}

inline auto GetPaddingVector(const MaybeLoopFramework& first,
                             const MaybeLoopFramework& second) {
  // two pointer to get the padding body.
  std::vector<int> padding_f;
  std::vector<int> padding_s;
  VLOG(4) << "GetPaddingVector for: " << utils::Join(first, ",") << " vs "
          << utils::Join(second, ",");

  std::function<void(int, int, int)> RecursivePadding =
      [&first, &second, &padding_f, &padding_s, &RecursivePadding](
          int pf, int ps, int padding_size) {
        if (pf == first.size() && ps == second.size()) {
          return;
        } else if (pf == first.size()) {
          PADDLE_ENFORCE(second[ps] == 1, "second[ps] must be '1' to padding.");
          padding_f.push_back(padding_size);
          RecursivePadding(pf, ps + 1, padding_size + 1);
        } else if (ps == second.size()) {
          PADDLE_ENFORCE(first[pf] == 1, "second[ps] must be '1' to padding.");
          padding_s.push_back(padding_size);
          RecursivePadding(pf + 1, ps, padding_size + 1);
        } else if (second[ps] == first[pf]) {
          RecursivePadding(pf + 1, ps + 1, padding_size + 1);
        } else if (second[ps] == 1) {
          padding_f.push_back(padding_size);
          RecursivePadding(pf, ps + 1, padding_size + 1);
        } else if (first[ps] == 1) {
          padding_s.push_back(padding_size);
          RecursivePadding(pf + 1, ps, padding_size + 1);
        } else {
          PADDLE_THROW("Padding Error.");
        }
      };
  RecursivePadding(0, 0, 0);
  VLOG(4) << "GetPaddingVector result: " << utils::Join(padding_f, ",")
          << " vs " << utils::Join(padding_s, ",");
  return std::tuple(padding_f, padding_s);
}

StmtPattern MergePatternImpl(const HorizontalFusionPattern& first,
                             const HorizontalFusionPattern& second) {
  const auto& [f, s] = GetPaddingVector(GetLoopFramework(StmtPattern(first)),
                                        GetLoopFramework(StmtPattern(second)));
  typename HorizontalFusionPattern::PaddingStmtPattern pad_first = {first, f};
  typename HorizontalFusionPattern::PaddingStmtPattern pad_second = {second, s};
  return HorizontalFusionPattern(
      {pad_first, pad_second},
      std::make_shared<FusionTracker>(first.tracker_, second.tracker_));
}

//

StmtPattern MergePattern(const StmtPattern& first, const StmtPattern& second) {
  VLOG(4) << "MergePattern: " << GetPatternName(first) << " x "
          << GetPatternName(second);
  const auto PatternMatch = adt::match{
      [&](const ReduceTreePattern& lhs, const ReduceTreePattern& rhs) {
        return MergePatternImpl(lhs, rhs);
      },
      [&](const ReduceTreePattern& lhs, const TrivialPattern& rhs) {
        return MergePatternImpl(lhs, rhs);
      },
      [&](const TrivialPattern& lhs, const ReducePattern& rhs) {
        return MergePatternImpl(lhs, rhs);
      },
      [&](const TrivialPattern& lhs, const TrivialPattern& rhs) {
        return MergePatternImpl(lhs, rhs);
      },
      [&](const TrivialPattern& lhs, const ReduceTreePattern& rhs) {
        return MergePatternImpl(lhs, rhs);
      },
      [&](const TrivialPattern& lhs, const ReduceTreePlusTrivialPattern& rhs) {
        return MergePatternImpl(lhs, rhs);
      },
      [&](const TrivialPattern& lhs, const AnchorPattern& rhs) {
        return MergePatternImpl(lhs, rhs);
      },
      [&](const AnchorPattern& lhs, const AnchorPattern& rhs) {
        return MergePatternImpl(lhs, rhs);
      },
      [&](const HorizontalFusionPattern& lhs,
          const HorizontalFusionPattern& rhs) {
        return MergePatternImpl(lhs, rhs);
      },
      [&](const auto& lhs, const auto& rhs) -> StmtPattern {
        CHECK(false) << "Found not support merge!" << GetPatternName(first)
                     << "X" << GetPatternName(second);
      },
  };
  return std::visit(PatternMatch, first.variant(), second.variant());
}

void SetReturnInstrImpl(const TrivialPattern& pattern) {}
void SetReturnInstrImpl(const ReducePattern& pattern) {}
void SetReturnInstrImpl(const ReduceTreePattern& pattern) {}
void SetReturnInstrImpl(const ReduceTreePlusTrivialPattern& pattern) {}
void SetReturnInstrImpl(const AnchorPattern& pattern) {}
void SetReturnInstrImpl(const HorizontalFusionPattern& pattern) {}
void SetReturnInstrImpl(const UnsupportPattern& pattern) {}

void SetReturnInstr(const StmtPattern& s) {
  std::visit([](const auto& impl) { SetReturnInstrImpl(impl); }, s.variant());
}

}  // namespace cinn::fusion
