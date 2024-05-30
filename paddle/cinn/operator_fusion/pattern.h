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

#include <unordered_set>
#include <variant>
#include <vector>
#include "glog/logging.h"
#include "paddle/cinn/operator_fusion/fusion_tracker/tracker.h"
#include "paddle/cinn/operator_fusion/pir_graph_analyzing/anchor_transform.h"
#include "paddle/cinn/operator_fusion/utils.h"
#include "paddle/pir/include/core/operation.h"

namespace cinn::fusion {

struct PatternContent {
  explicit PatternContent(pir::Operation* op) : op(op) {}
  pir::Operation* op;
  bool operator==(const PatternContent& other) const { return op == other.op; }
};

struct StmtPattern;

struct TrivialPattern {
  explicit TrivialPattern(const std::vector<pir::Operation*>& ops,
                          pir::Operation* sink_op,
                          const FusionTrackerPtr& tracker)
      : ops_(ops), sink_op_(sink_op), tracker_(tracker) {
    name_ = UniqueName();
  }
  std::vector<pir::Operation*> ops_;
  pir::Operation* sink_op_;
  std::vector<pir::Operation*> ops() const { return ops_; }
  pir::Operation* sink_op() const { return sink_op_; }

  static std::string UniqueName() {
    static std::atomic<int64_t> counter = 0;
    counter += 1;
    return "T_" + std::to_string(counter);
  }
  std::string name() const { return name_; }
  std::string name_;

  FusionTrackerPtr tracker_;
};

struct ReducePattern {
  explicit ReducePattern(const std::vector<pir::Operation*>& ops,
                         const FusionTrackerPtr& tracker)
      : ops_(ops), tracker_(tracker) {
    name_ = UniqueName();
  }
  std::vector<pir::Operation*> ops_;
  std::vector<pir::Operation*> ops() const { return ops_; }
  pir::Operation* GetReduceOp() const { return ops_.back(); }

  static std::string UniqueName() {
    static std::atomic<int64_t> counter = 0;
    counter += 1;
    return "T_" + std::to_string(counter);
  }
  std::string name() const { return name_; }
  std::string name_;

  FusionTrackerPtr tracker_;
};

struct ReduceTreePattern {
  explicit ReduceTreePattern(const std::vector<ReduceTreePattern>& childs,
                             const ReducePattern& root,
                             const FusionTrackerPtr& tracker)
      : childs_(childs), root_(root), tracker_(tracker) {
    name_ = UniqueName();
  }
  const ReducePattern& GetRootPattern() const { return root_; }
  std::vector<pir::Operation*> ops() const {
    std::vector<pir::Operation*> result{root_.ops()};
    for (const auto& child : childs_) {
      result = UniqueConcatVector(result, child.ops());
    }
    return result;
  }
  const std::vector<ReduceTreePattern>& childs() const { return childs_; }
  std::vector<ReduceTreePattern>& childs() { return childs_; }
  void InsertChild(const ReduceTreePattern& child) { childs_.push_back(child); }
  std::vector<ReducePattern> FlattenReducePattern() const {
    std::vector<ReducePattern> result;
    for (const auto& child : childs_) {
      result = ConcatVector(result, child.FlattenReducePattern());
    }
    return result;
  }

  static std::string UniqueName() {
    static std::atomic<int64_t> counter = 0;
    counter += 1;
    return "RTree_" + std::to_string(counter);
  }
  std::string name() const { return name_; }
  std::string name_;

  FusionTrackerPtr tracker_;

 private:
  std::vector<ReduceTreePattern> childs_;
  ReducePattern root_;
};

struct ReduceTreePlusTrivialPattern {
  explicit ReduceTreePlusTrivialPattern(const ReduceTreePattern& tree,
                                        const TrivialPattern& sink_trivial,
                                        const FusionTrackerPtr& tracker)
      : tree(tree), sink_trivial(sink_trivial), tracker_(tracker) {
    name_ = UniqueName();
  }
  ReduceTreePattern tree;
  TrivialPattern sink_trivial;
  std::vector<pir::Operation*> ops() const {
    return UniqueConcatVector(tree.ops(), sink_trivial.ops());
  }
  std::vector<size_t> fake_reduce_iter_idx;

  static std::string UniqueName() {
    static std::atomic<int64_t> counter = 0;
    counter += 1;
    return "RTreeT_" + std::to_string(counter);
  }
  std::string name() const { return name_; }
  std::string name_;

  FusionTrackerPtr tracker_;
};

struct AnchorPattern {
  explicit AnchorPattern(const std::vector<pir::Operation*>& ops,
                         const pir::Value& anchor,
                         const AnchorState& init_anchor_state,
                         const FusionTrackerPtr& tracker)
      : ops_(ops),
        anchor_(anchor),
        anchor_state(init_anchor_state),
        tracker_(tracker) {
    name_ = UniqueName();
  }
  AnchorState anchor_state;

  std::vector<pir::Operation*> ops() const { return ops_; }
  pir::Value anchor() const { return anchor_; }
  bool can_recompute() const {
    // Current Algorithm:
    // An AnchorPattern can be recomputed iff:
    // 1. It didn't go through any pattern merging during prior fusions, which
    // means it only has one output_expr in anchor_state.
    // 2. It only contains trivial ops.

    if (anchor_state.promise.size() > 1) {
      return false;
    }

    for (const auto& op : ops_) {
      const auto& op_kind = GetOpPatternKind(op);
      if (op_kind >= hlir::framework::kReduction) {
        return false;
      }
    }

    return true;
  }

  static std::string UniqueName() {
    static std::atomic<int64_t> counter = 0;
    counter += 1;
    return "Anchor_" + std::to_string(counter);
  }
  std::string name() const { return name_; }
  std::string name_;

  FusionTrackerPtr tracker_;

 private:
  std::vector<pir::Operation*> ops_;
  pir::Value anchor_;  // Choose only one anchor
};

struct HorizontalFusionPattern {
  struct PaddingStmtPattern {
    StmtPattern pattern;
    std::vector<int> padding_pos;
    PaddingStmtPattern(const StmtPattern& pattern,
                       const std::vector<int>& padding_pos)
        : pattern(pattern), padding_pos(padding_pos) {}
  };
  explicit HorizontalFusionPattern(
      const std::vector<PaddingStmtPattern>& patterns,
      const FusionTrackerPtr& tracker)
      : padding_patterns_(patterns), tracker_(tracker) {
    name_ = UniqueName();
  }
  std::vector<PaddingStmtPattern> padding_patterns_;
  std::vector<pir::Operation*> ops() const {
    std::vector<pir::Operation*> result;
    for (const auto& pattern : padding_patterns_) {
      auto ops = GetOpsInPattern(pattern.pattern);
      ExtendVector(&result, ops);
    }
    return result;
  }

  static std::string UniqueName() {
    static std::atomic<int64_t> counter = 0;
    counter += 1;
    return "Horizontal_" + std::to_string(counter);
  }
  std::string name() const { return name_; }
  std::string name_;

  FusionTrackerPtr tracker_;
};

struct UnsupportPattern {
  explicit UnsupportPattern(const std::vector<pir::Operation*>& ops,
                            const FusionTrackerPtr& tracker)
      : ops_(ops), tracker_(tracker) {
    name_ = UniqueName();
  }
  std::vector<pir::Operation*> ops_;
  std::vector<pir::Operation*> ops() const { return ops_; }
  static std::string UniqueName() {
    static std::atomic<int64_t> counter = 0;
    counter += 1;
    return "Unsupport_" + std::to_string(counter);
  }
  std::string name() const { return name_; }
  std::string name_;

  FusionTrackerPtr tracker_;
};

using StmtPatternBase = std::variant<TrivialPattern,
                                     ReducePattern,
                                     ReduceTreePattern,
                                     ReduceTreePlusTrivialPattern,
                                     HorizontalFusionPattern,
                                     UnsupportPattern,
                                     AnchorPattern>;

struct StmtPattern final : public StmtPatternBase {
  using StmtPatternBase::StmtPatternBase;
  const StmtPatternBase& variant() const {
    return static_cast<const StmtPatternBase&>(*this);
  }
};

std::string StmtPatternDebugStr(const StmtPattern& stmt) {
  std::stringstream ss;
  auto all_ops = GetOpsInPattern(stmt);
  ss << "StmtPattern, size " << all_ops.size() << " :\n";
  ss << OpsDebugStr(all_ops);
  return ss.str();
}

std::string GetPatternName(const StmtPattern& s) {
  return std::visit([](const auto& impl) { return impl.name(); }, s.variant());
}

FusionTrackerPtr GetPatternTracker(const StmtPattern& s) {
  return std::visit([](const auto& impl) { return impl.tracker_; },
                    s.variant());
}

std::vector<pir::Operation*> GetOpsInPattern(const StmtPattern& pattern) {
  return std::visit([](const auto& impl) { return impl.ops(); },
                    pattern.variant());
}

std::unordered_set<pir::Value> GetPatternInputValuesIncludeInner(
    const StmtPattern& A) {
  std::unordered_set<pir::Value> result;
  for (const auto& op : GetOpsInPattern(A)) {
    for (const auto& value : op->operands()) {
      result.insert(value.source());
    }
  }
  return result;
}

std::unordered_set<pir::Value> GetPatternOutputValuesIncludedInner(
    const StmtPattern& A) {
  std::unordered_set<pir::Value> result;
  for (const auto& op : GetOpsInPattern(A)) {
    for (const auto& value : op->results()) {
      result.insert(value);
    }
  }
  return result;
}

std::unordered_set<pir::Value> GetPatternInputValues(const StmtPattern& A) {
  auto all_input_values = GetPatternInputValuesIncludeInner(A);
  for (const auto& value : GetPatternOutputValuesIncludedInner(A)) {
    all_input_values.erase(value);
  }
  VLOG(4) << "GetPatternInputValues: " << all_input_values.size();
  return all_input_values;
}

}  // namespace cinn::fusion
