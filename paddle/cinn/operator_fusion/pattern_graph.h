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

#include "paddle/cinn/operator_fusion/pattern_node.h"
#include "paddle/cinn/operator_fusion/policy/policy_manager.h"
#include "paddle/cinn/operator_fusion/policy/relative_judge_policy.h"
#include "paddle/cinn/operator_fusion/utils.h"
#include "paddle/common/enforce.h"

namespace cinn::fusion {

template <typename T>
using PatternNodePtrSet = std::unordered_set<PatternNodePtr<T>>;
template <typename T>
using MergePatternFn =
    std::function<StmtPattern<T>(const StmtPattern<T>&, const StmtPattern<T>&)>;

template <typename T>
class PatternGraph {
 public:
  PatternGraph(const std::vector<PatternContent<T>>& nodes,
               const std::vector<pir::Value>& outputs,
               const PolicyManager<T> policy_manager,
               const PolicyManager<T> topo_manager);

  std::vector<PatternNodePtr<T>> ClusterOps();

  void SinkTrivialPattern();
  void HorizontalFusion();
  void ReduceLiftReduceTree();
  void ReduceTreeGrown();
  void ReduceTree_Trivial_Fusion();

  void RemoveNode(const PatternNodePtr<T>& node);
  void AppendNode(const PatternNodePtr<T>& node);
  std::string GraphInfo() const;
  PatternNodePtr<T> MergeNode(const PatternNodePtr<T>& upstream,
                              const PatternNodePtr<T>& downstream,
                              MergePatternFn<T> merge_pattern_fn);
  std::vector<PatternNodePtr<T>> SortByTopoOrder();

  const PatternNodePtrSet<T>& all_pattern_nodes() const {
    return all_pattern_nodes_;
  }
  const std::vector<pir::Value>& outputs() const { return outputs_; }
  const PolicyManager<T>& policy_manager() const { return policy_manager_; }
  const PolicyManager<T>& topo_manager() const { return topo_manager_; }

 private:
  PatternNodePtrSet<T> all_pattern_nodes_;
  std::vector<pir::Value> outputs_;
  PolicyManager<T> policy_manager_;
  PolicyManager<T> topo_manager_;
};

// PatternGraphFusionOperation := (GraphMatcher, GraphOperation)
// SearchAlgorithm := NodePattern | EdgePattern | GraphMatcher
// GraphOperation := Merge2Node | SplitNode | SplitAllAndMergeDownstream

struct NodePattern {};
struct EdgePattern {};
struct GraphPattern {};     // not implemented.
struct NodePairPattern {};  // not implemented.

template <typename Kind,
          typename Phrase,
          typename GraphMatcher,
          typename GraphOperation>
struct SearchAlgorithm {};

template <typename Phrase, typename GraphMatcher, typename GraphOperation>
struct SearchAlgorithm<NodePattern, Phrase, GraphMatcher, GraphOperation> {
  PatternGraph<Phrase>* graph_;
  PatternNodePtrSet<Phrase> visited_nodes;

  explicit SearchAlgorithm(PatternGraph<Phrase>* graph) {
    VLOG(4) << "Create NodePattern algorithm.";
    graph_ = graph;
  }

  PatternNodePtr<Phrase> FindMatchedNode() {
    for (PatternNodePtr<Phrase> iter_node : graph_->all_pattern_nodes()) {
      if (GraphMatcher()(*graph_, iter_node) &&
          !visited_nodes.count(iter_node)) {
        visited_nodes.insert(iter_node);
        VLOG(4) << "Find Matched Node: " << iter_node;
        return iter_node;
      }
    }
    VLOG(4) << "Can't find matched node any more.";
    return nullptr;
  }

  void operator()() {
    while (true) {
      PatternNodePtr<Phrase> node = FindMatchedNode();
      if (node == nullptr) {
        break;
      }
      GraphOperation()(graph_, node);
    }
  }
};

template <typename Phrase, typename GraphMatcher, typename GraphOperation>
struct SearchAlgorithm<NodePairPattern, Phrase, GraphMatcher, GraphOperation> {
  PatternGraph<Phrase>* graph_;
  std::set<std::pair<PatternNodePtr<Phrase>, PatternNodePtr<Phrase>>>
      visited_node_pair;
  explicit SearchAlgorithm(PatternGraph<Phrase>* graph) {
    VLOG(4) << "Create NodePairPattern algorithm.";
    graph_ = graph;
  }
  std::optional<std::pair<PatternNodePtr<Phrase>, PatternNodePtr<Phrase>>>
  FindMatchedPair() {
    for (PatternNodePtr<Phrase> i : graph_->all_pattern_nodes()) {
      for (PatternNodePtr<Phrase> j : graph_->all_pattern_nodes()) {
        if (i == j) continue;
        const auto& pair = std::make_pair(i, j);
        if (GraphMatcher()(*graph_, i, j) && !visited_node_pair.count(pair)) {
          visited_node_pair.insert(pair);
          VLOG(4) << "Find Matched Node Pair: (" << i << ", " << j << ")";
          return pair;
        }
      }
    }
    VLOG(4) << "Can't find matched node any more.";
    return {};
  }
  void operator()() {
    while (true) {
      const auto& node = FindMatchedPair();
      if (!node.has_value()) break;
      const auto& [i, j] = node.value();
      GraphOperation()(graph_, i, j);
    }
  }
};

// Operation

struct MergeReduceTreeOperation {
  template <typename Phrase>
  void operator()(PatternGraph<Phrase>* graph, PatternNodePtr<Phrase> node) {
    PADDLE_ENFORCE_EQ(
        node->downstream().size(),
        1,
        phi::errors::PreconditionNotMet(
            "The downstream of the ReduceTree node should be 1, but got %d.",
            node->downstream().size()));
    auto downstream = node->downstream().at(0);
    auto merged_node = graph->MergeNode(node, downstream, MergePattern<Phrase>);
    graph->RemoveNode(downstream);
    graph->RemoveNode(node);
    VLOG(4) << "MergeReduceTreeOperation: \nupstream " << node->DebugStr()
            << "\ndownstream " << downstream->DebugStr() << "\nmerged "
            << merged_node->DebugStr();
  }
};

struct MergeReduceTreeAndTrivialOperation {
  template <typename Phrase>
  void operator()(PatternGraph<Phrase>* graph, PatternNodePtr<Phrase> node) {
    PADDLE_ENFORCE_EQ(
        node->downstream().size(),
        1,
        phi::errors::PreconditionNotMet(
            "The downstream of the ReduceTree node should be 1, but got %d.",
            node->downstream().size()));
    auto downstream = node->downstream().at(0);
    auto fake_reduce_iter_idx =
        graph->policy_manager().GetFakeReduceIterIdx(node, downstream);
    const auto merge_pattern_fn = [&fake_reduce_iter_idx](
                                      const StmtPattern<Phrase>& first,
                                      const StmtPattern<Phrase>& secend) {
      auto rt_pattern = std::get<ReduceTreePlusTrivialPattern<Phrase>>(
          MergePattern<Phrase>(first, secend));
      rt_pattern.fake_reduce_iter_idx = fake_reduce_iter_idx;
      return rt_pattern;
    };
    PatternNodePtr<Phrase> merged_node =
        graph->MergeNode(node, downstream, merge_pattern_fn);
    graph->RemoveNode(downstream);
    graph->RemoveNode(node);
    VLOG(4) << "MergeReduceTreeAndTrivialOperation: \nupstream "
            << node->DebugStr() << "\ndownstream " << downstream->DebugStr()
            << "\nmerged " << merged_node->DebugStr();
  }
};

struct LiftReduceToReduceTreeOperation {
  template <typename Phrase>
  void operator()(PatternGraph<Phrase>* graph, PatternNodePtr<Phrase> node) {
    const auto& reduce_pattern = ToReducePattern<Phrase>(node->stmt_pattern());
    node->set_stmt_pattern(ReduceTreePattern<Phrase>({}, reduce_pattern));
    VLOG(4) << "LiftReduceToReduceTreeOperation: \nnode " << node->DebugStr();
  }
};

struct MergeTrivialPatternOperation {
  template <typename Phrase>
  void operator()(PatternGraph<Phrase>* graph,
                  PatternNodePtr<Phrase> upstream) {
    std::vector<PatternNodePtr<Phrase>> fusion_candidate =
        upstream->downstream();
    upstream->ClearDownstream();
    for (const auto& downstream : fusion_candidate) {
      if (std::holds_alternative<ReducePattern<Phrase>>(
              downstream->stmt_pattern()) ||
          std::holds_alternative<TrivialPattern<Phrase>>(
              downstream->stmt_pattern())) {
        auto merged_node =
            graph->MergeNode(upstream, downstream, MergePattern<Phrase>);
        graph->RemoveNode(downstream);
        VLOG(4) << "MergeTrivialPatternOperation: \nupstream "
                << upstream->DebugStr() << "\ndownstream "
                << downstream->DebugStr() << "\nmerged "
                << merged_node->DebugStr();
      } else {
        upstream->AddNodeToDownstream(downstream);
      }
    }
    if (upstream->downstream().empty()) {
      graph->RemoveNode(upstream);
    }
  }
};

struct LiftToHorizontalFusionPatternOperation {
  template <typename Phrase>
  void operator()(PatternGraph<Phrase>* graph, PatternNodePtr<Phrase> node) {
    node->set_stmt_pattern(HorizontalFusionPattern<Phrase>(
        {typename HorizontalFusionPattern<Phrase>::PaddingStmtPattern(
            node->stmt_pattern(), {})}));
  }
};

struct HorizontalFusionOperation {
  template <typename Phrase>
  void operator()(PatternGraph<Phrase>* graph,
                  const PatternNodePtr<Phrase>& i,
                  const PatternNodePtr<Phrase>& j) {
    VLOG(4) << "Start HorizontalFusionOperation";
    PADDLE_ENFORCE_EQ(
        GetPatternName(i->stmt_pattern()),
        HorizontalFusionPattern<Phrase>::name(),
        phi::errors::PreconditionNotMet(
            "The pattern of the first node should be HorizontalFusionPattern, "
            "but got %s.",
            GetPatternName(i->stmt_pattern())));
    PADDLE_ENFORCE_EQ(
        GetPatternName(j->stmt_pattern()),
        HorizontalFusionPattern<Phrase>::name(),
        phi::errors::PreconditionNotMet(
            "The pattern of the second node should be HorizontalFusionPattern, "
            "but got %s.",
            GetPatternName(j->stmt_pattern())));
    auto merged_node = graph->MergeNode(i, j, MergePattern<Phrase>);
    VLOG(4) << "MergeHorizontalPattern: \ni " << i->DebugStr() << "\nj "
            << j->DebugStr() << "\nmerged " << merged_node->DebugStr();
    graph->RemoveNode(i);
    graph->RemoveNode(j);
    VLOG(4) << "After HorizontalFusionOperation, Graph is"
            << graph->GraphInfo();
  }
};

// Matcher

template <typename StmtPattern>
struct AlwaysTrue {
  template <typename T>
  bool operator()(const PatternGraph<T>& graph, const PatternNodePtr<T>& node) {
    return true;
  }
};

template <typename StmtPattern>
struct StmtPatternGraphMatcher {
  template <typename T>
  bool operator()(const PatternGraph<T>& graph, const PatternNodePtr<T>& node) {
    return GetPatternName(node->stmt_pattern()) == StmtPattern::name();
  }
};

struct CanFuseRxTMatcher {
  template <typename T>
  bool operator()(const PatternGraph<T>& graph, const PatternNodePtr<T>& node) {
    return (
        std::holds_alternative<ReduceTreePattern<T>>(node->stmt_pattern()) &&
        !node->downstream().empty() &&
        std::holds_alternative<TrivialPattern<T>>(
            node->downstream().at(0)->stmt_pattern()));
  }
};

struct CanFuseReduceTreeMatcher {
  template <typename T>
  bool operator()(const PatternGraph<T>& graph, const PatternNodePtr<T>& node) {
    return StmtPatternGraphMatcher<ReduceTreePattern<T>>()(graph, node) &&
           !node->downstream().empty() &&
           std::holds_alternative<ReduceTreePattern<T>>(
               node->downstream().at(0)->stmt_pattern()) &&
           graph.policy_manager().CanFuse(node, node->downstream().at(0));
  }
};

struct CanFuseReduceTreeAndTrivialMatcher {
  template <typename T>
  bool operator()(const PatternGraph<T>& graph, const PatternNodePtr<T>& node) {
    return StmtPatternGraphMatcher<ReduceTreePattern<T>>()(graph, node) &&
           !node->downstream().empty() &&
           std::holds_alternative<TrivialPattern<T>>(
               node->downstream().at(0)->stmt_pattern()) &&
           graph.policy_manager().CanFuse(node, node->downstream().at(0));
  }
};

template <typename T>
struct HorizontalCheckMiddleOutputVar {
  bool IsAnyOpUseOutput(const std::vector<pir::Operation*>& ops,
                        const std::vector<pir::Value>& output_value) {
    std::unordered_set<pir::Value> set(output_value.begin(),
                                       output_value.end());
    for (const auto& op : ops) {
      for (const auto& var : op->operands()) {
        if (set.count(var.source())) {
          return true;
        }
      }
    }
    return false;
  }
  bool operator()(const PatternGraph<T>& graph,
                  const PatternNodePtr<T>& lhs,
                  const PatternNodePtr<T>& rhs) {
    const auto& output_value = graph.outputs();
    const auto& ops = ConcatVector(GetOpsInPattern(lhs->stmt_pattern()),
                                   GetOpsInPattern(rhs->stmt_pattern()));
    return !IsAnyOpUseOutput(ops, output_value);
  }
};

template <typename T>
struct HorizontalFusionConstrain {
  bool operator()(const PatternGraph<T>& graph,
                  const PatternNodePtr<T>& lhs,
                  const PatternNodePtr<T>& rhs) {
    if (!StmtPatternGraphMatcher<HorizontalFusionPattern<T>>()(graph, lhs)) {
      return false;
    }
    if (!StmtPatternGraphMatcher<HorizontalFusionPattern<T>>()(graph, rhs)) {
      return false;
    }
    const auto& lhs_pattern =
        std::get<HorizontalFusionPattern<T>>(lhs->stmt_pattern());
    const auto& rhs_pattern =
        std::get<HorizontalFusionPattern<T>>(rhs->stmt_pattern());

    return graph.topo_manager().CanFuse(lhs, rhs) &&
           IsLoopFrameworkEqual(lhs_pattern.padding_patterns_.back().pattern,
                                rhs_pattern.padding_patterns_.back().pattern);
  }
};

struct NonSinkNodeMatcher {
  template <typename T>
  bool operator()(const PatternGraph<T>& graph, const PatternNodePtr<T>& node) {
    return !node->downstream().empty();
  }
};

struct IsOutputNodeMatcher {
  template <typename T>
  bool operator()(const PatternGraph<T>& graph, const PatternNodePtr<T>& node) {
    bool res = IsAnyFirstInSecond(node->sink_op()->results(), graph.outputs());
    return res;
  }
};

struct IsNotOutputNodeMatcher {
  template <typename T>
  bool operator()(const PatternGraph<T>& graph, const PatternNodePtr<T>& node) {
    bool res = !IsOutputNodeMatcher()(graph, node);
    return res;
  }
};

template <int N>
struct DownstreamSmallerThan {
  template <typename T>
  bool operator()(const PatternGraph<T>& graph, const PatternNodePtr<T>& node) {
    return node->downstream().size() < N;
  }
};

template <typename... Args>
struct And {};

template <typename A>
struct And<A> {
  template <typename T>
  bool operator()(const PatternGraph<T>& graph, const PatternNodePtr<T>& node) {
    return A()(graph, node);
  }
  template <typename T>
  bool operator()(const PatternGraph<T>& graph,
                  const PatternNodePtr<T>& lhs,
                  const PatternNodePtr<T>& rhs) {
    return A()(graph, lhs, rhs);
  }
};

template <typename A, typename... Args>
struct And<A, Args...> {
  template <typename T>
  bool operator()(const PatternGraph<T>& graph, const PatternNodePtr<T>& node) {
    return A()(graph, node) && And<Args...>()(graph, node);
  }
  template <typename T>
  bool operator()(const PatternGraph<T>& graph,
                  const PatternNodePtr<T>& lhs,
                  const PatternNodePtr<T>& rhs) {
    return A()(graph, lhs, rhs) && And<Args...>()(graph, lhs, rhs);
  }
};

template <typename... Args>
struct Or {};

template <typename A>
struct Or<A> {
  template <typename T>
  bool operator()(const PatternGraph<T>& graph, const PatternNodePtr<T>& node) {
    return A()(graph, node);
  }
};

template <typename A, typename... Args>
struct Or<A, Args...> {
  template <typename T>
  bool operator()(const PatternGraph<T>& graph, const PatternNodePtr<T>& node) {
    return A()(graph, node) || Or<Args...>()(graph, node);
  }
};

template <typename A>
struct Not {
  template <typename T>
  bool operator()(const PatternGraph<T>& graph, const PatternNodePtr<T>& node) {
    return !A()(graph, node);
  }
};

template <typename Kind,
          typename Phrase,
          typename GraphMatcher,
          typename GraphOperation>
void GraphTransformer(PatternGraph<Phrase>* graph) {
  VLOG(4) << "Start GraphTransformer...";
  auto alog =
      SearchAlgorithm<Kind, Phrase, GraphMatcher, GraphOperation>(graph);
  alog();
}

}  // namespace cinn::fusion
