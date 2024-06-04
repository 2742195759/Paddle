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

#include "glog/logging.h"

#include "paddle/cinn/hlir/framework/pir/trivial_op_impl.h"
#include "paddle/cinn/operator_fusion/fusion_tracker/interpreter.h"

namespace cinn::fusion {

using TrivialOp = cinn::hlir::framework::pir::trivial_fusion_detail::TrivialOp;
using ReduceOp = cinn::hlir::framework::pir::trivial_fusion_detail::ReduceOp;
using FusionOp = std::variant<ReduceOp, TrivialOp>;

template <>
StmtPattern<BackendStage> ConvertToStmtPattern(
    const PatternContent<BackendStage>& content) {
  const auto& kind = GetOpPatternKind(content.op);
  if (kind == hlir::framework::kReduction) {
    CHECK(content.expr.has_value());
    return ReducePattern<BackendStage>({content.op},
                                       ReduceOp(content.expr.value()));
  } else if (kind == hlir::framework::kElementWise ||
             kind == hlir::framework::kBroadcast ||
             kind == hlir::framework::kInjective) {
    CHECK(content.expr.has_value());
    return TrivialPattern<BackendStage>(
        {content.op}, content.op, TrivialOp(content.expr.value()));
  } else {
    CHECK(false);
    return UnsupportPattern<BackendStage>({content.op});
  }
}

// template StmtPattern<BackendStage> RT_x_RT(const
// ReduceTreePattern<BackendStage>& upstream, const
// ReduceTreePattern<BackendStage>& downstream);

template <>
StmtPattern<BackendStage> MergePatternImpl(
    const ReduceTreePattern<BackendStage>& first,
    const TrivialPattern<BackendStage>& second) {
  return ReduceTreePlusTrivialPattern<BackendStage>(first, second);
}

template <>
StmtPattern<BackendStage> MergePatternImpl(
    const TrivialPattern<BackendStage>& first,
    const ReducePattern<BackendStage>& second) {
  const auto& ops = UniqueConcatVector(GetOpsInPattern<BackendStage>(first),
                                       GetOpsInPattern<BackendStage>(second));
  const auto& reduce_op =
      cinn::hlir::framework::pir::trivial_fusion_detail::TrivalxOther_Fusion(
          first.trivial_op, second.reduce_op);
  return ReducePattern<BackendStage>(ops, reduce_op);
}

template <>
StmtPattern<BackendStage> MergePatternImpl(
    const TrivialPattern<BackendStage>& first,
    const TrivialPattern<BackendStage>& second) {
  const auto& ops = UniqueConcatVector(GetOpsInPattern<BackendStage>(first),
                                       GetOpsInPattern<BackendStage>(second));
  const auto& trivial_op =
      cinn::hlir::framework::pir::trivial_fusion_detail::TrivalxOther_Fusion(
          first.trivial_op, second.trivial_op);
  return TrivialPattern<BackendStage>(ops, second.sink_op(), trivial_op);
}

template <>
StmtPattern<BackendStage> MergePatternImpl(
    const TrivialPattern<BackendStage>& first,
    const AnchorPattern<BackendStage>& second) {
  AnchorState<BackendStage> new_anchor_state = second.anchor_state;

  for (int i = 0; i < new_anchor_state.promise.size(); i++) {
    new_anchor_state.promise[i].root_fusion_op = std::visit(
        [first](const auto& arg) -> FusionOp {
          return cinn::hlir::framework::pir::trivial_fusion_detail::
              TrivalxOther_Fusion(first.trivial_op, arg);
        },
        new_anchor_state.promise[i].root_fusion_op);
  }

  return AnchorPattern<BackendStage>(
      UniqueConcatVector(GetOpsInPattern<BackendStage>(first),
                         GetOpsInPattern<BackendStage>(second)),
      second.anchor(),
      new_anchor_state);
}

template <>
StmtPattern<BackendStage> MergePatternImpl(
    const AnchorPattern<BackendStage>& source,
    const AnchorPattern<BackendStage>& dest) {
  const auto& contents =
      UniqueConcatVector(GetOpsInPattern<BackendStage>(source),
                         GetOpsInPattern<BackendStage>(dest));
  return AnchorPattern<BackendStage>(
      contents, source.anchor(), AnchorState<BackendStage>({}));
}

/// Start: Tmp Transform Operation for ReduceTree
std::vector<FusionOp> ReduceTransformRecursive(
    ReduceOp reduce_op,
    const ReduceTreePattern<BackendStage>& reduce_tree_pattern,
    const std::vector<size_t>& fake_reduce_iter_idx = {}) {
  FusionOp root_op = reduce_op;
  VLOG(4) << "ReduceTransformRecursive: " << *_GetFuncBodyPointer(root_op);
  std::vector<FusionOp> result;
  for (const auto& child_tree : reduce_tree_pattern.childs()) {
    const auto& child_reduce_op = child_tree.GetRootPattern().reduce_op;
    auto transformed_nodes = cinn::hlir::framework::pir::trivial_fusion_detail::
        TransformReduceLoopRange(
            child_reduce_op, &root_op, fake_reduce_iter_idx);
    for (auto& node : transformed_nodes) {
      auto child_flatten =
          ReduceTransformRecursive(std::get<ReduceOp>(node), child_tree);
      result.insert(result.end(), child_flatten.begin(), child_flatten.end());
    }
  }
  result.push_back(root_op);
  VLOG(4) << "ReduceTransformRecursive: End";
  return result;
}

std::vector<FusionOp> ReduceTreeTrivialTransformRecursive(
    TrivialOp trivial_op,
    const ReduceTreePlusTrivialPattern<BackendStage>& rt_pattern) {
  FusionOp root_op = trivial_op;
  VLOG(4) << "ReduceTrivialTransformRecursive: "
          << *_GetFuncBodyPointer(root_op);
  std::vector<FusionOp> result;

  const auto& child_tree = rt_pattern.tree;
  const auto& child_reduce_op = child_tree.GetRootPattern().reduce_op;
  auto transformed_nodes = cinn::hlir::framework::pir::trivial_fusion_detail::
      TransformReduceLoopRange(
          child_reduce_op, &root_op, rt_pattern.fake_reduce_iter_idx);
  for (auto& node : transformed_nodes) {
    auto child_flatten = ReduceTransformRecursive(
        std::get<ReduceOp>(node), child_tree, rt_pattern.fake_reduce_iter_idx);
    result.insert(result.end(), child_flatten.begin(), child_flatten.end());
  }
  //}
  result.push_back(
      cinn::hlir::framework::pir::trivial_fusion_detail::SinkTrivialLoopAlign(
          std::get<TrivialOp>(root_op),
          rt_pattern.tree.GetRootPattern().reduce_op,
          rt_pattern.fake_reduce_iter_idx));
  VLOG(4) << "ReduceTrivialTransformRecursive End;";
  return result;
}

/// End: Tmp Transform Operation for reduce tree
///
struct FusionOp2Expr {
  std::vector<ir::Expr> operator()(const TrivialOp& op) {
    return {op.GetFuncBody()};
  }
  std::vector<ir::Expr> operator()(const ReduceOp& op) {
    const auto& t_r = SplitReduceOp(op);
    return {t_r.first.GetFuncBody(), t_r.second.GetFuncBody()};
  }
};

struct ApplyTransform {
  explicit ApplyTransform(const ir::Expr& expr) : expr_(expr) {}
  ir::Expr operator()(const UnsupportTransformPtr& transform) {
    PADDLE_THROW("Can not do UnsupportTransform");
  }
  ir::Expr operator()(const IdentityTransformPtr& transform) { return expr_; }
  ir::Expr operator()(const AppendDimTransformPtr& transform) {
    PADDLE_THROW("AppendDimTransform not implemented");
  }
  ir::Expr operator()(const DeleteDimTransformPtr& transform) {
    PADDLE_THROW("DeleteDimTransform not implemented");
  }

 private:
  ir::Expr expr_;
};

std::vector<ir::Expr> GetExprFromPattern(
    const StmtPattern<BackendStage>& pattern);

static std::vector<ir::Var> GetAllForIters(const ir::Expr& expr) {
  using cinn::hlir::framework::pir::trivial_fusion_detail::ExprSetFinderUtils::
      ChildFors;
  using cinn::hlir::framework::pir::trivial_fusion_detail::ExprSetFinderUtils::
      ChildScheduleBlockRealizes;
  using cinn::hlir::framework::pir::trivial_fusion_detail::ExprSetFinderUtils::
      FindFather;
  using cinn::hlir::framework::pir::trivial_fusion_detail::ExprSetFinderUtils::
      IsFor;
  using cinn::hlir::framework::pir::trivial_fusion_detail::ExprSetFinderUtils::
      ScheduleBlockRealizeIsNotInit;
  const auto& all_father_fors =
      (ChildScheduleBlockRealizes * ScheduleBlockRealizeIsNotInit *
       FindFather(expr) * IsFor)(expr);
  std::vector<ir::Var> vars;
  for (const auto& for_expr : all_father_fors) {
    vars.push_back(for_expr.As<ir::For>()->loop_var);
  }
  VLOG(4) << "GetAllForIters : " << expr
          << "\n var is : " << utils::Join(vars, ",");
  return vars;
}

ir::Expr UnSqueezeExpr(const ir::Expr& expr,
                       const std::vector<int> padding_vec) {
  using cinn::hlir::framework::pir::trivial_fusion_detail::AppendBound;
  using cinn::hlir::framework::pir::trivial_fusion_detail::ExprSetFinderUtils::
      ChildFors;
  using cinn::hlir::framework::pir::trivial_fusion_detail::ExprSetFinderUtils::
      ChildRootScheduleBlockRealizes;
  using cinn::hlir::framework::pir::trivial_fusion_detail::ExprSetFinderUtils::
      ChildScheduleBlockRealizes;
  using cinn::hlir::framework::pir::trivial_fusion_detail::ExprSetFinderUtils::
      IsForIterVar;
  using cinn::hlir::framework::pir::trivial_fusion_detail::
      ExprTransformerUtils::ReplaceVarTransformer;
  using cinn::hlir::framework::pir::trivial_fusion_detail::
      ExprTransformerUtils::UnsqueezeForTransformer;
  VLOG(4) << "UnSqueezeExpr: " << expr
          << "\npadding vector: " << utils::Join(padding_vec, ", ");
  const auto& vars_in_expr = AppendBound(GetAllForIters(expr), expr);
  // get the all vars.
  int counter = 0;
  auto GenNextName = [&counter]() {
    counter += 1;
    return "expand_var_" + std::to_string(counter);
  };
  std::vector<ir::Var> vars;
  int pointer = 0;
  for (int i = 0; i < vars_in_expr.size() + padding_vec.size(); i++) {
    if (std::find(padding_vec.begin(), padding_vec.end(), i) !=
        padding_vec.end()) {
      vars.emplace_back(Expr(0), Expr(1), GenNextName());
    } else {
      vars.push_back(vars_in_expr[pointer++]);
    }
  }
  // update the is_reduce of expand_var.
  for (int i : padding_vec) {
    if (i == 0) {
      vars[i]->is_reduce_axis = false;
    } else {
      vars[i]->is_reduce_axis = vars[i - 1]->is_reduce_axis;
    }
  }

  // sequencely unsqueeze the ir::Expr.
  ir::Expr result = expr;
  for (int i : padding_vec) {
    if (i > 0) {
      result = UnsqueezeForTransformer((ChildFors * IsForIterVar(vars[i - 1])),
                                       vars[i])(result);
    } else {
      result = UnsqueezeForTransformer(ChildRootScheduleBlockRealizes,
                                       vars[i])(result);
    }
  }
  return result;
}

std::vector<ir::Expr> ApplyTransformToPromise(
    const ExprPromise<BackendStage>& promise) {
  std::function<ir::Expr(ir::Expr)> do_transform =
      [transform_route = promise.transform_route](ir::Expr target) -> ir::Expr {
    for (auto transform : transform_route) {
      target = std::visit(ApplyTransform(target), transform);
    }
    return target;
  };
  return MapVector(std::visit(FusionOp2Expr(), promise.root_fusion_op),
                   do_transform);
}

struct IrExprGetter {
  std::vector<ir::Expr> operator()(
      const TrivialPattern<BackendStage>& pattern) {
    return FusionOp2Expr()(pattern.trivial_op);
  }

  std::vector<ir::Expr> operator()(const ReducePattern<BackendStage>& pattern) {
    return FusionOp2Expr()(pattern.reduce_op);
  }

  std::vector<ir::Expr> operator()(
      const ReduceTreePattern<BackendStage>& pattern) {
    const auto& fusion_op =
        ReduceTransformRecursive(pattern.GetRootPattern().reduce_op, pattern);
    std::function<std::vector<ir::Expr>(const FusionOp& f)> func =
        [](const FusionOp& op) { return std::visit(FusionOp2Expr(), op); };
    return VectorFlatMap(fusion_op, func);
  }

  std::vector<ir::Expr> operator()(
      const ReduceTreePlusTrivialPattern<BackendStage>& pattern) {
    std::function<std::vector<ir::Expr>(const FusionOp& f)> func =
        [](const FusionOp& op) { return std::visit(FusionOp2Expr(), op); };
    const auto& fusion_ops = ReduceTreeTrivialTransformRecursive(
        pattern.sink_trivial.trivial_op, pattern);
    return VectorFlatMap(fusion_ops, func);
  }

  std::vector<ir::Expr> operator()(
      const HorizontalFusionPattern<BackendStage>& pattern) {
    std::vector<ir::Expr> result;
    VLOG(4) << "Get Fusion Ops from HorizontalFusionPattern: "
            << pattern.padding_patterns_.size();
    for (const auto& sub_pattern : pattern.padding_patterns_) {
      std::function<ir::Expr(ir::Expr)> func =
          [&sub_pattern](const ir::Expr& expr) {
            return UnSqueezeExpr(expr, sub_pattern.padding_pos);
          };
      result = ConcatVector(
          result, MapVector(GetExprFromPattern(sub_pattern.pattern), func));
    }
    return result;
  }

  std::vector<ir::Expr> operator()(const AnchorPattern<BackendStage>& pattern) {
    std::vector<ir::Expr> result;
    for (auto promise : pattern.anchor_state.promise) {
      result = ConcatVector(result, ApplyTransformToPromise(promise));
    }
    return result;
  }

  std::vector<ir::Expr> operator()(
      const UnsupportPattern<BackendStage>& pattern) {
    CHECK(false) << "Not Implemented.";
  }
};

// tmp transform for reduce_tree and reduce_tree_trivial.
std::vector<ir::Tensor> GetOutputTensors(const ir::Expr& op_expr) {
  using cinn::hlir::framework::pir::trivial_fusion_detail::ExprSetFinderUtils::
      ChildScheduleBlockRealizes;
  using cinn::hlir::framework::pir::trivial_fusion_detail::ExprSetFinderUtils::
      ChildTensorStores;
  using cinn::hlir::framework::pir::trivial_fusion_detail::ExprSetFinderUtils::
      ScheduleBlockRealizeIsNotInit;
  const auto& tensors =
      (ChildScheduleBlockRealizes * ScheduleBlockRealizeIsNotInit *
       ChildTensorStores)(op_expr);
  std::function<ir::Tensor(ir::Expr)> func = [](const ir::Expr& expr) {
    return expr.As<ir::Store>()->tensor.as_tensor_ref();
  };
  return MapVector(tensors, func);
}

std::vector<ir::Tensor> GetInputTensors(const ir::Expr& op_expr) {
  using cinn::hlir::framework::pir::trivial_fusion_detail::ExprSetFinderUtils::
      ChildScheduleBlockRealizes;
  using cinn::hlir::framework::pir::trivial_fusion_detail::ExprSetFinderUtils::
      ChildTensorLoads;
  using cinn::hlir::framework::pir::trivial_fusion_detail::ExprSetFinderUtils::
      ScheduleBlockRealizeIsNotInit;
  const auto& exprs =
      (ChildScheduleBlockRealizes * ScheduleBlockRealizeIsNotInit *
       ChildTensorLoads)(op_expr);
  std::function<ir::Tensor(ir::Expr)> func = [](const ir::Expr& expr) {
    return expr.As<ir::Load>()->tensor.as_tensor_ref();
  };
  const auto& inputs = MapVector(exprs, func);
  const auto& outputs = GetOutputTensors(op_expr);
  return FilterVector(inputs, [&outputs](const ir::Tensor& tensor) {
    return std::find(outputs.begin(), outputs.end(), tensor) == outputs.end();
  });
}

std::vector<ir::Expr> TopoSort(const std::vector<ir::Expr>& op_exprs) {
  // Topo Sort is important for CINN GroupSchedule.
  std::map<ir::Tensor, std::vector<const ir::Expr*>> tensor2defining_op;
  std::map<ir::Tensor, std::vector<const ir::Expr*>> tensor2used_op;
  for (const auto& op : op_exprs) {
    auto inputs = GetInputTensors(op);
    auto outputs = GetOutputTensors(op);

    if (VLOG_IS_ON(5)) {
      VLOG(4) << "Ir::Expr is: \n" << op;
      VLOG(4) << "Inputs: ";
      for (const auto& input : inputs) {
        VLOG(4) << input->name;
      }
      VLOG(4) << "Outputs: ";
      for (const auto& output : outputs) {
        VLOG(4) << output->name;
      }
    }
    for (const auto& input : inputs) {
      tensor2used_op[input].push_back(&op);
    }
    for (const auto& output : outputs) {
      tensor2defining_op[output].push_back(&op);
    }
  }

  // Collect Downstreams
  std::map<const ir::Expr*, std::vector<const ir::Expr*>> op2downstreams;
  std::map<const ir::Expr*, int> degrees;
  for (const auto& op : op_exprs) {
    degrees[&op] = 0;
  }
  for (const auto& op : op_exprs) {
    auto outputs = GetOutputTensors(op);
    std::vector<const ir::Expr*> downstreams;
    for (const auto& output : outputs) {
      downstreams = ConcatVector(downstreams, tensor2used_op[output]);
    }
    for (const auto& downstream : downstreams) {
      degrees[downstream]++;
    }
    op2downstreams[&op] = downstreams;
  }

  // Topo Sort
  std::vector<const ir::Expr*> result;
  std::queue<const ir::Expr*> q;
  for (const auto& op : op_exprs) {
    if (degrees[&op] == 0) {
      q.push(&op);
    }
  }
  while (!q.empty()) {
    auto* cur = q.front();
    VLOG(4) << "Topo Sort Visit Order is:" << GetOutputTensors(*cur)[0]->name;
    q.pop();
    result.push_back(cur);
    for (const auto& downstream : op2downstreams[cur]) {
      degrees[downstream]--;
      if (degrees[downstream] == 0) {
        q.push(downstream);
      }
    }
  }
  CHECK_EQ(result.size(), op_exprs.size());
  std::vector<ir::Expr> sorted_result;
  for (const auto& op : result) {
    sorted_result.push_back(*op);
  }
  return sorted_result;
}

std::vector<ir::Expr> GetExprFromPattern(
    const StmtPattern<BackendStage>& pattern) {
  const auto& results = std::visit(IrExprGetter(), pattern.variant());
  return TopoSort(results);
}

template <>
ExprPromise<BackendStage> InitExprPromiseImpl(
    const TrivialPattern<BackendStage>& pattern, pir::Value anchor) {
  return ExprPromise<BackendStage>(anchor, pattern.trivial_op);
}

template <>
ExprPromise<BackendStage> InitExprPromiseImpl(
    const ReducePattern<BackendStage>& pattern, pir::Value anchor) {
  return ExprPromise<BackendStage>(anchor, pattern.reduce_op);
}

template <>
TrivialPattern<BackendStage> RecoverAnchorPatternToTrivial(
    const AnchorPattern<BackendStage>& anchor_pattern) {
  PADDLE_ENFORCE(anchor_pattern.anchor_state.promise.size() == 1 &&
                     std::holds_alternative<TrivialOp>(
                         anchor_pattern.anchor_state.promise[0].root_fusion_op),
                 phi::errors::PreconditionNotMet(
                     "Can only recover AnchorPatter whose anchor_state size is "
                     "1 (exact %d) and holds TrivialOp",
                     anchor_pattern.anchor_state.promise.size()));

  return TrivialPattern<BackendStage>(
      anchor_pattern.ops(),
      anchor_pattern.anchor().defining_op(),
      std::get<TrivialOp>(
          anchor_pattern.anchor_state.promise[0].root_fusion_op));
}

void RunRenameInstr(const FusionInstrPtr& instr, FusionInterpreter* scope);
void RunCombineInstr(const FusionInstrPtr& instr, FusionInterpreter* scope);
void RunReturnInstr(const FusionInstrPtr& instr, FusionInterpreter* scope);
void RunInitPatternInstr(const FusionInstrPtr& instr, FusionInterpreter* scope);
void RunTrivialInlineInstr(const FusionInstrPtr& instr,
                           FusionInterpreter* scope);
void RunTmpTransformInstr(const FusionInstrPtr& instr,
                          FusionInterpreter* scope);
void RunTmpTransformWithFakeReduceIterInstr(const FusionInstrPtr& instr,
                                            FusionInterpreter* scope);
void RunAnchorTransformInstr(const FusionInstrPtr& instr,
                             FusionInterpreter* scope);

PatternExpr FusionInterpreter::Run() {
  for (const auto instr : tracker->instructions_) {
    switch (instr->type()) {
      case T_Rename:
        RunRenameInstr(instr, this);
        break;
      case T_Combine:
        RunCombineInstr(instr, this);
        break;
      case T_InitPattern:
        RunInitPatternInstr(instr, this);
        break;
      case T_TrivialInline:
        RunTrivialInlineInstr(instr, this);
        break;
      case T_TmpTransform:
        RunTmpTransformInstr(instr, this);
        break;
      case T_TmpTransformWithFakeReduceIter:
        RunTmpTransformWithFakeReduceIterInstr(instr, this);
        break;
      case T_AnchorTransform:
        RunAnchorTransformInstr(instr, this);
        break;
      case T_Return:
        auto ret_ptr = std::dynamic_pointer_cast<ReturnInstr>(instr);
        if (!ret_ptr) PADDLE_THROW("Non ReturnInstr return T_Return as type.");
        return scope[ret_ptr->ret_name_];
      default:
        PADDLE_THROW("Unsupported Fusion Instrution");
    }
  }
}

}  // namespace cinn::fusion
