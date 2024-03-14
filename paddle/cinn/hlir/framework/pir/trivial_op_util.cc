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

#include "paddle/cinn/hlir/framework/pir/trivial_op_util.h"

#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/framework/compile_error.h"
#include "paddle/cinn/hlir/framework/pir/op_lowering_util.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/cinn/hlir/op/external_api_registry.h"
#include "paddle/cinn/hlir/pe/map_expr_to_ir.h"
#include "paddle/cinn/ir/dim.h"
#include "paddle/cinn/ir/group_schedule/base_group_scheduler.h"
#include "paddle/cinn/ir/group_schedule/st_shape_group_scheduler.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"
#include "paddle/cinn/ir/schedule/ir_schedule_util.h"
#include "paddle/cinn/lang/placeholder.h"
#include "paddle/cinn/optim/schedule_block_dce.h"
#include "paddle/cinn/optim/transform_gpu_forloop.h"
#include "paddle/common/ddim.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"

namespace cinn {
namespace hlir {
namespace framework {
namespace pir {
namespace trivial_fusion_detail {

namespace ComposeUtils {

std::vector<ir::Var> ExprVec2VarVec(const std::vector<ir::Expr>& in) {
  std::vector<ir::Var> out;
  for (auto& expr : in) {
    out.push_back(expr.as_var_ref());
  }
  return out;
}

std::vector<ir::Expr> VarVec2ExprVec(const std::vector<ir::Var>& in) {
  return std::vector<ir::Expr>(in.begin(), in.end());
}

std::vector<ir::Expr> GetEachTensorLoadExpr(const ir::Expr& body,
                                            const ir::Tensor& tensor) {
  VLOG(4) << "Start GetEachTensorLoadExpr: " << tensor;
  std::set<Expr> load_exprs = cinn::ir::ir_utils::CollectIRNodesWithoutTensor(
      body, [&tensor](const Expr* expr) {
        return expr->As<ir::Load>() && expr->As<ir::Load>()->is_addr_tensor() &&
               expr->As<ir::Load>()->tensor.as_tensor_ref()->name ==
                   tensor->name;
      });
  for (auto& t : load_exprs) {
    VLOG(4) << "GetEachTensorLoadExpr: " << t << " " << t.ptr();
  }
  return std::vector(load_exprs.begin(), load_exprs.end());
}

MappingTargetExprToDestExprMutator::MappingTargetExprToDestExprMutator(
    const ir::Expr& source, const ir::Expr& dest)
    : source_(source), dest_(dest) {}

void MappingTargetExprToDestExprMutator::operator()(Expr* expr) {
  IRMutator::Visit(expr, expr);
}

void MappingTargetExprToDestExprMutator::Visit(const ir::Load* load, Expr* op) {
  VLOG(4) << "SubstitudeTargetExprWithDestExpr: " << load << " vs "
          << source_.ptr();
  if (load == source_.ptr()) {
    VLOG(4) << "substitude find!";
    *op = dest_;
  } else {
    IRMutator::Visit(load, op);
  }
}
void MappingTargetExprToDestExprMutator::Visit(const ir::Store* store,
                                               Expr* op) {
  VLOG(4) << "SubstitudeTargetExprWithDestExpr: " << store << " vs "
          << source_.ptr();
  if (store == source_.ptr()) {
    VLOG(4) << "substitude find!";
    *op = dest_;
  } else {
    IRMutator::Visit(store, op);
  }
}
void MappingTargetExprToDestExprMutator::Visit(const ir::Reduce* reduce,
                                               Expr* op) {
  VLOG(4) << "SubstitudeTargetExprWithDestExpr: " << reduce << " vs "
          << source_.ptr();
  if (reduce == source_.ptr()) {
    VLOG(4) << "substitude find!";
    *op = dest_;
  } else {
    IRMutator::Visit(reduce, op);
  }
}

bool CheckIterEq(const std::vector<ir::Var>& up_iter,
                 const std::vector<ir::Var>& down_iter) {
  if (up_iter.size() != down_iter.size()) return false;

  for (int i = 0; i < up_iter.size(); ++i) {
    const ir::Var& up_iter_var = up_iter[i];
    const ir::Var& down_iter_var = down_iter[i];

    if (up_iter_var != down_iter_var) return false;
    if (up_iter_var->lower_bound.as_int64() !=
        down_iter_var->lower_bound.as_int64())
      return false;
    if (up_iter_var->upper_bound.as_int64() !=
        down_iter_var->upper_bound.as_int64())
      return false;
  }
  return true;
}

ir::Expr CopyedReplaceExpr(const Expr& source,
                           const std::vector<Var>& replaced,
                           const std::vector<Expr>& candidates) {
  VLOG(4) << "Copyed Replace Expr Start";
  CHECK_EQ(replaced.size(), candidates.size())
      << "In ReplaceExpr, the size of Vars to be replaced must be equal to "
         "the "
         "size of cadidate Exprs! Please check.";
  auto copyed_source = ir::ir_utils::IRCopy(source);
  if (replaced.empty()) return copyed_source;
  std::map<Var, Expr, ir::CompVar> replacing_map;
  for (int i = 0; i < replaced.size(); ++i) {
    // If the Var to be replaced is equal to the candidate, we skip it.
    if (candidates[i].is_var() && candidates[i].as_var_ref() == replaced[i])
      continue;
    replacing_map[replaced[i]] = candidates[i];
  }
  ir::MappingVarToExprMutator mapper(replacing_map);
  mapper(&copyed_source);
  VLOG(4) << "Copyed Replace Expr End";
  return copyed_source;
}

void SubstitudeTargetExprWithDestExpr(const ir::Expr& source,
                                      const ir::Expr& dest,
                                      ir::Expr* body) {
  VLOG(4) << "Start SubstitudeTargetExprWithDestExpr";
  MappingTargetExprToDestExprMutator mapper(source, dest);
  mapper(body);
  VLOG(4) << "End SubstitudeTargetExprWithDestExpr";
}

ir::Expr SubstitudeIndexVector(const Expr& source,
                               const std::vector<Var>& load_vars,
                               const std::vector<ir::Expr>& indices) {
  return CopyedReplaceExpr(source, load_vars, indices);
}
}  // namespace ComposeUtils

namespace SearchUtils {

using ExprSet = std::vector<ir::Expr>;
using Func = std::function<ExprSet(const ir::Expr& x)>;
Mapping::Mapping(Func f, std::string s) {
  f_ = f;
  name = s;
}
ExprSet Mapping::operator()(const ir::Expr& x) const { return f_(x); }
ir::Expr Mapping::GetSingle(const ir::Expr& x) const {
  Mapping call = (*this) * Mapping::GetIdentity();
  const auto& o = call.operator()(x);
  if (o.size() != 1) {
    PADDLE_THROW("Try to get single result, but we get %d.", o.size());
  }
  return *o.begin();
}
Mapping Mapping::operator*(Mapping x) const {
  auto new_f = [self = *this, x = x](const ir::Expr& e) -> ExprSet {
    const auto& rs = self.f_(e);
    VLOG(6) << "Mapping Info : " << self.name;
    VLOG(6) << "        Inputs  :" << e;
    for (const auto& r : rs) {
      VLOG(6) << "      Outputs : \n" << r;
    }
    std::vector<ir::Expr> res;
    for (const auto& r : rs) {
      const auto& x_res = x.f_(r);
      res.insert(res.begin(), x_res.begin(), x_res.end());
    }
    return res;
  };
  return Mapping(std::function(new_f), x.name + "*" + this->name);
}
Mapping Mapping::GetIdentity() {
  return Mapping([](const ir::Expr& e) { return std::vector<ir::Expr>{e}; },
                 "identity");
}

Mapping Identity = Mapping::GetIdentity();

Mapping Store2Value = Mapping(
    [](const ir::Expr& e) -> ExprSet {
      if (e.As<ir::Store>()) {
        return {e.As<ir::Store>()->value};
      }
      return {};
    },
    "Store2Value");

Mapping Realizer2ScheduleBlock = Mapping(
    [](const ir::Expr& e) -> ExprSet {
      if (e.As<ir::ScheduleBlockRealize>()) {
        return {e.As<ir::ScheduleBlockRealize>()->schedule_block};
      }
      return {};
    },
    "Realizer2ScheduleBlock");

Mapping ScheduleBlock2Body = Mapping(
    [](const ir::Expr& e) -> ExprSet {
      if (e.As<ir::ScheduleBlock>()) {
        return {e.As<ir::ScheduleBlock>()->body};
      }
      return {};
    },
    "ScheduleBlock2Body");

Mapping ScheduleBlockRealizeNotRoot = FilterMaker(
    [](const ir::Expr& e) -> bool {
      return (e.As<ir::ScheduleBlockRealize>() &&
              e.As<ir::ScheduleBlockRealize>()
                      ->schedule_block.As<ir::ScheduleBlock>()
                      ->name.find("root") == std::string::npos);
    },
    "ScheduleBlockRealizeNotRoot");

Mapping ScheduleBlockRealizeIsNotInit = FilterMaker(
    [](const ir::Expr& e) -> bool {
      return (e.As<ir::ScheduleBlockRealize>() &&
              e.As<ir::ScheduleBlockRealize>()
                      ->schedule_block.As<ir::ScheduleBlock>()
                      ->name.find("__reduce_init") == std::string::npos);
    },
    "ScheduleBlockRealizeIsNotInit");

Mapping ScheduleBlockRealizeIsInit = FilterMaker(
    [](const ir::Expr& e) -> bool {
      return (e.As<ir::ScheduleBlockRealize>() &&
              e.As<ir::ScheduleBlockRealize>()
                      ->schedule_block.As<ir::ScheduleBlock>()
                      ->name.find("__reduce_init") != std::string::npos);
    },
    "ScheduleBlockRealizeIsInit");

Mapping IsFor = FilterMaker(
    [](const ir::Expr& e) -> bool { return e.As<ir::For>(); }, "IsFor");

Mapping ChildScheduleBlocks =
    Collector([](const ir::Expr* e) { return e->As<ir::ScheduleBlock>(); },
              "ChildScheduleBlocks");

Mapping ChildScheduleBlockRealizes =
    Collector(
        [](const ir::Expr* e) { return e->As<ir::ScheduleBlockRealize>(); },
        "ChildScheduleBlockRealizes") *
    ScheduleBlockRealizeNotRoot;

Mapping IsForIterVar(const ir::Var& var) {
  return FilterMaker(
      [var = var](const ir::Expr& e) -> bool {
        return e.As<ir::For>() && e.As<ir::For>()->loop_var == var;
      },
      "IsForIterVar");
}

Mapping For2Min =
    Mapping([](const ir::Expr& e) -> ExprSet { return {e.As<ir::For>()->min}; },
            "For2Min");

Mapping For2Max = Mapping(
    [](const ir::Expr& e) -> ExprSet { return {e.As<ir::For>()->extent}; },
    "For2Max");

Mapping ChildStores = Collector(
    [](const ir::Expr* e) { return e->As<ir::Store>(); }, "ChildStores");

Mapping ChildTensorLoads = Collector(
    [](const ir::Expr* e) {
      return e->As<ir::Load>() && e->As<ir::Load>()->is_addr_tensor();
    },
    "ChildLoads");

Mapping ChildTensorStores = Collector(
    [](const ir::Expr* e) {
      return e->As<ir::Load>() && e->As<ir::Store>()->is_addr_tensor();
    },
    "ChildTensorStores");

Mapping FilterLoadByTensor(const ir::Tensor& tensor) {
  return FilterMaker(
      [tensor = tensor](const ir::Expr& e) -> bool {
        return e.As<ir::Load>() &&
               e.As<ir::Load>()->tensor.as_tensor_ref()->name == tensor->name;
      },
      "FilterLoadByTensor(" + tensor->name + ")");
}

Mapping ChildFors =
    Collector([](const ir::Expr* e) { return e->As<ir::For>(); }, "ChildFors");

Mapping FindFather(const ir::Expr& root) {
  const auto& f = [&](const auto& child) -> ExprSet {
    Mapping find_child =
        Collector([child](const ir::Expr* e) { return *e == child; });
    const auto& father_collector = Collector(
        [&](const ir::Expr* current) { return !find_child(*current).empty(); });
    return father_collector(root);
  };
  return Mapping(f, "FindFather");
}
}  // namespace SearchUtils

namespace TransformerUtils {
using TransformFunc = std::function<ir::Expr(ir::Expr)>;

Transformer::Transformer(TransformFunc f) { f_ = f; }
ir::Expr Transformer::operator()(const ir::Expr& x) const { return f_(x); }
Transformer Transformer::operator*(const Transformer& x) const {
  auto new_f = [self = *this, x = x](const ir::Expr& e) -> ir::Expr {
    const auto& rs = self.f_(e);
    return x.f_(rs);
  };
  return Transformer(std::function(new_f));
}

Transformer Identity = Transformer([](const ir::Expr& e) { return e; });
Transformer WrapForTransformer(const ir::Var& v) {
  const auto& f = [=](const ir::Expr& e) -> ir::Expr {
    auto block = e;
    if (!block.As<ir::Block>()) {
      block = ir::Block::Make({e});
    }
    return ir::For::Make(v,
                         v->lower_bound,
                         v->upper_bound,
                         ir::ForType::Serial,
                         ir::DeviceAPI::Host,
                         block);
  };
  return Transformer(f);
}

Transformer WrapForsTransformer(const std::vector<ir::Var>& vs) {
  const auto& f = [&](const ir::Expr& e) -> ir::Expr {
    Transformer t = Identity;
    for (const auto& v : vs) {
      t = WrapForTransformer(v) * t;
    }
    return t(e);
  };
  return Transformer(f);
}

Transformer ChangeTensorLoadTransformer(const ir::Tensor& tensor,
                                        const ir::Expr dst_load) {
  const auto& f = [&](const ir::Expr& e) -> ir::Expr {
    auto copied_e = ir::ir_utils::IRCopy(e);
    const auto& load = (SearchUtils::ChildTensorLoads *
                        SearchUtils::FilterLoadByTensor(tensor))
                           .GetSingle(copied_e);
    ComposeUtils::MappingTargetExprToDestExprMutator(load, dst_load)(&copied_e);
    return copied_e;
  };
  return Transformer(f);
}

void ReplaceTarget(ir::Expr* e, const ir::Expr& t, const ir::Expr dst) {
  ComposeUtils::MappingTargetExprToDestExprMutator(t, dst)(e);
}

Transformer WrapStoreTransformer(const ir::Tensor& tensor,
                                 const std::vector<ir::Expr>& indices) {
  const auto& f = [=](const ir::Expr& e) -> ir::Expr {
    return ir::Store::Make(tensor, e, indices);
  };
  return Transformer(f);
}

std::vector<ir::Var> CreateInnerBlockVars(
    const std::vector<ir::Var>& block_vars) {
  int i = 0;
  std::vector<ir::Var> vars;
  for (const auto& v : block_vars) {
    vars.emplace_back("inner_block_" + std::to_string(i++));
  }
  return vars;
}

Transformer ChangeVarTransformer(const std::vector<ir::Var>& target_vars,
                                 const std::vector<ir::Var>& dest_vars) {
  const auto& f = [=](const ir::Expr& e) -> ir::Expr {
    return ComposeUtils::CopyedReplaceExpr(
        e,
        target_vars,
        std::vector<ir::Expr>(dest_vars.begin(), dest_vars.end()));
  };
  return Transformer(f);
}

Transformer SubstitudeByScheduleBlockRealize(const ir::Expr& realize) {
  const auto& f = [=](const ir::Expr& e) -> ir::Expr {
    const auto& iter_values =
        realize.As<ir::ScheduleBlockRealize>()->iter_values;
    const auto& iter_vars = realize.As<ir::ScheduleBlockRealize>()
                                ->schedule_block.As<ir::ScheduleBlock>()
                                ->iter_vars;
    return TransformerUtils::ChangeVarTransformer(
        iter_vars, ComposeUtils::ExprVec2VarVec(iter_values))(e);
  };
  return Transformer(f);
}

Transformer WrapScheduleRealizer(const std::vector<ir::Var>& block_vars,
                                 const std::string& tensor_name) {
  const auto& f = [=](const ir::Expr& e) -> ir::Expr {
    if (e.As<ir::ScheduleBlock>()) {
      PADDLE_THROW("please input a non-schedule block expr.");
    }
    const auto& inner_block_var = CreateInnerBlockVars(block_vars);
    const auto& replaced_e =
        ChangeVarTransformer(block_vars, inner_block_var)(e);
    const auto& schedule_block = ir::ScheduleBlock::Make(
        inner_block_var, {}, {}, tensor_name, replaced_e);
    const auto& schedule_realizer = ir::ScheduleBlockRealize::Make(
        std::vector<ir::Expr>(block_vars.begin(), block_vars.end()),
        schedule_block);
    return schedule_realizer;
  };
  return Transformer(f);
}
}  // namespace TransformerUtils

std::vector<OpPatternKind> GetOpPatternKindVector(
    const std::vector<::pir::Operation*>& ops) {
  const auto& op_pattern_map =
      Operator::GetAttrs<cinn::hlir::framework::OpPatternKind>("OpPattern");
  std::vector<OpPatternKind> op_patterns;
  const auto ConvertToPattern = [&op_pattern_map](const ::pir::Operation* op) {
    const std::string cinn_op_name = CompatibleInfo::OpName(*op);
    const hlir::framework::Operator* cinn_op = Operator::Get(cinn_op_name);
    return op_pattern_map[cinn_op];
  };
  std::transform(ops.begin(),
                 ops.end(),
                 std::back_inserter(op_patterns),
                 ConvertToPattern);
  return op_patterns;
}

bool IsTrivialKind(OpPatternKind kind) {
  return kind == OpPatternKind::kElementWise ||
         kind == OpPatternKind::kBroadcast || kind == OpPatternKind::kInjective;
}

void CheckFusionInputValid(const std::vector<ir::Expr>& op_compute_bodies,
                           const std::vector<OpPatternKind>& op_patterns) {
  if (VLOG_IS_ON(4)) {
    for (const auto& func : op_compute_bodies) {
      VLOG(4) << "TrivialOpFusion: {FuncBody is} :" << func;
    }
    for (const auto& op_ptn : op_patterns) {
      VLOG(4) << "OpPattern is :" << op_ptn;
    }
  }
  VLOG(4) << "      op_patterns.size() = " << op_compute_bodies.size();
  VLOG(4) << "op_compute_bodies.size() = " << op_patterns.size();
  PADDLE_ENFORCE_EQ(
      op_patterns.size(), op_compute_bodies.size(), "ops and  size not equal");
}

}  // namespace trivial_fusion_detail
}  // namespace pir
}  // namespace framework
}  // namespace hlir
}  // namespace cinn
