// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/distributed/ps/table/tensor_accessor.h"
#include "Eigen/Dense"

namespace paddle {
namespace distributed {

int CommMergeAccessor::Initialize() { return 0; }

void CommMergeAccessor::SetTableInfo(AccessorInfo &info) {
  info.select_dim = SelectDim();
  info.select_size = SelectSize();
  info.update_dim = UpdateDim();
  info.update_size = UpdateSize();
  info.fea_dim = fea_dim();
}

size_t CommMergeAccessor::GetTableInfo(InfoKey key) {
  switch (key) {
    case SELECT_DIM:
      return SelectDim();
    case SELECT_SIZE:
      return SelectSize();
    case UPDATE_DIM:
      return UpdateDim();
    case UPDATE_SIZE:
      return UpdateSize();
    case FEA_DIM:
      return fea_dim();
    default:
      return 0;
  }
  return 0;
}

// pull value 维度
size_t CommMergeAccessor::SelectDim() { return _config.embedx_dim(); }

// pull value 各个维度的size
size_t CommMergeAccessor::SelectDimSize(size_t dim) { return sizeof(float); }

// pull value 各维度相加总size
size_t CommMergeAccessor::SelectSize() { return SelectDim() * sizeof(float); }

// push value 维度
size_t CommMergeAccessor::UpdateDim() { return _config.embedx_dim(); }

// push value 各个维度的size
size_t CommMergeAccessor::UpdateDimSize(size_t dim) { return sizeof(float); }

// push value 各维度相加总size
size_t CommMergeAccessor::UpdateSize() { return UpdateDim() * sizeof(float); }

// 判断该value 是否进行shrink
bool CommMergeAccessor::Shrink(float * /*value*/) { return false; }

// 判断该value 是否在save阶段dump,
// param作为参数用于标识save阶段，如downpour的xbox与batch_model
bool CommMergeAccessor::Save(float * /*value*/, int /*param*/) { return true; }

// keys不存在时，为values生成随机值
int32_t CommMergeAccessor::Create(float **value, size_t num) { return 0; }

// 从values中选取到select_values中
int32_t CommMergeAccessor::Select(float **select_values, const float **values,
                                  size_t num) {
  return 0;
}

// 将update_values聚合到一起
int32_t CommMergeAccessor::Merge(float **update_values,
                                 const float **other_update_values,
                                 size_t num) {
  Eigen::Map<Eigen::MatrixXf> u_mat(update_values[0], 1, num);
  Eigen::Map<const Eigen::MatrixXf> o_mat(other_update_values[0], 1, num);
  u_mat += o_mat;
  return 0;
}

// 将update_values聚合到一起，通过it.next判定是否进入下一个key
//  int32_t merge(float** update_values, iterator it);
// 将update_values更新应用到values中
int32_t CommMergeAccessor::Update(float **values, const float **update_values,
                                  size_t num) {
  return 0;
}

int CommMergeAccessor::SetWeight(float **values, const float **update_values,
                                 size_t num) {
  return 0;
}

}  // namespace distributed
}  // namespace paddle
