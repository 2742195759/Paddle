/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/gdb_tools.h"

namespace paddle {
namespace framework {
namespace gdb{
  void PrintTensor(Variable* v){
    auto & t = v->Get<LoDTensor>();
    std::cout << t << std::endl;
  }

  std::string String(const char * n){
    return std::string(n);
  }

  void func_keeper(void*) {
    return ;
  }

  template <class T>
  void Print(T*);

  void PrintRuntimeContext(RuntimeContext* rc){
    std::cout << "RuntimeContext:" << std::endl;
    std::cout << "-   Inputs    :" << std::endl;
    for (const auto & pair : rc->inputs) {
      std::cout << pair.first << ": (vector<Variable*>) " ;
      for (const auto& v: pair.second){
        std::cout << v << "  " ;
      }
      std::cout << std::endl;
    }
    std::cout << "-   Outputs   :" << std::endl;
    for (const auto & pair : rc->outputs) {
      std::cout << pair.first << ": (vector<Variable*>) " ;
      for (const auto& v: pair.second){
        std::cout << v << "  " ;
      }
      std::cout << std::endl;
    }
  }
}
}  // namespace framework
}  // namespace paddle
