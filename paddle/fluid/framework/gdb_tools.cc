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
  
  template <T>
  void RawPrint(const T & a, int space=0, bool new_line=false){
    std::cout << std::string(space, ' ');
    std::cout << a;
    if (new_line){
      std::cout << std::endl;
    }
  }

  void Print(const std::string& s, int space=0, bool new_line=false){
    RawPrint(s, space, new_line);
  }

  void Print(const Variable& v){
    if (!v.IsInitialized()) {
      std::cout << "Not IsInitialized";
    }
    auto & t = v.Get<LoDTensor>();
    std::cout << t ;
  }

  void Print(const Variable* v){
    std::cout << v;
  }

  std::string String(const char * n){
    return std::string(n);
  }

  void func_keeper(void*) {
    return ;
  }


  template <class T>
  void Print(const std::vector<T>&vec) {
    std::cout << "(@=" << vec.size() << ") [";
    for (const auto& v: vec){
      Print(v);
      std::cout << ", " ;
    }
    std::cout << "]";
  }


  template <class K, class V>
  void Print(const std::map<K, V>&mm) {
    std::cout << "(@" << mm.size() << ") {";
    for (const auto& pair: mm){
      Print(pair.first);
      std::cout << ": " ;
      Print(pair.second);
      std::cout << ", " ;
    }
    std::cout << "}";
  }

  void Print(const RuntimeContext& rc, int space=0){
    RawPrint("RuntimeContext:", space, true)
    RawPrint("Inputs:", space+4, true)
    Print(rc.inputs, space+8);
    RawPrint("Outputs:", space+4, true)
    Print(rc.outputs, space+8);
  }
}
}  // namespace framework
}  // namespace paddle
