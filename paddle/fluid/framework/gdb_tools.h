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

#include "paddle/fluid/framework/executor.h"

#include <memory>

#include "paddle/fluid/framework/feed_fetch_method.h"
#include "paddle/fluid/framework/trainer_desc.pb.h"
#include "paddle/fluid/framework/trainer_factory.h"
#include "paddle/fluid/operators/controlflow/conditional_block_op_helper.h"
#include "paddle/fluid/operators/controlflow/recurrent_op_helper.h"
#include "paddle/fluid/operators/controlflow/while_op_helper.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/profiler.h"
#include "paddle/fluid/platform/profiler/event_tracing.h"
#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif
#include "paddle/fluid/framework/executor_gc_helper.h"
#include <stdarg.h>
#include <stdio.h>

DECLARE_bool(benchmark);
DECLARE_bool(use_mkldnn);

#define FUNCTION_KEEPER \
paddle::framework::gdb::func_keeper((void*)(paddle::framework::gdb::ToString)); \
paddle::framework::gdb::func_keeper((void*)(paddle::framework::gdb::PrintTensor)); 

namespace paddle {
namespace framework {
namespace gdb{
  void PrintTensor(Variable* v);
  std::string ToString(const char * n);

  void func_keeper(void *);
  //struct SymbolKeeper{
    //SymbolKeeper(int count, ...){
      //va_list args;
      //va_start (args, count);
      //for (int i=0;i<count;++i){
        //void* arg = va_arg(args, void*);
        //ptr[i] = arg;
      //}
      //va_end(args);
    //}
    //void* ptr[100];
  //};
  //extern struct SymbolKeeper symbol_keeper;
}
}  // namespace framework
}  // namespace paddle

