/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "paddle/fluid/platform/complex.h"
#include "paddle/phi/core/utils/array.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"

namespace paddle {
namespace operators {

// Define the binary functors used in elementwise ops.
// Note: InverseXxxFunctor is needed when calling ElementwiseComputeEx on CPU.

// Add
template <typename T>
using AddFunctor = phi::funcs::AddFunctor<T>;

template <typename T>
using InverseAddFunctor = phi::funcs::InverseAddFunctor<T>;

// Subtract
template <typename T>
using SubFunctor = phi::funcs::SubtractFunctor<T>;

template <typename T>
using InverseSubFunctor = phi::funcs::InverseSubtractFunctor<T>;

// Multiply
template <typename T>
using MulFunctor = phi::funcs::MultiplyFunctor<T>;

template <typename T>
using InverseMulFunctor = phi::funcs::InverseMultiplyFunctor<T>;

// Divide
template <typename T>
using DivFunctor = phi::funcs::DivideFunctor<T>;

template <typename T>
using InverseDivFunctor = phi::funcs::InverseDivideFunctor<T>;

// Floor Divide
template <typename T>
struct FloorDivFunctor {
  inline HOSTDEVICE T operator()(const T a, const T b) const {
    PADDLE_ENFORCE(b != 0, DIV_ERROR_INFO);
    return static_cast<T>(std::trunc(a / b));
  }
};

template <typename T>
struct InverseFloorDivFunctor {
  inline HOSTDEVICE T operator()(const T a, const T b) const {
    PADDLE_ENFORCE(a != 0, DIV_ERROR_INFO);
    return static_cast<T>(std::trunc(b / a));
  }
};

#undef DIV_ERROR_INFO

// Maximum
template <typename T>
struct MaxFunctor {
  inline HOSTDEVICE T operator()(const T a, const T b) const {
    return a > b ? a : b;
  }
};

// Minmum
template <typename T>
struct MinFunctor {
  inline HOSTDEVICE T operator()(const T a, const T b) const {
    return a < b ? a : b;
  }
};

template <typename T>
using Complex = paddle::platform::complex<T>;

template <typename T>
struct MinGradXFunctor {
  inline HOSTDEVICE T operator()(const T x, const T y, const T dout) const {
    return dout * static_cast<T>(x < y);
  }
};
template <typename T>
struct MinGradYFunctor {
  inline HOSTDEVICE T operator()(const T x, const T y, const T dout) const {
    return dout * static_cast<T>(x >= y);
  }
};

template <typename InT, typename OutT>
struct MinGradXYFunctor {
  inline HOSTDEVICE phi::Array<OutT, 2> operator()(const InT x, const InT y,
                                                   const InT dout) {
    phi::Array<OutT, 2> outs;
    // dx = dout * (x < y)
    outs[0] = static_cast<OutT>(dout * static_cast<InT>(x < y));
    // dy = dout * (x >= y)
    outs[1] = static_cast<OutT>(dout * static_cast<InT>(x >= y));
    return outs;
  }
};

// Ternary compare
template <typename T>
struct MaxGradXFunctor {
  inline HOSTDEVICE T operator()(const T x, const T y, const T dout) const {
    return dout * static_cast<T>(x > y);
  }
};
template <typename T>
struct MaxGradYFunctor {
  inline HOSTDEVICE T operator()(const T x, const T y, const T dout) const {
    return dout * static_cast<T>(x <= y);
  }
};

template <typename InT, typename OutT>
struct MaxGradXYFunctor {
  inline HOSTDEVICE phi::Array<OutT, 2> operator()(const InT x, const InT y,
                                                   const InT dout) {
    phi::Array<OutT, 2> outs;
    // dx = dout * (x > y)
    outs[0] = static_cast<OutT>(dout * static_cast<InT>(x > y));
    // dy = dout * (x <= y)
    outs[1] = static_cast<OutT>(dout * static_cast<InT>(x <= y));
    return outs;
  }
};

}  // namespace operators
}  // namespace paddle
