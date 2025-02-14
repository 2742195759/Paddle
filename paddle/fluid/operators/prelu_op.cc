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

#include "paddle/fluid/operators/prelu_op.h"

#include <memory>
#include <string>

namespace paddle {
namespace operators {

framework::OpKernelType innerGetKernelTypeForVar(
    const Tensor &tensor, const framework::OpKernelType &expected_kernel_type) {
#ifdef PADDLE_WITH_MKLDNN
  auto isOneDNNKernelChosen =
      (expected_kernel_type.data_layout_ == framework::DataLayout::kMKLDNN);
  auto isNotOneDNNTensor = (tensor.layout() != framework::DataLayout::kMKLDNN);
  auto isModelNHWC =
      (paddle::platform::MKLDNNDeviceContext::tls()
           .get_cur_paddle_data_layout() == framework::DataLayout::kNHWC);
  // All inputs (including alpha) need shape rotating
  if (isOneDNNKernelChosen && isNotOneDNNTensor && isModelNHWC) {
    return framework::OpKernelType(expected_kernel_type.data_type_,
                                   tensor.place(),
                                   framework::DataLayout::kNHWC);
  }
#endif
  return framework::OpKernelType(expected_kernel_type.data_type_,
                                 tensor.place(), tensor.layout());
}

class PReluOp : public framework::OperatorWithKernel {
 public:
  PReluOp(const std::string &type, const framework::VariableNameMap &inputs,
          const framework::VariableNameMap &outputs,
          const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "prelu");
    OP_INOUT_CHECK(ctx->HasInput("Alpha"), "Input", "Alpha", "prelu");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "prelu");

    auto x_dim = ctx->GetInputDim("X");
    std::string mode = ctx->Attrs().Get<std::string>("mode");
    if (mode == "all") {
      PADDLE_ENFORCE_EQ(phi::product(ctx->GetInputDim("Alpha")), 1,
                        platform::errors::InvalidArgument(
                            "For mode 'all', size of weight Alpha must be one. "
                            "But recevied alpha's size: %d.",
                            product(ctx->GetInputDim("Alpha"))));
    } else if (mode == "channel") {
      auto x_rank = x_dim.size();
      PADDLE_ENFORCE_GE(x_rank, 2,
                        platform::errors::InvalidArgument(
                            "For mode 'channel', rank of input X must be "
                            "equal or larger than 2. But recevied X's "
                            "rank: %d",
                            x_rank));
      const std::string data_format_str =
          ctx->Attrs().Get<std::string>("data_format");
      PADDLE_ENFORCE_EQ(data_format_str == "NCHW" || data_format_str == "NHWC",
                        true,
                        platform::errors::InvalidArgument(
                            "For mode 'channel', data_format must be one of "
                            "NCHW and NHWC. But recevied data_format: %s",
                            data_format_str));
      if (data_format_str == "NCHW" || ctx->IsRunMKLDNNKernel()) {
        PADDLE_ENFORCE_EQ(
            product(ctx->GetInputDim("Alpha")) == x_dim[1], true,
            platform::errors::InvalidArgument(
                "For mode 'channel', size of weight Alpha must be "
                "equal to the number of channels of input(x). But "
                "recevied alpha's size: %d, x_dim[1]: %d",
                product(ctx->GetInputDim("Alpha")), x_dim[1]));
      } else {
        PADDLE_ENFORCE_EQ(
            product(ctx->GetInputDim("Alpha")) == x_dim[x_rank - 1], true,
            platform::errors::InvalidArgument(
                "For mode 'channel', size of weight Alpha must be "
                "equal to the number of channels of input(x). But "
                "recevied alpha's size: %d, x_dim[%d]: %d",
                product(ctx->GetInputDim("Alpha")), x_rank - 1,
                x_dim[x_rank - 1]));
      }

    } else if (mode == "element") {
      auto alpha_dim = ctx->GetInputDim("Alpha");
      auto alpha_rank = alpha_dim.size();
      auto x_rank = x_dim.size();
      PADDLE_ENFORCE_GE(x_rank, 1,
                        platform::errors::InvalidArgument(
                            "For mode 'element', rank of input X must be "
                            "equal or larger than 2. But recevied X's "
                            "rank: %d",
                            x_rank));
      PADDLE_ENFORCE_EQ(
          alpha_rank, x_rank,
          platform::errors::InvalidArgument(
              "For mode 'element', rank of weight Alpha must be ",
              "equal to the rank of input(x). But recevied alpha's rank: %d, "
              "x's rank: %d.",
              alpha_rank, x_rank));
      size_t x_product = 1;
      size_t alpha_product = 1;
      for (int64_t i = x_rank - 1; i > 0; i--) {
        x_product *= x_dim[i];
        alpha_product *= alpha_dim[i];
      }
      PADDLE_ENFORCE_EQ(
          alpha_product, x_product,
          platform::errors::InvalidArgument(
              "For mode 'element', the size of weight Alpha must be "
              "equal to the size of input(x). But recevied alpha's size: %d, "
              "x's size: %d.",
              alpha_product, x_product));
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Attr(mode) of prelu must be one of 'all', 'channel', or 'element'. "
          "But recevied "
          "mode: '%s'.",
          mode));
    }
    ctx->ShareDim("X", /*->*/ "Out");
    ctx->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input_data_type =
        framework::OperatorWithKernel::IndicateVarDataType(ctx, "X");

#ifdef PADDLE_WITH_MKLDNN
    if (this->CanMKLDNNBeUsed(ctx, input_data_type)) {
      return framework::OpKernelType(input_data_type, ctx.GetPlace(),
                                     framework::DataLayout::kMKLDNN,
                                     framework::LibraryType::kMKLDNN);
    }
#endif
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string &var_name, const Tensor &tensor,
      const framework::OpKernelType &expected_kernel_type) const {
    return innerGetKernelTypeForVar(tensor, expected_kernel_type);
  }
};

class PReluOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input tensor of prelu operator.");
    AddInput("Alpha", "The alpha weight of prelu operator.");
    AddOutput("Out", "The output tensor of prelu operator.");
    AddComment(R"DOC(
PRelu Operator.
The equation is:
$$
f(x) =
\begin{cases}
\alpha * x, \quad  \text{if} \ x < 0 \\
x,         \qquad  \text{if} \ x >= 0
\end{cases}
$$
The input `X` can carry the LoD (Level of Details) information,
or not. And the output shares the LoD information with input `X`.
There are modes:
  all: all elements share same weight
  channel: elements in a channel share same weight
  element: each element has a weight
)DOC");
    AddAttr<std::string>("mode", "The mode for inputs to share weights.")
        .SetDefault("all");
    AddAttr<std::string>("data_format",
                         "Data format that specifies the layout of input")
        .SetDefault("NCHW");
    AddAttr<bool>("use_mkldnn",
                  "(bool, default false) Only used in mkldnn kernel")
        .SetDefault(false)
        .AsExtra();
    AddAttr<std::string>(
        "mkldnn_data_type",
        "(string, default \"float32\"). Data type of mkldnn kernel")
        .SetDefault("float32")
        .InEnum({"float32", "bfloat16"})
        .AsExtra();
    AddAttr<bool>("is_test",
                  "(bool, default false) Set to true for inference only, false "
                  "for training. Some layers may run faster when this is true.")
        .SetDefault(false)
        .AsExtra();
  }
};

// The operator to calculate gradients of a prelu operator.
class PReluGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "prelu");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   "Out@GRAD", "prelu");

    auto x_grad_name = framework::GradVarName("X");
    auto alpha_grad_name = framework::GradVarName("Alpha");

    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, ctx->GetInputDim("X"));
    }
    if (ctx->HasOutput(alpha_grad_name)) {
      ctx->SetOutputDim(alpha_grad_name, ctx->GetInputDim("Alpha"));
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input_data_type =
        framework::OperatorWithKernel::IndicateVarDataType(ctx, "X");

#ifdef PADDLE_WITH_MKLDNN
    if (this->CanMKLDNNBeUsed(ctx, input_data_type)) {
      return framework::OpKernelType(input_data_type, ctx.GetPlace(),
                                     framework::DataLayout::kMKLDNN,
                                     framework::LibraryType::kMKLDNN);
    }
#endif
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string &var_name, const Tensor &tensor,
      const framework::OpKernelType &expected_kernel_type) const {
    return innerGetKernelTypeForVar(tensor, expected_kernel_type);
  }
};

template <typename T>
class PReluGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("prelu_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("Alpha", this->Input("Alpha"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetOutput(framework::GradVarName("Alpha"), this->InputGrad("Alpha"));
    op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(prelu, ops::PReluOp, ops::PReluOpMaker,
                  ops::PReluGradOpMaker<paddle::framework::OpDesc>,
                  ops::PReluGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(prelu_grad, ops::PReluGradOp);
REGISTER_OP_CPU_KERNEL(
    prelu, ops::PReluKernel<paddle::platform::CPUDeviceContext, float>,
    ops::PReluKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    prelu_grad, ops::PReluGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::PReluGradKernel<paddle::platform::CPUDeviceContext, double>);
