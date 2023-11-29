#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import inspect
import unittest

import nets
import numpy as np
from decorator_helper import prog_scope
from test_imperative_base import new_program_scope

import paddle
import paddle.nn.functional as F
from paddle import base
from paddle.base import core, dygraph
from paddle.base.dygraph import to_variable
from paddle.base.framework import Program, default_main_program, program_guard
from paddle.incubate.layers.nn import (
    batch_fc,
    partial_concat,
    partial_sum,
    rank_attention,
    shuffle_batch,
)
from paddle.pir_utils import test_with_pir_api
from paddle.tensor import random


class LayerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.seed = 111

    @classmethod
    def tearDownClass(cls):
        pass

    def _get_place(self, force_to_use_cpu=False):
        # this option for ops that only have cpu kernel
        if force_to_use_cpu:
            return core.CPUPlace()
        else:
            if core.is_compiled_with_cuda():
                return core.CUDAPlace(0)
            return core.CPUPlace()

    @contextlib.contextmanager
    def static_graph(self):
        with new_program_scope():
            paddle.seed(self.seed)
            paddle.framework.random._manual_program_seed(self.seed)
            yield

    def get_static_graph_result(
        self, feed, fetch_list, with_lod=False, force_to_use_cpu=False
    ):
        exe = base.Executor(self._get_place(force_to_use_cpu))
        exe.run(paddle.static.default_startup_program())
        return exe.run(
            paddle.static.default_main_program(),
            feed=feed,
            fetch_list=fetch_list,
            return_numpy=(not with_lod),
        )

    @contextlib.contextmanager
    def dynamic_graph(self, force_to_use_cpu=False):
        with base.dygraph.guard(
            self._get_place(force_to_use_cpu=force_to_use_cpu)
        ):
            paddle.seed(self.seed)
            paddle.framework.random._manual_program_seed(self.seed)
            yield


class TestLayer(LayerTest):
    def test_custom_layer_with_kwargs(self):
        class CustomLayer(paddle.nn.Layer):
            def __init__(self, input_size, linear1_size=4):
                super().__init__()
                self.linear1 = paddle.nn.Linear(
                    input_size, linear1_size, bias_attr=False
                )
                self.linear2 = paddle.nn.Linear(
                    linear1_size, 1, bias_attr=False
                )

            def forward(self, x, do_linear2=False):
                ret = self.linear1(x)
                if do_linear2:
                    ret = self.linear2(ret)
                return ret

        with self.dynamic_graph():
            inp = np.ones([3, 3], dtype='float32')
            x = to_variable(inp)
            custom = CustomLayer(input_size=3, linear1_size=2)
            ret = custom(x, do_linear2=False)
            np.testing.assert_array_equal(ret.numpy().shape, [3, 2])
            ret = custom(x, do_linear2=True)
            np.testing.assert_array_equal(ret.numpy().shape, [3, 1])

    def test_dropout(self):
        inp = np.ones([3, 32, 32], dtype='float32')
        with self.static_graph():
            t = paddle.static.data(
                name='data',
                shape=[3, 32, 32],
                dtype='float32',
            )
            dropout = paddle.nn.Dropout(p=0.35)
            ret = dropout(t)
            ret2 = paddle.nn.functional.dropout(t, p=0.35)
            static_ret, static_ret2 = self.get_static_graph_result(
                feed={'data': inp}, fetch_list=[ret, ret2]
            )
        with self.dynamic_graph():
            t = to_variable(inp)
            dropout = paddle.nn.Dropout(p=0.35)
            dy_ret = dropout(t)
            dy_ret2 = paddle.nn.functional.dropout(t, p=0.35)
            dy_ret_value = dy_ret.numpy()
            dy_ret2_value = dy_ret2.numpy()

        np.testing.assert_array_equal(static_ret, static_ret2)
        np.testing.assert_array_equal(dy_ret_value, dy_ret2_value)
        np.testing.assert_array_equal(static_ret, dy_ret_value)

    def test_linear(self):
        inp = np.ones([3, 32, 32], dtype='float32')
        with self.static_graph():
            t = paddle.static.data(
                name='data', shape=[3, 32, 32], dtype='float32'
            )
            linear = paddle.nn.Linear(
                32,
                4,
                bias_attr=paddle.nn.initializer.Constant(value=1),
            )
            ret = linear(t)
            static_ret = self.get_static_graph_result(
                feed={'data': inp}, fetch_list=[ret]
            )[0]
        with self.dynamic_graph():
            t = to_variable(inp)
            linear = paddle.nn.Linear(
                32,
                4,
                bias_attr=paddle.nn.initializer.Constant(value=1),
            )
            dy_ret = linear(t)
            dy_ret_value = dy_ret.numpy()

        np.testing.assert_array_equal(static_ret, dy_ret_value)

        with self.static_graph():
            # the input of Linear must be Variable.
            def test_Variable():
                inp = np.ones([3, 32, 32], dtype='float32')
                linear = paddle.nn.Linear(
                    32,
                    4,
                    bias_attr=paddle.nn.initializer.Constant(value=1),
                )
                linear_ret1 = linear(inp)

            self.assertRaises(TypeError, test_Variable)

            # the input dtype of Linear must be float16 or float32 or float64
            # float16 only can be set on GPU place
            def test_type():
                inp = np.ones([3, 32, 32], dtype='int32')
                linear = paddle.nn.Linear(
                    32,
                    4,
                    bias_attr=paddle.nn.initializer.Constant(value=1),
                )
                linear_ret2 = linear(inp)

            self.assertRaises(TypeError, test_type)

    def test_cvm(self):
        inp = np.ones([10, 10], dtype='float32')
        arr = [[0.6931472, -1.904654e-09, 1, 1, 1, 1, 1, 1, 1, 1]] * 10
        cvm1 = np.array(arr, dtype='float32')
        cvm2 = np.ones([10, 8], dtype='float32')
        show_clk = np.ones([10, 2], dtype='float32')
        with self.static_graph():
            x = paddle.static.data(
                name='data',
                shape=[10, 10],
                dtype='float32',
            )
            u = paddle.static.data(
                name='show_click',
                shape=[10, 2],
                dtype='float32',
            )
            no_cvm = paddle.static.nn.continuous_value_model(x, u, True)
            static_ret1 = self.get_static_graph_result(
                feed={'data': inp, 'show_click': show_clk},
                fetch_list=[no_cvm],
            )[0]
        with self.static_graph():
            x = paddle.static.data(
                name='data',
                shape=[10, 10],
                dtype='float32',
            )
            u = paddle.static.data(
                name='show_click',
                shape=[10, 2],
                dtype='float32',
            )
            cvm = paddle.static.nn.continuous_value_model(x, u, False)
            static_ret2 = self.get_static_graph_result(
                feed={'data': inp, 'show_click': show_clk}, fetch_list=[cvm]
            )[0]
        np.testing.assert_allclose(static_ret1, cvm1, rtol=1e-5, atol=1e-06)
        np.testing.assert_allclose(static_ret2, cvm2, rtol=1e-5, atol=1e-06)

    def test_Flatten(self):
        inp = np.ones([3, 4, 4, 5], dtype='float32')
        with self.static_graph():
            t = paddle.static.data(
                name='data', shape=[3, 4, 4, 5], dtype='float32'
            )
            flatten = paddle.nn.Flatten()
            ret = flatten(t)
            static_ret = self.get_static_graph_result(
                feed={'data': inp}, fetch_list=[ret]
            )[0]
        with self.dynamic_graph():
            t = to_variable(inp)
            flatten = paddle.nn.Flatten()
            dy_ret = flatten(t)
            dy_ret_value = dy_ret.numpy()

        np.testing.assert_array_equal(static_ret, dy_ret_value)

        with self.static_graph():
            # the input of Linear must be Variable.
            def test_Variable():
                inp = np.ones([3, 32, 32], dtype='float32')
                linear = paddle.nn.Linear(
                    32,
                    4,
                    bias_attr=paddle.nn.initializer.Constant(value=1),
                )
                linear_ret1 = linear(inp)

            self.assertRaises(TypeError, test_Variable)

            # the input dtype of Linear must be float16 or float32 or float64
            # float16 only can be set on GPU place
            def test_type():
                inp = np.ones([3, 32, 32], dtype='int32')
                linear = paddle.nn.Linear(
                    32,
                    4,
                    bias_attr=paddle.nn.initializer.Constant(value=1),
                )
                linear_ret2 = linear(inp)

            self.assertRaises(TypeError, test_type)

    @test_with_pir_api
    def test_SyncBatchNorm(self):
        if core.is_compiled_with_cuda():
            with self.static_graph():
                t = paddle.static.data(
                    name='t', shape=[-1, 3, 5, 5], dtype='float32'
                )
                my_sync_bn = paddle.nn.SyncBatchNorm(3)
                ret = my_sync_bn(t)
                static_ret = self.get_static_graph_result(
                    feed={'t': np.ones([3, 3, 5, 5], dtype='float32')},
                    fetch_list=[ret],
                )[0]

            with self.dynamic_graph():
                t = np.ones([3, 3, 5, 5], dtype='float32')
                my_syncbn = paddle.nn.SyncBatchNorm(3)
                dy_ret = my_syncbn(to_variable(t))
                dy_ret_value = dy_ret.numpy()
            np.testing.assert_array_equal(static_ret, dy_ret_value)

    def test_relu(self):
        with self.static_graph():
            t = paddle.static.data(name='t', shape=[-1, 3, 3], dtype='float32')
            ret = F.relu(t)
            static_ret = self.get_static_graph_result(
                feed={'t': np.ones([3, 3], dtype='float32')}, fetch_list=[ret]
            )[0]

        with self.dynamic_graph():
            t = np.ones([3, 3], dtype='float32')
            dy_ret = F.relu(to_variable(t))
            dy_ret_value = dy_ret.numpy()

        np.testing.assert_allclose(static_ret, dy_ret_value, rtol=1e-05)

    def test_matmul(self):
        with self.static_graph():
            t = paddle.static.data(name='t', shape=[-1, 3, 3], dtype='float32')
            t2 = paddle.static.data(
                name='t2', shape=[-1, 3, 3], dtype='float32'
            )
            ret = paddle.matmul(t, t2)
            static_ret = self.get_static_graph_result(
                feed={
                    't': np.ones([3, 3], dtype='float32'),
                    't2': np.ones([3, 3], dtype='float32'),
                },
                fetch_list=[ret],
            )[0]

        with self.dynamic_graph():
            t = np.ones([3, 3], dtype='float32')
            t2 = np.ones([3, 3], dtype='float32')
            dy_ret = paddle.matmul(to_variable(t), to_variable(t2))
            dy_ret_value = dy_ret.numpy()

        np.testing.assert_allclose(static_ret, dy_ret_value, rtol=1e-05)

    def test_elementwise_math(self):
        n = np.ones([3, 3], dtype='float32')
        n2 = np.ones([3, 3], dtype='float32') * 1.1
        n3 = np.ones([3, 3], dtype='float32') * 2
        n4 = np.ones([3, 3], dtype='float32') * 3
        n5 = np.ones([3, 3], dtype='float32') * 4
        n6 = np.ones([3, 3], dtype='float32') * 5

        with self.static_graph():
            t = paddle.static.data(name='t', shape=[-1, 3, 3], dtype='float32')
            t2 = paddle.static.data(
                name='t2', shape=[-1, 3, 3], dtype='float32'
            )
            t3 = paddle.static.data(
                name='t3', shape=[-1, 3, 3], dtype='float32'
            )
            t4 = paddle.static.data(
                name='t4', shape=[-1, 3, 3], dtype='float32'
            )
            t5 = paddle.static.data(
                name='t5', shape=[-1, 3, 3], dtype='float32'
            )
            t6 = paddle.static.data(
                name='t6', shape=[-1, 3, 3], dtype='float32'
            )

            ret = paddle.add(t, t2)
            ret = paddle.pow(ret, t3)
            ret = paddle.divide(ret, t4)
            ret = paddle.subtract(ret, t5)
            ret = paddle.multiply(ret, t6)

            static_ret = self.get_static_graph_result(
                feed={'t': n, 't2': n2, 't3': n3, 't4': n4, 't5': n5, 't6': n6},
                fetch_list=[ret],
            )[0]

        with self.dynamic_graph():
            ret = paddle.add(to_variable(n), to_variable(n2))
            ret = paddle.pow(ret, to_variable(n3))
            ret = paddle.divide(ret, to_variable(n4))
            ret = paddle.subtract(ret, to_variable(n5))
            dy_ret = paddle.multiply(ret, to_variable(n6))
            dy_ret_value = dy_ret.numpy()

        np.testing.assert_allclose(static_ret, dy_ret_value, rtol=1e-05)

    def test_elementwise_minmax(self):
        n = np.ones([3, 3], dtype='float32')
        n2 = np.ones([3, 3], dtype='float32') * 2

        with self.dynamic_graph():
            min_ret = paddle.minimum(to_variable(n), to_variable(n2))
            max_ret = paddle.maximum(to_variable(n), to_variable(n2))
            min_ret_value = min_ret.numpy()
            max_ret_value = max_ret.numpy()

        np.testing.assert_allclose(n, min_ret_value, rtol=1e-05)
        np.testing.assert_allclose(n2, max_ret_value, rtol=1e-05)

    def test_conv2d_transpose(self):
        inp_np = np.arange(0, 24).reshape([2, 3, 2, 2]).astype('float32')
        with self.static_graph():
            img = paddle.static.data(
                name='pixel', shape=[-1, 3, 2, 2], dtype='float32'
            )
            out = paddle.static.nn.conv2d_transpose(
                input=img,
                num_filters=10,
                filter_size=27,
                act='sigmoid',
                bias_attr=paddle.nn.initializer.Constant(value=1),
            )
            static_rlt = self.get_static_graph_result(
                feed={'pixel': inp_np}, fetch_list=[out]
            )[0]
        with self.static_graph():
            img = paddle.static.data(
                name='pixel', shape=[-1, 3, 2, 2], dtype='float32'
            )
            conv2d_transpose = paddle.nn.Conv2DTranspose(
                3,
                10,
                27,
                bias_attr=paddle.nn.initializer.Constant(value=1),
            )
            out = conv2d_transpose(img)
            out = paddle.nn.functional.sigmoid(out)
            static_rlt2 = self.get_static_graph_result(
                feed={'pixel': inp_np}, fetch_list=[out]
            )[0]
        with self.dynamic_graph():
            conv2d_transpose = paddle.nn.Conv2DTranspose(
                3,
                10,
                27,
                bias_attr=paddle.nn.initializer.Constant(value=1),
            )
            dy_rlt = conv2d_transpose(to_variable(inp_np))
            dy_rlt = paddle.nn.functional.sigmoid(dy_rlt)
            dy_rlt_value = dy_rlt.numpy()
        np.testing.assert_allclose(static_rlt2, static_rlt, rtol=1e-05)
        np.testing.assert_allclose(dy_rlt_value, static_rlt2, rtol=1e-05)

        with self.dynamic_graph():
            images = np.ones([2, 3, 5, 5], dtype='float32')
            custom_weight = np.random.randn(3, 3, 2, 2).astype("float32")
            weight_attr = base.ParamAttr(
                initializer=paddle.nn.initializer.Assign(custom_weight)
            )
            conv2d1 = paddle.nn.Conv2DTranspose(3, 3, [2, 2])
            conv2d2 = paddle.nn.Conv2DTranspose(
                3,
                3,
                [2, 2],
                weight_attr=weight_attr,
            )
            dy_ret1 = conv2d1(to_variable(images))
            dy_ret2 = conv2d2(to_variable(images))
            self.assertFalse(np.array_equal(dy_ret1.numpy(), dy_ret2.numpy()))

            conv2d1_weight_np = conv2d1.weight.numpy()
            conv2d1_bias = conv2d1.bias
            self.assertFalse(
                np.array_equal(conv2d1_weight_np, conv2d2.weight.numpy())
            )
            conv2d2.weight.set_value(conv2d1_weight_np)
            np.testing.assert_array_equal(
                conv2d1_weight_np, conv2d2.weight.numpy()
            )
            conv2d2.bias.set_value(conv2d1_bias)
            dy_ret1 = conv2d1(to_variable(images))
            dy_ret2 = conv2d2(to_variable(images))
            np.testing.assert_array_equal(dy_ret1.numpy(), dy_ret2.numpy())

            conv2d2.weight = conv2d1.weight
            conv2d2.bias = conv2d1.bias
            np.testing.assert_array_equal(
                conv2d1.weight.numpy(), conv2d2.weight.numpy()
            )
            np.testing.assert_array_equal(
                conv2d1.bias.numpy(), conv2d2.bias.numpy()
            )

        with self.static_graph():
            # the input of Conv2DTranspose must be Variable.
            def test_Variable():
                images = np.ones([2, 3, 5, 5], dtype='float32')
                conv2d = paddle.nn.Conv2DTranspose(3, 3, [2, 2])
                conv2d_ret1 = conv2d(images)

            self.assertRaises(TypeError, test_Variable)

            # the input dtype of Conv2DTranspose must be float16 or float32 or float64
            # float16 only can be set on GPU place
            def test_type():
                images = paddle.static.data(
                    name='pixel', shape=[-1, 3, 5, 5], dtype='int32'
                )
                conv2d = paddle.nn.Conv2DTranspose(3, 3, [2, 2])
                conv2d_ret2 = conv2d(images)

            self.assertRaises(TypeError, test_type)

    def test_bilinear_tensor_product(self):
        def _test_static_specific(inp_np_x, inp_np_y):
            with self.static_graph():
                data_x = paddle.static.data(
                    name='x', shape=[1, 3], dtype="float32"
                )
                data_y = paddle.static.data(
                    name='y', shape=[1, 3], dtype="float32"
                )
                out = paddle.static.nn.common.bilinear_tensor_product(
                    data_x,
                    data_y,
                    6,
                    bias_attr=paddle.nn.initializer.Constant(value=1),
                    act='sigmoid',
                )

                static_rlt = self.get_static_graph_result(
                    feed={'x': inp_np_x, 'y': inp_np_y}, fetch_list=[out]
                )[0]

            return static_rlt

        def _test_static(inp_np_x, inp_np_y):
            with self.static_graph():
                data_x = paddle.static.data(
                    name='x', shape=[1, 3], dtype="float32"
                )
                data_y = paddle.static.data(
                    name='y', shape=[1, 3], dtype="float32"
                )
                btp = paddle.nn.Bilinear(
                    3,
                    3,
                    6,
                    bias_attr=paddle.nn.initializer.Constant(value=1),
                )
                out = btp(data_x, data_y)
                out = paddle.nn.functional.sigmoid(out)
                static_rlt2 = self.get_static_graph_result(
                    feed={'x': inp_np_x, 'y': inp_np_y}, fetch_list=[out]
                )[0]

            return static_rlt2

        def _test_dygraph_1(inp_np_x, inp_np_y):
            with self.dynamic_graph():
                btp = paddle.nn.Bilinear(
                    3,
                    3,
                    6,
                    bias_attr=paddle.nn.initializer.Constant(value=1),
                )
                dy_rlt = btp(
                    to_variable(inp_np_x),
                    to_variable(inp_np_y),
                )
                dy_rlt = paddle.nn.functional.sigmoid(dy_rlt)
                dy_rlt_value = dy_rlt.numpy()

            with self.dynamic_graph():
                btp2 = paddle.nn.Bilinear(3, 3, 6)
                dy_rlt2 = btp2(
                    to_variable(inp_np_x),
                    to_variable(inp_np_y),
                )
                dy_rlt2 = paddle.nn.functional.sigmoid(dy_rlt2)
                dy_rlt2_value = dy_rlt2.numpy()

            with self.static_graph():
                data_x2 = paddle.static.data(
                    name='x', shape=[1, 3], dtype="float32"
                )
                data_y2 = paddle.static.data(
                    name='y', shape=[1, 3], dtype="float32"
                )
                out2 = paddle.static.nn.common.bilinear_tensor_product(
                    data_x2, data_y2, 6, act='sigmoid'
                )

                static_rlt3 = self.get_static_graph_result(
                    feed={'x': inp_np_x, 'y': inp_np_y}, fetch_list=[out2]
                )[0]

            return dy_rlt_value, dy_rlt2_value, static_rlt3

        def _test_dygraph_2(inp_np_x, inp_np_y):
            with self.dynamic_graph():
                custom_weight = np.random.randn(6, 3, 3).astype("float32")
                weight_attr = base.ParamAttr(
                    initializer=paddle.nn.initializer.Assign(custom_weight)
                )
                btp1 = paddle.nn.Bilinear(3, 3, 6)
                btp2 = paddle.nn.Bilinear(3, 3, 6, weight_attr=weight_attr)
                dy_rlt1 = btp1(
                    to_variable(inp_np_x),
                    to_variable(inp_np_y),
                )
                dy_rlt1 = paddle.nn.functional.sigmoid(dy_rlt1)
                dy_rlt2 = btp2(
                    to_variable(inp_np_x),
                    to_variable(inp_np_y),
                )
                dy_rlt2 = paddle.nn.functional.sigmoid(dy_rlt2)
                self.assertFalse(
                    np.array_equal(dy_rlt1.numpy(), dy_rlt2.numpy())
                )
                btp2.weight.set_value(btp1.weight.numpy())
                btp2.bias.set_value(btp1.bias)
                dy_rlt1 = btp1(
                    to_variable(inp_np_x),
                    to_variable(inp_np_y),
                )
                dy_rlt2 = btp2(
                    to_variable(inp_np_x),
                    to_variable(inp_np_y),
                )
                np.testing.assert_array_equal(dy_rlt1.numpy(), dy_rlt2.numpy())

                btp2.weight = btp1.weight
                btp2.bias = btp1.bias
                np.testing.assert_array_equal(
                    btp1.weight.numpy(), btp2.weight.numpy()
                )
                np.testing.assert_array_equal(
                    btp1.bias.numpy(), btp2.bias.numpy()
                )

        inp_np_x = np.array([[1, 2, 3]]).astype('float32')
        inp_np_y = np.array([[4, 5, 6]]).astype('float32')

        static_rlt = _test_static_specific(inp_np_x, inp_np_y)
        static_rlt2 = _test_static(inp_np_x, inp_np_y)
        dy_rlt_value, dy_rlt2_value, static_rlt3 = _test_dygraph_1(
            inp_np_x, inp_np_y
        )
        np.testing.assert_array_equal(dy_rlt2_value, static_rlt3)
        np.testing.assert_array_equal(static_rlt2, static_rlt)
        np.testing.assert_array_equal(dy_rlt_value, static_rlt)

        with paddle.pir_utils.IrGuard():
            static_pir_result = _test_static(inp_np_x, inp_np_y)
        np.testing.assert_array_equal(static_pir_result, static_rlt)

    def test_embeding(self):
        inp_word = np.array([[[1]]]).astype('int64')
        dict_size = 20
        with self.static_graph():
            data_t = paddle.static.data(
                name='word', shape=[-1, 1], dtype='int64'
            )
            data_t.desc.set_need_check_feed(False)
            emb = paddle.static.nn.embedding(
                input=data_t.squeeze(-2),
                size=[dict_size, 32],
                param_attr='emb.w',
                is_sparse=False,
            )
            static_rlt = self.get_static_graph_result(
                feed={'word': inp_word}, fetch_list=[emb]
            )[0]
        with self.static_graph():
            data_t = paddle.static.data(
                name='word', shape=[-1, 1], dtype='int64'
            )
            data_t.desc.set_need_check_feed(False)
            emb2 = paddle.nn.Embedding(
                dict_size, 32, weight_attr='emb.w', sparse=False
            )
            emb_rlt = emb2(data_t)
            static_rlt2 = self.get_static_graph_result(
                feed={'word': inp_word}, fetch_list=[emb_rlt]
            )[0]
        with self.dynamic_graph():
            emb2 = paddle.nn.Embedding(
                dict_size, 32, weight_attr='emb.w', sparse=False
            )
            dy_rlt = emb2(to_variable(inp_word))
            dy_rlt_value = dy_rlt.numpy()

        np.testing.assert_allclose(static_rlt2[0], static_rlt)
        np.testing.assert_allclose(dy_rlt_value[0], static_rlt)

        with self.dynamic_graph():
            custom_weight = np.random.randn(dict_size, 32).astype("float32")
            weight_attr = base.ParamAttr(
                initializer=paddle.nn.initializer.Assign(custom_weight)
            )
            emb1 = paddle.nn.Embedding(dict_size, 32, sparse=False)
            emb2 = paddle.nn.Embedding(
                dict_size, 32, weight_attr=weight_attr, sparse=False
            )
            rep1 = emb1(to_variable(inp_word))
            rep2 = emb2(to_variable(inp_word))
            self.assertFalse(np.array_equal(emb1.weight.numpy(), custom_weight))
            np.testing.assert_array_equal(emb2.weight.numpy(), custom_weight)
            self.assertFalse(np.array_equal(rep1.numpy(), rep2.numpy()))
            emb2.weight.set_value(emb1.weight.numpy())
            rep2 = emb2(to_variable(inp_word))
            np.testing.assert_array_equal(rep1.numpy(), rep2.numpy())

            emb2.weight = emb1.weight
            np.testing.assert_array_equal(
                emb1.weight.numpy(), emb2.weight.numpy()
            )

    def test_one_hot(self):
        with self.dynamic_graph():
            label = base.dygraph.to_variable(np.array([[1], [1], [3], [0]]))
            one_hot_label1 = paddle.nn.functional.one_hot(label, 4)
            one_hot_label2 = paddle.nn.functional.one_hot(
                label, base.dygraph.to_variable(np.array([4]))
            )
            np.testing.assert_array_equal(
                one_hot_label1.numpy(), one_hot_label2.numpy()
            )

    def test_split(self):
        with self.dynamic_graph():
            input = base.dygraph.to_variable(np.random.random((3, 8, 5)))
            x0, x1 = paddle.split(input, num_or_sections=2, axis=1)
            x00, x11 = paddle.split(
                input,
                num_or_sections=2,
                axis=base.dygraph.to_variable(np.array([1])),
            )
            np.testing.assert_array_equal(x0.numpy(), x00.numpy())
            np.testing.assert_array_equal(x1.numpy(), x11.numpy())

    def test_topk(self):
        with self.dynamic_graph():
            input = base.dygraph.to_variable(np.random.random((13, 11)))
            top5_values1, top5_indices1 = paddle.topk(input, k=5)
            top5_values2, top5_indices2 = paddle.topk(
                input, k=base.dygraph.to_variable(np.array([5]))
            )
            np.testing.assert_array_equal(
                top5_values1.numpy(), top5_values2.numpy()
            )
            np.testing.assert_array_equal(
                top5_indices1.numpy(), top5_indices2.numpy()
            )

    def test_conv3d(self):
        with self.static_graph():
            images = paddle.static.data(
                name='pixel', shape=[-1, 3, 6, 6, 6], dtype='float32'
            )
            ret = paddle.static.nn.conv3d(
                input=images, num_filters=3, filter_size=2
            )
            static_ret = self.get_static_graph_result(
                feed={'pixel': np.ones([2, 3, 6, 6, 6], dtype='float32')},
                fetch_list=[ret],
            )[0]

        with self.static_graph():
            images = paddle.static.data(
                name='pixel', shape=[-1, 3, 6, 6, 6], dtype='float32'
            )
            conv3d = paddle.nn.Conv3D(
                in_channels=3, out_channels=3, kernel_size=2
            )
            ret = conv3d(images)
            static_ret2 = self.get_static_graph_result(
                feed={'pixel': np.ones([2, 3, 6, 6, 6], dtype='float32')},
                fetch_list=[ret],
            )[0]

        with self.dynamic_graph():
            images = np.ones([2, 3, 6, 6, 6], dtype='float32')
            conv3d = paddle.nn.Conv3D(
                in_channels=3, out_channels=3, kernel_size=2
            )
            dy_ret = conv3d(to_variable(images))
            dy_rlt_value = dy_ret.numpy()

        np.testing.assert_allclose(static_ret, dy_rlt_value, rtol=1e-05)
        np.testing.assert_allclose(static_ret, static_ret2, rtol=1e-05)

        with self.dynamic_graph():
            images = np.ones([2, 3, 6, 6, 6], dtype='float32')
            custom_weight = np.random.randn(3, 3, 2, 2, 2).astype("float32")
            weight_attr = base.ParamAttr(
                initializer=paddle.nn.initializer.Assign(custom_weight)
            )
            conv3d1 = paddle.nn.Conv3D(
                in_channels=3, out_channels=3, kernel_size=2
            )
            conv3d2 = paddle.nn.Conv3D(
                in_channels=3,
                out_channels=3,
                kernel_size=2,
                weight_attr=weight_attr,
            )
            dy_ret1 = conv3d1(to_variable(images))
            dy_ret2 = conv3d2(to_variable(images))
            self.assertFalse(np.array_equal(dy_ret1.numpy(), dy_ret2.numpy()))

            conv3d1_weight_np = conv3d1.weight.numpy()
            conv3d1_bias = conv3d1.bias
            self.assertFalse(
                np.array_equal(conv3d1_weight_np, conv3d2.weight.numpy())
            )
            conv3d2.weight.set_value(conv3d1_weight_np)
            np.testing.assert_array_equal(
                conv3d1_weight_np, conv3d2.weight.numpy()
            )
            conv3d1.bias.set_value(conv3d1_bias)
            dy_ret1 = conv3d1(to_variable(images))
            dy_ret2 = conv3d2(to_variable(images))
            np.testing.assert_array_equal(dy_ret1.numpy(), dy_ret2.numpy())

            conv3d2.weight = conv3d1.weight
            conv3d2.bias = conv3d1.bias
            np.testing.assert_array_equal(
                conv3d1.weight.numpy(), conv3d2.weight.numpy()
            )
            np.testing.assert_array_equal(
                conv3d1.bias.numpy(), conv3d2.bias.numpy()
            )

    def test_group_norm(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()

        shape = (2, 4, 3, 3)

        def _test_static_specific(input):
            with self.static_graph():
                X = paddle.static.data(
                    name='X', shape=shape, dtype='float32', lod_level=1
                )
                ret = paddle.static.nn.group_norm(
                    input=X,
                    groups=2,
                    param_attr=paddle.nn.initializer.Uniform(
                        low=-0.5, high=0.5
                    ),
                    bias_attr=paddle.nn.initializer.Constant(value=1),
                )
                static_ret = self.get_static_graph_result(
                    feed={
                        'X': base.create_lod_tensor(
                            data=input, recursive_seq_lens=[[1, 1]], place=place
                        )
                    },
                    fetch_list=[ret],
                    with_lod=True,
                )[0]

            return static_ret

        def _test_static(input):
            with self.static_graph():
                X = paddle.static.data(
                    name='X', shape=shape, dtype='float32', lod_level=1
                )
                groupNorm = paddle.nn.GroupNorm(
                    num_channels=shape[1],
                    num_groups=2,
                    weight_attr=paddle.nn.initializer.Uniform(
                        low=-0.5, high=0.5
                    ),
                    bias_attr=paddle.nn.initializer.Constant(value=1),
                )
                ret = groupNorm(X)
                static_ret2 = self.get_static_graph_result(
                    feed={
                        'X': base.create_lod_tensor(
                            data=input, recursive_seq_lens=[[1, 1]], place=place
                        )
                    },
                    fetch_list=[ret, groupNorm.weight],
                    with_lod=True,
                )[0]

            return static_ret2

        def _test_dygraph(input):
            with self.dynamic_graph():
                groupNorm = paddle.nn.GroupNorm(
                    num_channels=shape[1],
                    num_groups=2,
                    weight_attr=paddle.nn.initializer.Uniform(
                        low=-0.5, high=0.5
                    ),
                    bias_attr=paddle.nn.initializer.Constant(value=1),
                )
                dy_ret = groupNorm(to_variable(input))
                dy_rlt_value = dy_ret.numpy()
            return dy_rlt_value

        input = np.random.random(shape).astype('float32')
        static_ret = _test_static_specific(input)
        static_ret2 = _test_static(input)
        dy_rlt_value = _test_dygraph(input)
        np.testing.assert_allclose(static_ret, dy_rlt_value, rtol=1e-05)
        np.testing.assert_allclose(static_ret, static_ret2, rtol=1e-05)

        with paddle.pir_utils.IrGuard():
            static_ret_pir = _test_static(input)

        np.testing.assert_allclose(static_ret2, static_ret_pir, rtol=1e-05)

    def test_instance_norm(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()

        shape = (2, 4, 3, 3)

        def _test_static_specific(input):
            with self.static_graph():
                X = paddle.static.data(name='X', shape=shape, dtype='float32')
                ret = paddle.static.nn.instance_norm(input=X)
                static_ret = self.get_static_graph_result(
                    feed={'X': input}, fetch_list=[ret]
                )[0]
            return static_ret

        def _test_static(input):
            with self.static_graph():
                X = paddle.static.data(name='X', shape=shape, dtype='float32')
                instanceNorm = paddle.nn.InstanceNorm2D(num_features=shape[1])
                ret = instanceNorm(X)
                static_ret2 = self.get_static_graph_result(
                    feed={'X': input}, fetch_list=[ret]
                )[0]
            return static_ret2

        def _test_dygraph_1(input):
            with self.dynamic_graph():
                instanceNorm = paddle.nn.InstanceNorm2D(num_features=shape[1])
                dy_ret = instanceNorm(to_variable(input))
                dy_rlt_value = dy_ret.numpy()

            return dy_rlt_value

        def _test_dygraph_2(input):
            with self.dynamic_graph():
                instanceNorm = paddle.nn.InstanceNorm2D(num_features=shape[1])
                dy_ret = instanceNorm(to_variable(input))
                dy_rlt_value2 = dy_ret.numpy()
            return dy_rlt_value2

        input = np.random.random(shape).astype('float32')
        static_ret = _test_static_specific(input)
        static_ret2 = _test_static(input)
        dy_rlt_value = _test_dygraph_1(input)
        dy_rlt_value2 = _test_dygraph_2(input)

        np.testing.assert_allclose(static_ret, dy_rlt_value, rtol=1e-05)
        np.testing.assert_allclose(static_ret, dy_rlt_value2, rtol=1e-05)
        np.testing.assert_allclose(static_ret, static_ret2, rtol=1e-05)

        with paddle.pir_utils.IrGuard():
            static_ret_pir = _test_static(input)

        np.testing.assert_allclose(static_ret2, static_ret_pir, rtol=1e-05)

        def _test_errors():
            with self.static_graph():
                # the input of InstanceNorm must be Variable.
                def test_Variable():
                    instanceNorm = paddle.nn.InstanceNorm2D(
                        num_features=shape[1]
                    )
                    ret1 = instanceNorm(input)

                self.assertRaises(TypeError, test_Variable)

                # the input dtype of InstanceNorm must be float32 or float64
                def test_type():
                    input = np.random.random(shape).astype('int32')
                    instanceNorm = paddle.nn.InstanceNorm2D(
                        num_features=shape[1]
                    )
                    ret2 = instanceNorm(input)

                self.assertRaises(TypeError, test_type)

        _test_errors()

    def test_spectral_norm(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()

        shape = (2, 4, 3, 3)

        input = np.random.random(shape).astype('float32')

        with self.static_graph():
            Weight = paddle.static.data(
                name='Weight', shape=shape, dtype='float32', lod_level=1
            )
            ret = paddle.static.nn.spectral_norm(
                weight=Weight, dim=1, power_iters=2
            )
            static_ret = self.get_static_graph_result(
                feed={
                    'Weight': base.create_lod_tensor(
                        data=input, recursive_seq_lens=[[1, 1]], place=place
                    ),
                },
                fetch_list=[ret],
                with_lod=True,
            )[0]

        with self.static_graph():
            Weight = paddle.static.data(
                name='Weight', shape=shape, dtype='float32', lod_level=1
            )
            spectralNorm = paddle.nn.SpectralNorm(shape, dim=1, power_iters=2)
            ret = spectralNorm(Weight)
            static_ret2 = self.get_static_graph_result(
                feed={
                    'Weight': base.create_lod_tensor(
                        data=input, recursive_seq_lens=[[1, 1]], place=place
                    )
                },
                fetch_list=[ret],
                with_lod=True,
            )[0]

        with self.dynamic_graph():
            spectralNorm = paddle.nn.SpectralNorm(shape, dim=1, power_iters=2)
            dy_ret = spectralNorm(to_variable(input))
            dy_rlt_value = dy_ret.numpy()

        np.testing.assert_allclose(static_ret, dy_rlt_value, rtol=1e-05)
        np.testing.assert_allclose(static_ret, static_ret2, rtol=1e-05)

    def test_conv3d_transpose(self):
        input_array = (
            np.arange(0, 48).reshape([2, 3, 2, 2, 2]).astype('float32')
        )

        with self.static_graph():
            img = paddle.static.data(
                name='pixel', shape=[-1, 3, 2, 2, 2], dtype='float32'
            )
            out = paddle.static.nn.conv3d_transpose(
                input=img, num_filters=12, filter_size=12, use_cudnn=True
            )
            static_rlt = self.get_static_graph_result(
                feed={'pixel': input_array}, fetch_list=[out]
            )[0]
        with self.static_graph():
            img = paddle.static.data(
                name='pixel', shape=[-1, 3, 2, 2, 2], dtype='float32'
            )
            conv3d_transpose = paddle.nn.Conv3DTranspose(
                in_channels=3, out_channels=12, kernel_size=12
            )
            out = conv3d_transpose(img)
            static_rlt2 = self.get_static_graph_result(
                feed={'pixel': input_array}, fetch_list=[out]
            )[0]
        with self.dynamic_graph():
            conv3d_transpose = paddle.nn.Conv3DTranspose(
                in_channels=3, out_channels=12, kernel_size=12
            )
            dy_rlt = conv3d_transpose(to_variable(input_array))
            dy_rlt_value = dy_rlt.numpy()
        np.testing.assert_allclose(static_rlt2, static_rlt, rtol=1e-05)
        np.testing.assert_allclose(dy_rlt_value, static_rlt, rtol=1e-05)

        with self.dynamic_graph():
            images = np.ones([2, 3, 6, 6, 6], dtype='float32')
            custom_weight = np.random.randn(3, 3, 2, 2, 2).astype("float32")
            weight_attr = base.ParamAttr(
                initializer=paddle.nn.initializer.Assign(custom_weight)
            )
            conv3d1 = paddle.nn.Conv3DTranspose(
                in_channels=3,
                out_channels=3,
                kernel_size=2,
                bias_attr='conv3d1_b',
            )
            conv3d2 = paddle.nn.Conv3DTranspose(
                in_channels=3,
                out_channels=3,
                kernel_size=2,
                weight_attr=weight_attr,
                bias_attr='conv3d2_b',
            )
            dy_ret1 = conv3d1(to_variable(images))
            dy_ret2 = conv3d2(to_variable(images))
            self.assertFalse(np.array_equal(dy_ret1.numpy(), dy_ret2.numpy()))

            conv3d1_weight_np = conv3d1.weight.numpy()
            conv3d1_bias = conv3d1.bias
            self.assertFalse(
                np.array_equal(conv3d1_weight_np, conv3d2.weight.numpy())
            )
            conv3d2.weight.set_value(conv3d1_weight_np)
            np.testing.assert_array_equal(
                conv3d1_weight_np, conv3d2.weight.numpy()
            )
            conv3d1.bias.set_value(conv3d1_bias)
            dy_ret1 = conv3d1(to_variable(images))
            dy_ret2 = conv3d2(to_variable(images))
            np.testing.assert_array_equal(dy_ret1.numpy(), dy_ret2.numpy())

            conv3d2.weight = conv3d1.weight
            conv3d2.bias = conv3d1.bias
            np.testing.assert_array_equal(
                conv3d1.weight.numpy(), conv3d2.weight.numpy()
            )
            np.testing.assert_array_equal(
                conv3d1.bias.numpy(), conv3d2.bias.numpy()
            )

    def test_while_loop(self):
        with self.static_graph():
            i = paddle.tensor.fill_constant(shape=[1], dtype='int64', value=0)
            ten = paddle.tensor.fill_constant(
                shape=[1], dtype='int64', value=10
            )

            def cond(i):
                return paddle.less_than(i, ten)

            def body(i):
                return i + 1

            out = paddle.static.nn.while_loop(cond, body, [i])
            static_ret = self.get_static_graph_result(feed={}, fetch_list=out)

        with self.dynamic_graph():
            i = paddle.tensor.fill_constant(shape=[1], dtype='int64', value=0)
            ten = paddle.tensor.fill_constant(
                shape=[1], dtype='int64', value=10
            )

            def cond1(i):
                return paddle.less_than(i, ten)

            def body1(i):
                return i + 1

            dy_ret = paddle.static.nn.while_loop(cond1, body1, [i])
            with self.assertRaises(ValueError):
                j = paddle.tensor.fill_constant(
                    shape=[1], dtype='int64', value=0
                )

                def body2(i):
                    return i + 1, i + 2

                paddle.static.nn.while_loop(cond1, body2, [j])

        np.testing.assert_array_equal(static_ret[0], dy_ret[0].numpy())

    def test_compare(self):
        value_a = np.arange(3)
        value_b = np.arange(3)
        # less than
        with self.static_graph():
            a = paddle.static.data(name='a', shape=[-1, 1], dtype='int64')
            b = paddle.static.data(name='b', shape=[-1, 1], dtype='int64')
            cond = paddle.less_than(x=a, y=b)
            static_ret = self.get_static_graph_result(
                feed={"a": value_a, "b": value_b}, fetch_list=[cond]
            )[0]
        with self.dynamic_graph():
            da = to_variable(value_a)
            db = to_variable(value_b)
            dcond = paddle.less_than(x=da, y=db)

            for i in range(len(static_ret)):
                self.assertTrue(dcond.numpy()[i] == static_ret[i])

        # less equal
        with self.static_graph():
            a1 = paddle.static.data(name='a1', shape=[-1, 1], dtype='int64')
            b1 = paddle.static.data(name='b1', shape=[-1, 1], dtype='int64')
            cond1 = paddle.less_equal(x=a1, y=b1)
            static_ret1 = self.get_static_graph_result(
                feed={"a1": value_a, "b1": value_b}, fetch_list=[cond1]
            )[0]
        with self.dynamic_graph():
            da1 = to_variable(value_a)
            db1 = to_variable(value_b)
            dcond1 = paddle.less_equal(x=da1, y=db1)

            for i in range(len(static_ret1)):
                self.assertTrue(dcond1.numpy()[i] == static_ret1[i])

        # greater than
        with self.static_graph():
            a2 = paddle.static.data(name='a2', shape=[-1, 1], dtype='int64')
            b2 = paddle.static.data(name='b2', shape=[-1, 1], dtype='int64')
            cond2 = paddle.greater_than(x=a2, y=b2)
            static_ret2 = self.get_static_graph_result(
                feed={"a2": value_a, "b2": value_b}, fetch_list=[cond2]
            )[0]
        with self.dynamic_graph():
            da2 = to_variable(value_a)
            db2 = to_variable(value_b)
            dcond2 = paddle.greater_than(x=da2, y=db2)

            for i in range(len(static_ret2)):
                self.assertTrue(dcond2.numpy()[i] == static_ret2[i])

        # greater equal
        with self.static_graph():
            a3 = paddle.static.data(name='a3', shape=[-1, 1], dtype='int64')
            b3 = paddle.static.data(name='b3', shape=[-1, 1], dtype='int64')
            cond3 = paddle.greater_equal(x=a3, y=b3)
            static_ret3 = self.get_static_graph_result(
                feed={"a3": value_a, "b3": value_b}, fetch_list=[cond3]
            )[0]
        with self.dynamic_graph():
            da3 = to_variable(value_a)
            db3 = to_variable(value_b)
            dcond3 = paddle.greater_equal(x=da3, y=db3)

            for i in range(len(static_ret3)):
                self.assertTrue(dcond3.numpy()[i] == static_ret3[i])

        # equal
        with self.static_graph():
            a4 = paddle.static.data(name='a4', shape=[-1, 1], dtype='int64')
            b4 = paddle.static.data(name='b4', shape=[-1, 1], dtype='int64')
            cond4 = paddle.equal(x=a4, y=b4)
            static_ret4 = self.get_static_graph_result(
                feed={"a4": value_a, "b4": value_b}, fetch_list=[cond4]
            )[0]
        with self.dynamic_graph():
            da4 = to_variable(value_a)
            db4 = to_variable(value_b)
            dcond4 = paddle.equal(x=da4, y=db4)

            for i in range(len(static_ret4)):
                self.assertTrue(dcond4.numpy()[i] == static_ret4[i])

        # not equal
        with self.static_graph():
            a5 = paddle.static.data(name='a5', shape=[-1, 1], dtype='int64')
            b5 = paddle.static.data(name='b5', shape=[-1, 1], dtype='int64')
            cond5 = paddle.equal(x=a5, y=b5)
            static_ret5 = self.get_static_graph_result(
                feed={"a5": value_a, "b5": value_b}, fetch_list=[cond5]
            )[0]
        with self.dynamic_graph():
            da5 = to_variable(value_a)
            db5 = to_variable(value_b)
            dcond5 = paddle.equal(x=da5, y=db5)

            for i in range(len(static_ret5)):
                self.assertTrue(dcond5.numpy()[i] == static_ret5[i])

    def test_cond(self):
        def less_than_branch(a, b):
            return paddle.add(a, b)

        def greater_equal_branch(a, b):
            return paddle.subtract(a, b)

        with self.static_graph():
            a = paddle.tensor.fill_constant(
                shape=[1], dtype='float32', value=0.1
            )
            b = paddle.tensor.fill_constant(
                shape=[1], dtype='float32', value=0.23
            )
            out = paddle.static.nn.cond(
                a >= b,
                lambda: greater_equal_branch(a, b),
                lambda: less_than_branch(a, b),
            )
            place = (
                base.CUDAPlace(0)
                if core.is_compiled_with_cuda()
                else base.CPUPlace()
            )
            exe = base.Executor(place)
            ret = exe.run(fetch_list=[out])
            static_res = ret[0]

        with self.dynamic_graph():
            a = base.dygraph.to_variable(np.array([0.1]).astype('float32'))
            b = base.dygraph.to_variable(np.array([0.23]).astype('float32'))
            out = paddle.static.nn.cond(
                a < b,
                lambda: less_than_branch(a, b),
                lambda: greater_equal_branch(a, b),
            )
            out2 = paddle.static.nn.cond(
                a >= b,
                lambda: greater_equal_branch(a, b),
                lambda: less_than_branch(a, b),
            )
            dynamic_res = out.numpy()
            dynamic_res2 = out2.numpy()
            np.testing.assert_array_equal(dynamic_res, dynamic_res2)
            with self.assertRaises(TypeError):
                paddle.static.nn.cond(a < b, 'str', 'str')
            with self.assertRaises(TypeError):
                paddle.static.nn.cond(a >= b, 'str', 'str')

        np.testing.assert_array_equal(static_res, dynamic_res)

    def test_case(self):
        def fn_1():
            return paddle.tensor.fill_constant(
                shape=[1, 2], dtype='float32', value=1
            )

        def fn_2():
            return paddle.tensor.fill_constant(
                shape=[2, 2], dtype='int32', value=2
            )

        def fn_3():
            return paddle.tensor.fill_constant(
                shape=[3], dtype='int32', value=3
            )

        with self.static_graph():
            x = paddle.tensor.fill_constant(
                shape=[1], dtype='float32', value=0.3
            )
            y = paddle.tensor.fill_constant(
                shape=[1], dtype='float32', value=0.1
            )
            z = paddle.tensor.fill_constant(
                shape=[1], dtype='float32', value=0.2
            )

            pred_1 = paddle.less_than(z, x)  # true: 0.2 < 0.3
            pred_2 = paddle.less_than(x, y)  # false: 0.3 < 0.1
            pred_3 = paddle.equal(x, y)  # false: 0.3 == 0.1

            out_1 = paddle.static.nn.case(
                pred_fn_pairs=[(pred_1, fn_1), (pred_2, fn_2)], default=fn_3
            )
            out_2 = paddle.static.nn.case(
                pred_fn_pairs=[(pred_2, fn_2), (pred_3, fn_3)]
            )

            place = (
                base.CUDAPlace(0)
                if core.is_compiled_with_cuda()
                else base.CPUPlace()
            )
            exe = base.Executor(place)
            static_res1, static_res2 = exe.run(fetch_list=[out_1, out_2])

        with self.dynamic_graph():
            x = paddle.tensor.fill_constant(
                shape=[1], dtype='float32', value=0.3
            )
            y = paddle.tensor.fill_constant(
                shape=[1], dtype='float32', value=0.1
            )
            z = paddle.tensor.fill_constant(
                shape=[1], dtype='float32', value=0.2
            )

            pred_1 = paddle.less_than(z, x)  # true: 0.2 < 0.3
            pred_2 = paddle.less_than(x, y)  # false: 0.3 < 0.1
            pred_3 = paddle.equal(x, y)  # false: 0.3 == 0.1

            out_1 = paddle.static.nn.case(
                pred_fn_pairs=[(pred_1, fn_1), (pred_2, fn_2)], default=fn_3
            )
            out_2 = paddle.static.nn.case(
                pred_fn_pairs=[(pred_2, fn_2), (pred_3, fn_3)]
            )
            dynamic_res1 = out_1.numpy()
            dynamic_res2 = out_2.numpy()

        np.testing.assert_array_equal(static_res1, dynamic_res1)
        np.testing.assert_array_equal(static_res2, dynamic_res2)

    def test_switch_case(self):
        def fn_1():
            return paddle.tensor.fill_constant(
                shape=[1, 2], dtype='float32', value=1
            )

        def fn_2():
            return paddle.tensor.fill_constant(
                shape=[2, 2], dtype='int32', value=2
            )

        def fn_3():
            return paddle.tensor.fill_constant(
                shape=[3], dtype='int32', value=3
            )

        with self.static_graph():
            index_1 = paddle.tensor.fill_constant(
                shape=[1], dtype='int32', value=1
            )
            index_2 = paddle.tensor.fill_constant(
                shape=[1], dtype='int32', value=2
            )

            out_1 = paddle.static.nn.switch_case(
                branch_index=index_1,
                branch_fns={1: fn_1, 2: fn_2},
                default=fn_3,
            )
            out_2 = paddle.static.nn.switch_case(
                branch_index=index_2,
                branch_fns=[(1, fn_1), (2, fn_2)],
                default=fn_3,
            )
            out_3 = paddle.static.nn.switch_case(
                branch_index=index_2,
                branch_fns=[(0, fn_1), (4, fn_2), (7, fn_3)],
            )

            place = (
                base.CUDAPlace(0)
                if core.is_compiled_with_cuda()
                else base.CPUPlace()
            )
            exe = base.Executor(place)
            static_res1, static_res2, static_res3 = exe.run(
                fetch_list=[out_1, out_2, out_3]
            )

        with self.dynamic_graph():
            index_1 = paddle.tensor.fill_constant(
                shape=[1], dtype='int32', value=1
            )
            index_2 = paddle.tensor.fill_constant(
                shape=[1], dtype='int32', value=2
            )

            out_1 = paddle.static.nn.switch_case(
                branch_index=index_1,
                branch_fns={1: fn_1, 2: fn_2},
                default=fn_3,
            )
            out_2 = paddle.static.nn.switch_case(
                branch_index=index_2,
                branch_fns=[(1, fn_1), (2, fn_2)],
                default=fn_3,
            )
            out_3 = paddle.static.nn.switch_case(
                branch_index=index_2,
                branch_fns=[(0, fn_1), (4, fn_2), (7, fn_3)],
            )

            dynamic_res1 = out_1.numpy()
            dynamic_res2 = out_2.numpy()
            dynamic_res3 = out_3.numpy()

        np.testing.assert_array_equal(static_res1, dynamic_res1)
        np.testing.assert_array_equal(static_res2, dynamic_res2)
        np.testing.assert_array_equal(static_res3, dynamic_res3)

    def test_crop_tensor(self):
        with self.static_graph():
            x = paddle.static.data(
                name="x1", shape=[-1, 6, 5, 8], dtype="float32"
            )

            dim1 = paddle.static.data(name="dim1", shape=[1], dtype="float32")
            dim2 = paddle.static.data(name="dim2", shape=[1], dtype="float32")
            crop_shape1 = (1, 2, 4, 4)
            crop_shape2 = paddle.static.data(
                name="crop_shape", shape=[4], dtype="float32"
            )
            crop_shape3 = [-1, dim1, dim2, 4]
            crop_offsets1 = [0, 0, 1, 0]
            crop_offsets2 = paddle.static.data(
                name="crop_offset", shape=[4], dtype="float32"
            )
            crop_offsets3 = [0, dim1, dim2, 0]

            out1 = paddle.crop(x, shape=crop_shape1, offsets=crop_offsets1)
            out2 = paddle.crop(x, shape=crop_shape2, offsets=crop_offsets2)
            out3 = paddle.crop(x, shape=crop_shape3, offsets=crop_offsets3)

            self.assertIsNotNone(out1)
            self.assertIsNotNone(out2)
            self.assertIsNotNone(out3)

    def test_shard_index(self):
        with self.static_graph():
            x = paddle.static.data(
                name="label", shape=[-1, 4, 1], dtype='int64'
            )
            shard_label = paddle.shard_index(
                input=x, index_num=20, nshards=2, shard_id=0
            )

        self.assertIsNotNone(shard_label)

    def test_accuracy(self):
        x = np.random.rand(3, 32, 32).astype("float32")
        y = np.array([[1], [0], [1]])
        with self.static_graph():
            data = paddle.static.data(
                name="input", shape=[-1, 32, 32], dtype="float32"
            )
            label = paddle.static.data(name="label", shape=[-1, 1], dtype="int")
            data_new = paddle.reshape(data, [3, 32 * 32])
            fc_out = paddle.nn.Linear(32 * 32, 10)(data_new)
            predict = paddle.nn.functional.softmax(fc_out)
            result = paddle.static.accuracy(input=predict, label=label, k=5)
            place = base.CPUPlace()
            exe = base.Executor(place)

            exe.run(base.default_startup_program())
            # x = np.random.rand(3, 32, 32).astype("float32")
            # y = np.array([[1], [0], [1]])

            static_out = exe.run(
                feed={"input": x, "label": y}, fetch_list=result
            )

        with self.dynamic_graph(force_to_use_cpu=True):
            data = to_variable(x)
            label = to_variable(y)
            data_new = paddle.reshape(data, [3, 32 * 32])
            fc_out = paddle.nn.Linear(32 * 32, 10)(data_new)
            predict = paddle.nn.functional.softmax(fc_out)
            dynamic_out = paddle.static.accuracy(
                input=predict, label=label, k=5
            )

        np.testing.assert_array_equal(static_out[0], dynamic_out.numpy())


class TestBook(LayerTest):
    def setUp(self):
        self.only_static_set = set({"make_word_embedding"})
        self.not_compare_static_dygraph_set = set(
            {
                "make_gaussian_random",
                "make_kldiv_loss",
                "make_uniform_random_batch_size_like",
            }
        )
        self.all_close_compare = set({"make_spectral_norm"})

    def test_all_layers(self):
        attrs = (getattr(self, name) for name in dir(self))
        methods = filter(inspect.ismethod, attrs)
        for method in methods:
            if not method.__name__.startswith('make_'):
                continue
            self._low_data_bound = 0
            self._high_data_bound = 2
            self._batch_size = 2
            self._feed_dict = {}
            self._force_to_use_cpu = False
            with self.static_graph():
                static_var = method()
                if isinstance(static_var, tuple):
                    static_var = static_var[0]

                if static_var is not None:
                    fetch_list = [static_var.name]
                    static_result = self.get_static_graph_result(
                        feed=self._feed_dict,
                        fetch_list=fetch_list,
                        force_to_use_cpu=self._force_to_use_cpu,
                    )

                else:
                    continue
            if method.__name__ in self.only_static_set:
                continue

            with self.dynamic_graph(self._force_to_use_cpu):
                dy_result = method()
                if isinstance(dy_result, tuple):
                    dy_result = dy_result[0]
                dy_result_value = dy_result.numpy()

            if method.__name__ in self.all_close_compare:
                np.testing.assert_allclose(
                    static_result[0],
                    dy_result_value,
                    rtol=1e-05,
                    atol=0,
                    err_msg=f'Result of function [{method.__name__}] compare failed',
                )
                continue

            if method.__name__ not in self.not_compare_static_dygraph_set:
                np.testing.assert_array_equal(
                    static_result[0],
                    dy_result_value,
                    err_msg=f'Result of function [{method.__name__}] not equal',
                )

    def _get_np_data(self, shape, dtype, append_batch_size=True):
        np.random.seed(self.seed)
        if append_batch_size:
            shape = [self._batch_size] + shape
        if dtype == 'float32':
            return np.random.random(shape).astype(dtype)
        elif dtype == 'float64':
            return np.random.random(shape).astype(dtype)
        elif dtype == 'int32':
            return np.random.randint(
                self._low_data_bound, self._high_data_bound, shape
            ).astype(dtype)
        elif dtype == 'int64':
            return np.random.randint(
                self._low_data_bound, self._high_data_bound, shape
            ).astype(dtype)

    def _get_data(
        self, name, shape, dtype, set_feed_dict=True, append_batch_size=True
    ):
        if dygraph.base.enabled():
            return to_variable(
                value=self._get_np_data(shape, dtype, append_batch_size),
                name=name,
                zero_copy=False,
            )
        else:
            if set_feed_dict:
                self._feed_dict[name] = self._get_np_data(
                    shape, dtype, append_batch_size
                )
            if append_batch_size:
                shape = [-1] + shape
            data = paddle.static.data(
                name=name,
                shape=shape,
                dtype=dtype,
            )
            data.desc.set_need_check_feed(False)
            return data

    def make_fit_a_line(self):
        with program_guard(
            base.default_main_program(),
            startup_program=base.default_startup_program(),
        ):
            x = self._get_data(name='x', shape=[13], dtype='float32')
            y_predict = paddle.nn.Linear(13, 1)(x)
            y = self._get_data(name='y', shape=[1], dtype='float32')
            cost = paddle.nn.functional.square_error_cost(
                input=y_predict, label=y
            )
            avg_cost = paddle.mean(cost)
            return avg_cost

    def make_recognize_digits_mlp(self):
        with program_guard(
            base.default_main_program(), base.default_startup_program()
        ):
            # Change g_program, so the rest layers use `g_program`
            images = self._get_data(name='pixel', shape=[784], dtype='float32')
            label = self._get_data(name='label', shape=[1], dtype='int64')
            hidden1 = paddle.nn.Linear(784, 128)(images)
            hidden1 = paddle.nn.functional.relu(hidden1)
            hidden2 = paddle.nn.Linear(128, 64)(hidden1)
            hidden2 = paddle.nn.functional.relu(hidden2)
            hidden1 = paddle.nn.Linear(128, 10, "sftmax.w1")(hidden1)
            hidden2 = paddle.nn.Linear(64, 10, "sftmax.w2")(hidden2)
            hidden = hidden1 + hidden2
            predict = paddle.nn.functional.softmax(hidden)
            cost = paddle.nn.functional.cross_entropy(
                input=predict, label=label, reduction='none', use_softmax=False
            )
            avg_cost = paddle.mean(cost)
            return avg_cost

    def make_conv2d_transpose(self):
        with program_guard(
            base.default_main_program(), base.default_startup_program()
        ):
            img = self._get_data(name='pixel', shape=[3, 2, 2], dtype='float32')
            return paddle.static.nn.conv2d_transpose(
                input=img, num_filters=10, output_size=28
            )

    def make_recognize_digits_conv(self):
        with program_guard(
            base.default_main_program(), base.default_startup_program()
        ):
            images = self._get_data(
                name='pixel', shape=[1, 28, 28], dtype='float32'
            )
            label = self._get_data(name='label', shape=[1], dtype='int64')
            conv_pool_1 = nets.simple_img_conv_pool(
                input=images,
                filter_size=5,
                num_filters=2,
                pool_size=2,
                pool_stride=2,
                act="relu",
            )
            conv_pool_2 = nets.simple_img_conv_pool(
                input=conv_pool_1,
                filter_size=5,
                num_filters=4,
                pool_size=2,
                pool_stride=2,
                act="relu",
            )

            conv_pool_2_new = paddle.reshape(
                conv_pool_2,
                [
                    conv_pool_2.shape[0],
                    conv_pool_2.shape[1]
                    * conv_pool_2.shape[2]
                    * conv_pool_2.shape[3],
                ],
            )
            predict = paddle.nn.Linear(
                conv_pool_2.shape[1]
                * conv_pool_2.shape[2]
                * conv_pool_2.shape[3],
                10,
            )(conv_pool_2_new)
            predict = paddle.nn.functional.softmax(predict)
            cost = paddle.nn.functional.cross_entropy(
                input=predict, label=label, reduction='none', use_softmax=False
            )
            avg_cost = paddle.mean(cost)
            return avg_cost

    def make_word_embedding(self):
        with program_guard(
            base.default_main_program(), base.default_startup_program()
        ):
            dict_size = 10000
            embed_size = 32
            first_word = self._get_data(name='firstw', shape=[1], dtype='int64')
            second_word = self._get_data(
                name='secondw', shape=[1], dtype='int64'
            )
            third_word = self._get_data(name='thirdw', shape=[1], dtype='int64')
            forth_word = self._get_data(name='forthw', shape=[1], dtype='int64')
            next_word = self._get_data(name='nextw', shape=[1], dtype='int64')

            embed_first = paddle.static.nn.embedding(
                input=first_word,
                size=[dict_size, embed_size],
                dtype='float32',
                param_attr='shared_w',
            )
            embed_second = paddle.static.nn.embedding(
                input=second_word,
                size=[dict_size, embed_size],
                dtype='float32',
                param_attr='shared_w',
            )

            embed_third = paddle.static.nn.embedding(
                input=third_word,
                size=[dict_size, embed_size],
                dtype='float32',
                param_attr='shared_w',
            )
            embed_forth = paddle.static.nn.embedding(
                input=forth_word,
                size=[dict_size, embed_size],
                dtype='float32',
                param_attr='shared_w',
            )

            concat_embed = paddle.concat(
                [embed_first, embed_second, embed_third, embed_forth],
                axis=1,
            )

            hidden1 = paddle.static.nn.fc(
                x=concat_embed, size=256, activation='sigmoid'
            )
            predict_word = paddle.static.nn.fc(
                x=hidden1, size=dict_size, activation='softmax'
            )
            cost = paddle.nn.functional.cross_entropy(
                input=predict_word,
                label=next_word,
                reduction='none',
                use_softmax=False,
            )
            avg_cost = paddle.mean(cost)
            return avg_cost

    def make_pool2d(self):
        with program_guard(
            base.default_main_program(), base.default_startup_program()
        ):
            x = self._get_data(name='x', shape=[3, 224, 224], dtype='float32')
            return paddle.nn.functional.max_pool2d(
                x, kernel_size=[5, 3], stride=[1, 2], padding=(2, 1)
            )

    def make_pool2d_infershape(self):
        with program_guard(
            base.default_main_program(), base.default_startup_program()
        ):
            theta = self._get_data("theta", shape=[2, 3], dtype='float32')
            x = paddle.nn.functional.affine_grid(
                theta, out_shape=[2, 3, 244, 244]
            )
            return paddle.nn.functional.max_pool2d(
                x, kernel_size=[5, 3], stride=[1, 2], padding=(2, 1)
            )

    def make_softmax(self):
        with program_guard(
            base.default_main_program(), base.default_startup_program()
        ):
            data = self._get_data(name='data', shape=[10], dtype='float32')
            hid = paddle.nn.Linear(10, 20)(data)
            return paddle.nn.functional.softmax(hid, axis=1)

    @prog_scope()
    def make_nce(self):
        window_size = 5
        words = []
        for i in range(window_size):
            words.append(
                self._get_data(name=f'word_{i}', shape=[1], dtype='int64')
            )

        dict_size = 10000
        label_word = int(window_size // 2) + 1

        embs = []
        for i in range(window_size):
            if i == label_word:
                continue

            emb = paddle.static.nn.embedding(
                input=words[i],
                size=[dict_size, 32],
                param_attr='emb.w',
                is_sparse=True,
            )

            embs.append(emb)

        embs = paddle.concat(embs, axis=1)
        loss = paddle.static.nn.nce(
            input=embs,
            label=words[label_word],
            num_total_classes=dict_size,
            param_attr='nce.w',
            bias_attr='nce.b',
        )
        avg_loss = paddle.mean(loss)
        return avg_loss

    def make_multiplex(self):
        with program_guard(
            base.default_main_program(), base.default_startup_program()
        ):
            x1 = self._get_data(name='x1', shape=[4], dtype='float32')
            x2 = self._get_data(name='x2', shape=[4], dtype='float32')
            index = self._get_data(name='index', shape=[1], dtype='int32')
            out = paddle.multiplex(inputs=[x1, x2], index=index)
            return out

    def make_softmax_with_cross_entropy(self):
        with program_guard(
            base.default_main_program(), base.default_startup_program()
        ):
            x = self._get_data(name='x', shape=[16], dtype='float32')
            y = self._get_data(name='label', shape=[1], dtype='int64')
            loss, softmax = paddle.nn.functional.softmax_with_cross_entropy(
                x, y, return_softmax=True
            )
            self.assertIsNotNone(loss)
            self.assertIsNotNone(softmax)

            loss = paddle.nn.functional.softmax_with_cross_entropy(x, y)
            self.assertIsNotNone(loss)

            x1 = self._get_data(name='x1', shape=[16, 32, 64], dtype='float32')
            y1 = self._get_data(name='label1', shape=[1, 32, 64], dtype='int64')
            y2 = self._get_data(name='label2', shape=[16, 1, 64], dtype='int64')
            y3 = self._get_data(name='label3', shape=[16, 32, 1], dtype='int64')
            loss1 = paddle.nn.functional.softmax_with_cross_entropy(
                x1, y1, axis=1
            )
            loss2 = paddle.nn.functional.softmax_with_cross_entropy(
                x1, y2, axis=2
            )
            loss3 = paddle.nn.functional.softmax_with_cross_entropy(
                x1, y3, axis=3
            )
            loss4 = paddle.nn.functional.softmax_with_cross_entropy(
                x1, y3, axis=-1
            )
            self.assertIsNotNone(loss1)
            self.assertIsNotNone(loss2)
            self.assertIsNotNone(loss3)
            self.assertIsNotNone(loss4)
            return loss4

    def make_scatter(self):
        with program_guard(
            base.default_main_program(), base.default_startup_program()
        ):
            x = self._get_data(
                name='x', shape=[3, 3], append_batch_size=False, dtype='float32'
            )
            idx = self._get_data(
                name='idx', shape=[2], append_batch_size=False, dtype='int32'
            )
            updates = self._get_data(
                name='updates',
                shape=[2, 3],
                dtype='float32',
                append_batch_size=False,
            )
            out = paddle.scatter(x, index=idx, updates=updates)
            return out

    def make_one_hot(self):
        with base.framework._dygraph_place_guard(place=base.CPUPlace()):
            label = self._get_data(name="label", shape=[1], dtype="int32")
            one_hot_label = paddle.nn.functional.one_hot(label, 10)
            return one_hot_label

    def make_label_smooth(self):
        # TODO(minqiyang): support gpu ut
        self._force_to_use_cpu = True
        with base.framework._dygraph_place_guard(place=base.CPUPlace()):
            label = self._get_data(name="label", shape=[1], dtype="int32")
            one_hot_label = paddle.nn.functional.one_hot(label, 10)
            smooth_label = F.label_smooth(label=one_hot_label, epsilon=0.1)
            return smooth_label

    def make_topk(self):
        with program_guard(
            base.default_main_program(), base.default_startup_program()
        ):
            data = self._get_data(name="label", shape=[200], dtype="float32")
            values, indices = paddle.topk(data, k=5)
            return values
            return indices

    def make_l2_normalize(self):
        with program_guard(
            base.default_main_program(), base.default_startup_program()
        ):
            x = self._get_data(name='x', shape=[8, 7, 10], dtype="float32")
            output = paddle.nn.functional.normalize(x, axis=1)
            return output

    def make_shape(self):
        with program_guard(
            base.default_main_program(), base.default_startup_program()
        ):
            input = self._get_data(
                name="input", shape=[3, 100, 100], dtype="float32"
            )
            out = paddle.shape(input)
            return out

    def make_pad2d(self):
        with program_guard(
            base.default_main_program(), base.default_startup_program()
        ):
            input = self._get_data(
                name="input", shape=[3, 100, 100], dtype="float32"
            )

            tmp_pad = paddle.nn.Pad2D(
                padding=[1, 2, 3, 4],
                mode='reflect',
                data_format='NCHW',
                name="shape",
            )
            out = tmp_pad(input)
            return out

    def make_mish(self):
        with program_guard(
            base.default_main_program(), base.default_startup_program()
        ):
            input = self._get_data(name="input", shape=[16], dtype="float32")
            out = paddle.nn.functional.mish(input, name='mish')
            return out

    def make_cross_entropy(self):
        with program_guard(
            base.default_main_program(), base.default_startup_program()
        ):
            x = self._get_data(name="x", shape=[30, 10], dtype="float32")
            label = self._get_data(name="label", shape=[30, 1], dtype="int64")
            mode = 'channel'
            out = paddle.nn.functional.cross_entropy(
                x,
                label,
                soft_label=False,
                ignore_index=4,
                reduction='none',
                use_softmax=False,
            )
            return out

    def make_uniform_random_batch_size_like(self):
        with program_guard(
            base.default_main_program(), base.default_startup_program()
        ):
            input = self._get_data(
                name="input", shape=[13, 11], dtype='float32'
            )
            out = random.uniform_random_batch_size_like(input, [-1, 11])
            return out

    def make_gaussian_random(self):
        with program_guard(
            base.default_main_program(), base.default_startup_program()
        ):
            out = random.gaussian(shape=[20, 30])
            return out

    def make_sum(self):
        with program_guard(
            base.default_main_program(), base.default_startup_program()
        ):
            input = self._get_data(
                name="input", shape=[13, 11], dtype='float32'
            )

            out = paddle.add_n(input)
            return out

    def make_slice(self):
        starts = [1, 0, 2]
        ends = [3, 3, 4]
        axes = [0, 1, 2]

        with program_guard(
            base.default_main_program(), base.default_startup_program()
        ):
            input = self._get_data(
                name="input", shape=[3, 4, 5, 6], dtype='float32'
            )

            out = paddle.slice(input, axes=axes, starts=starts, ends=ends)
            return out

    def make_scale_variable(self):
        with program_guard(
            base.default_main_program(), base.default_startup_program()
        ):
            input = self._get_data(
                name="input", shape=[3, 4, 5, 6], dtype='float32'
            )
            scale_var = self._get_data(
                name="scale",
                shape=[1],
                dtype='float32',
                append_batch_size=False,
            )
            out = paddle.scale(input, scale=scale_var)
            return out

    def make_bilinear_tensor_product_layer(self):
        with program_guard(
            base.default_main_program(), base.default_startup_program()
        ):
            data = self._get_data(name='data', shape=[4], dtype="float32")

            theta = self._get_data(name="theta", shape=[5], dtype="float32")
            out = paddle.static.nn.common.bilinear_tensor_product(
                data, theta, 6
            )
            return out

    def make_batch_norm(self):
        with program_guard(
            base.default_main_program(), base.default_startup_program()
        ):
            data = self._get_data(
                name='data', shape=[32, 128, 128], dtype="float32"
            )
            out = paddle.static.nn.batch_norm(data)
            return out

    def make_batch_norm_momentum_variable(self):
        with program_guard(
            base.default_main_program(), base.default_startup_program()
        ):
            data = self._get_data(
                name='data', shape=[32, 128, 128], dtype="float32"
            )
            momentum = self._get_data(
                name='momentum',
                shape=[1],
                dtype='float32',
                append_batch_size=False,
            )
            out = paddle.static.nn.batch_norm(data, momentum=momentum)
            return out

    def make_range(self):
        with program_guard(
            base.default_main_program(), base.default_startup_program()
        ):
            paddle.arange(0, 10, 2, 'int32')
            paddle.arange(0.1, 10.0, 0.2, 'float32')
            paddle.arange(0.1, 10.0, 0.2, 'float64')
            start = paddle.tensor.fill_constant(
                shape=[1], value=0.1, dtype="float32"
            )
            end = paddle.tensor.fill_constant(
                shape=[1], value=10.0, dtype="float32"
            )
            step = paddle.tensor.fill_constant(
                shape=[1], value=0.2, dtype="float32"
            )
            y = paddle.arange(start, end, step, 'float64')
            return y

    def make_spectral_norm(self):
        with program_guard(
            base.default_main_program(), base.default_startup_program()
        ):
            weight = self._get_data(
                name='weight',
                shape=[2, 3, 32, 32],
                dtype="float32",
                append_batch_size=False,
            )
            out = paddle.static.nn.spectral_norm(weight, dim=1, power_iters=1)
            return out

    def make_kldiv_loss(self):
        with program_guard(
            base.default_main_program(), base.default_startup_program()
        ):
            x = self._get_data(
                name='x',
                shape=[32, 128, 128],
                dtype="float32",
                append_batch_size=False,
            )
            target = self._get_data(
                name='target',
                shape=[32, 128, 128],
                dtype="float32",
                append_batch_size=False,
            )
            loss = paddle.nn.functional.kl_div(
                input=x, label=target, reduction='batchmean'
            )
            return loss

    def make_pixel_shuffle(self):
        with program_guard(
            base.default_main_program(), base.default_startup_program()
        ):
            x = self._get_data(name="X", shape=[9, 4, 4], dtype="float32")
            out = paddle.nn.functional.pixel_shuffle(x, upscale_factor=3)
            return out

    def make_mse_loss(self):
        with program_guard(
            base.default_main_program(), base.default_startup_program()
        ):
            x = self._get_data(name="X", shape=[1], dtype="float32")
            y = self._get_data(name="Y", shape=[1], dtype="float32")
            out = paddle.nn.functional.mse_loss(input=x, label=y)
            return out

    def make_square_error_cost(self):
        with program_guard(
            base.default_main_program(), base.default_startup_program()
        ):
            x = self._get_data(name="X", shape=[1], dtype="float32")
            y = self._get_data(name="Y", shape=[1], dtype="float32")
            out = paddle.nn.functional.square_error_cost(input=x, label=y)
            return out

    def test_affine_grid(self):
        with self.static_graph():
            data = paddle.static.data(
                name='data', shape=[-1, 2, 3, 3], dtype="float32"
            )
            out = paddle.argsort(x=data, axis=1)

            theta = paddle.static.data(
                name="theta", shape=[-1, 2, 3], dtype="float32"
            )
            out_shape = paddle.static.data(
                name="out_shape", shape=[-1], dtype="int32"
            )
            data_0 = paddle.nn.functional.affine_grid(theta, out_shape)
            data_1 = paddle.nn.functional.affine_grid(theta, [5, 3, 28, 28])

            self.assertIsNotNone(data_0)
            self.assertIsNotNone(data_1)

    def test_stridedslice(self):
        axes = [0, 1, 2]
        starts = [1, 0, 2]
        ends = [3, 3, 4]
        strides = [1, 1, 1]
        with self.static_graph():
            x = paddle.static.data(
                name="x", shape=[-1, 245, 30, 30], dtype="float32"
            )
            out = paddle.strided_slice(
                x, axes=axes, starts=starts, ends=ends, strides=strides
            )
            return out

    def test_shuffle_batch(self):
        # TODO(minqiyang): dygraph do not support lod now
        with self.static_graph():
            x = paddle.static.data(
                name='X', shape=[-1, 4, 50], dtype='float32', lod_level=0
            )
            out1 = shuffle_batch(x)
            default_main_program().random_seed = 1000
            out2 = shuffle_batch(x)
            self.assertIsNotNone(out1)
            self.assertIsNotNone(out2)
            return out1

    def test_partial_sum(self):
        with self.static_graph():
            x = paddle.static.data(name="x", shape=[None, 3], dtype="float32")
            y = paddle.static.data(name="y", shape=[None, 3], dtype="float32")
            sum = partial_sum([x, y], start_index=0, length=2)
            return sum

    def test_batch_fc(self):
        with self.static_graph():
            input = paddle.static.data(
                name="input", shape=[16, 2, 3], dtype="float32"
            )
            out = batch_fc(
                input=input,
                param_size=[16, 3, 10],
                param_attr=base.ParamAttr(
                    learning_rate=1.0,
                    name="w_0",
                    initializer=paddle.nn.initializer.XavierNormal(),
                ),
                bias_size=[16, 10],
                bias_attr=base.ParamAttr(
                    learning_rate=1.0,
                    name="b_0",
                    initializer=paddle.nn.initializer.XavierNormal(),
                ),
                act="relu",
            )
        return out

    def test_rank_attention(self):
        with self.static_graph():
            input = paddle.static.data(
                name="input", shape=[None, 2], dtype="float32"
            )
            rank_offset = paddle.static.data(
                name="rank_offset", shape=[None, 7], dtype="int32"
            )
            out = rank_attention(
                input=input,
                rank_offset=rank_offset,
                rank_param_shape=[18, 3],
                rank_param_attr=base.ParamAttr(
                    learning_rate=1.0,
                    name="ubm_rank_param.w_0",
                    initializer=paddle.nn.initializer.XavierNormal(),
                ),
                max_rank=3,
            )
            return out

    def test_row_conv(self):
        # TODO(minqiyang): dygraph do not support lod now
        with self.static_graph():
            x = paddle.static.data(
                name='x', shape=[-1, 16], dtype='float32', lod_level=1
            )
            out = paddle.static.nn.row_conv(input=x, future_context_size=2)
            return out

    def test_simple_conv2d(self):
        # TODO(minqiyang): dygraph do not support layers with param now
        with self.static_graph():
            images = paddle.static.data(
                name='pixel', shape=[-1, 3, 48, 48], dtype='float32'
            )
            return paddle.static.nn.conv2d(
                input=images, num_filters=3, filter_size=[4, 4]
            )

    def test_squeeze(self):
        # TODO(minqiyang): dygraph do not support layers with param now
        with self.static_graph():
            x = paddle.static.data(
                name='x', shape=[-1, 1, 1, 4], dtype='float32'
            )
            out = paddle.squeeze(x, axis=[2])
            return out

    def test_flatten(self):
        # TODO(minqiyang): dygraph do not support op without kernel now
        with self.static_graph():
            x = paddle.static.data(
                name='x',
                shape=[4, 4, 3],
                dtype="float32",
            )
            out = paddle.flatten(x, 1, -1, name="flatten")
            return out

    def test_linspace(self):
        program = Program()
        with program_guard(program):
            out = paddle.linspace(20, 10, 5, 'float64')
            self.assertIsNotNone(out)
        print(str(program))

    def test_unfold(self):
        with self.static_graph():
            x = paddle.static.data(
                name='x', shape=[-1, 3, 20, 20], dtype='float32'
            )
            out = paddle.nn.functional.unfold(x, [3, 3], 1, 1, 1)
            return out

    def test_partial_concat(self):
        with self.static_graph():
            x = paddle.static.data(name="x", shape=[None, 3], dtype="float32")
            y = paddle.static.data(name="y", shape=[None, 3], dtype="float32")
            concat1 = partial_concat([x, y], start_index=0, length=2)
            concat2 = partial_concat(x, start_index=0, length=-1)
            return concat1, concat2

    def test_addmm(self):
        with program_guard(
            base.default_main_program(), base.default_startup_program()
        ):
            input = paddle.static.data(
                name='input_data',
                shape=[3, 3],
                dtype='float32',
            )
            x = paddle.static.data(name='x', shape=[3, 2], dtype='float32')
            y = paddle.static.data(name='y', shape=[2, 3], dtype='float32')

            out = paddle.addmm(input=input, x=x, y=y)
            return out

    def test_warpctc_with_padding(self):
        # TODO(minqiyang): dygraph do not support lod now
        with self.static_graph():
            input_length = paddle.static.data(
                name='logits_length', shape=[11], dtype='int64'
            )
            label_length = paddle.static.data(
                name='labels_length', shape=[12], dtype='int64'
            )
            label = paddle.static.data(
                name='label', shape=[12, 1], dtype='int32'
            )
            predict = paddle.static.data(
                name='predict', shape=[4, 4, 8], dtype='float32'
            )
            output = paddle.nn.functional.ctc_loss(
                log_probs=predict,
                labels=label,
                input_lengths=input_length,
                label_lengths=label_length,
                reduction='none',
            )
            return output


class ExampleNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.weight = self.create_parameter(
            shape=[1, 1], attr=paddle.ParamAttr(trainable=False)
        )

    def forward(self):
        # only for test parameter trainable attr
        pass


class TestLayerParameterTrainableSet(unittest.TestCase):
    def test_layer_parameter_set(self):
        with base.dygraph.guard():
            net = ExampleNet()
            self.assertFalse(net.weight.trainable)


class TestLayerTrainingAttribute(unittest.TestCase):
    def test_set_train_eval_in_dynamic_mode(self):
        with base.dygraph.guard():
            net = paddle.nn.Dropout()
            net.train()
            self.assertTrue(net.training)
            net.eval()
            self.assertFalse(net.training)

    def test_set_train_eval_in_static_mode(self):
        net = paddle.nn.Dropout()
        net.train()
        self.assertTrue(net.training)
        net.eval()
        self.assertFalse(net.training)


class MyLayer(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self._linear = paddle.nn.Linear(1, 1)
        self._dropout = paddle.nn.Dropout(p=0.5)

    def forward(self, input):
        temp = self._linear(input)
        temp = self._dropout(temp)
        return temp


class MySuperLayer(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self._mylayer = MyLayer()

    def forward(self, input):
        temp = self._mylayer(input)
        return temp


class TestSubLayerCount(unittest.TestCase):
    def test_sublayer(self):
        with base.dygraph.guard():
            mySuperlayer = MySuperLayer()
            self.assertTrue(len(mySuperlayer.sublayers()) == 3)
            self.assertTrue(len(mySuperlayer.sublayers(include_self=True)) == 4)


class TestExcludedLayersSupportBool(unittest.TestCase):
    def test_support_tuple(self):
        with base.dygraph.guard():
            model = MyLayer()
            model.float16(excluded_layers=[paddle.nn.Linear])
            self.assertTrue(model._linear.weight.dtype == paddle.float32)
            model.bfloat16(excluded_layers=(paddle.nn.Linear))
            self.assertTrue(model._linear.weight.dtype == paddle.float32)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
