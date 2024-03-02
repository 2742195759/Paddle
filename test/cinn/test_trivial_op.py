#!/usr/bin/env python3

# Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

import os
import unittest

os.environ['FLAGS_cinn_new_group_scheduler'] = '1'
os.environ['FLAGS_group_schedule_tiling_first'] = '1'
os.environ['FLAGS_prim_all'] = 'true'
os.environ['FLAGS_print_ir'] = '1'
os.environ['FLAGS_enable_pir_api'] = '1'
os.environ['FLAGS_use_cinn'] = '1'
os.environ['GLOG_vmodule'] = 'op_lowering_impl=4'
import paddle

build_strategy = paddle.static.BuildStrategy()
build_strategy.build_cinn_pass = True


class TestTrivalFusion(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_trival_fusion_elementwise(self):
        @paddle.jit.to_static(full_graph=True, build_strategy=build_strategy)
        def func(x):
            x = x * 2
            # x = x.reshape((-1, 128))
            x = x + 1
            # x = x.reshape((128, -1))
            # x = x.transpose((0, 1, 2)
            x = x * 2
            x = paddle.nn.functional.relu(x)
            x = x * 2
            x = x * 2
            x = x * 2
            x = paddle.nn.functional.relu(x)
            x = paddle.nn.functional.relu(x)
            x = paddle.nn.functional.relu(x)
            x = paddle.nn.functional.relu(x)
            return x

        x = paddle.rand((32, 32, 128))
        out = func(x)
        print(out)
        print(out.shape)

    def test_trival_fusion_tranpose(self):
        @paddle.jit.to_static(full_graph=True, build_strategy=build_strategy)
        def func(x):
            x = x * 2
            x = paddle.transpose(x, perm=[0, 2, 1])
            return x

        x = paddle.rand((32, 32, 128))
        out = func(x)
        print(out)
        print(out.shape)

    def test_trival_fusion_reshape(self):
        @paddle.jit.to_static(full_graph=True, build_strategy=build_strategy)
        def func(x):
            x = x * 2
            x = x.reshape((-1, 128))
            x = paddle.transpose(x, perm=[1, 0])
            return x

        x = paddle.rand((32, 32, 128))
        out = func(x)
        print(out)
        print(out.shape)

    # Error: build_cinn_pass don't group the concat op.
    def test_trival_fusion_concat(self):
        @paddle.jit.to_static(full_graph=True, build_strategy=build_strategy)
        def func(x, y):
            x = x * 2
            y = y * 2
            z = paddle.concat([x, y], axis=-1)
            return z

        x = paddle.rand((32, 32, 128))
        y = paddle.rand((32, 32, 128))
        out = func(x, y)
        print(out)
        print(out.shape)

    def test_trival_fusion_slice(self):
        @paddle.jit.to_static(full_graph=True, build_strategy=build_strategy)
        def func(x, y):
            x = x * 2
            y = y * 2
            x = x[:, :, 0]
            y = y[:, :, 0]
            return x + y

        x = paddle.rand((32, 32, 128))
        y = paddle.rand((32, 32, 128))
        out = func(x, y)
        print(out)
        print(out.shape)

    def test_trival_fusion_gather_nd(self):
        @paddle.jit.to_static(full_graph=True, build_strategy=build_strategy)
        def func(x, y):
            x = x * 2
            output = paddle.gather_nd(x, index)
            return output

        x = paddle.to_tensor(
            [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]
        )
        index = paddle.to_tensor([[0, 1]])
        out = func(x, index)
        print(out)
        print(out.shape)

    def test_compose_of_broadcast(self):
        @paddle.jit.to_static(full_graph=True, build_strategy=build_strategy)
        def func(x, y, z):
            output = x + y + z
            return output

        x = paddle.rand((32, 1, 128))
        y = paddle.rand((1, 32, 128))
        z = paddle.rand((1, 32, 1))
        out = func(x, y, z)
        print(out)
        print(out.shape)


class RotaryPosEmb(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, cos, sin, position_ids):
        cos = cos.squeeze(axis=[0, 2])  # [seq_len, dim]
        sin = sin.squeeze(axis=[0, 2])  # [seq_len, dim]

        cos = cos[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
        sin = sin[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed

    def rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return paddle.concat([-x2, x1], axis=-1)  # shape is the same as x


class TestRotaryPosEmb(unittest.TestCase):
    def prepare_data(self):
        self.q = paddle.randn([1, 2048, 8, 96], dtype="float32")
        self.q.stop_gradient = False

        self.k = paddle.randn([1, 2048, 8, 96], dtype="float32")
        self.k.stop_gradient = False

        self.cos = paddle.randn([1, 2048, 1, 96], dtype="float32")
        self.cos.stop_gradient = False

        self.sin = paddle.randn([1, 2048, 1, 96], dtype="float32")
        self.sin.stop_gradient = False

        self.position_ids = paddle.arange(end=2048, dtype="int64").unsqueeze(0)
        self.position_ids.stop_gradient = False

    def test_eval(self):
        paddle.seed(2022)
        self.prepare_data()
        net = RotaryPosEmb()
        net.eval()
        net = paddle.jit.to_static(
            net, full_graph=True, build_strategy=build_strategy
        )
        out = net(self.q, self.k, self.cos, self.sin, self.position_ids)
        print(out[0].shape)
        print(out[1].shape)
        print(out)
        # loss = (out[0] + out[1]).sum()
        # loss.backward()
        return out


if __name__ == "__main__":
    unittest.main()
