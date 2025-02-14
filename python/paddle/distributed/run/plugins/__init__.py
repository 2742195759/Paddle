# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import six

__all__ = []


def log(ctx):
    ctx.logger.info("-----------  Configuration  ----------------------")
    for arg, value in sorted(six.iteritems(vars(ctx.args))):
        ctx.logger.info("%s: %s" % (arg, value))
    ctx.logger.info("--------------------------------------------------")


def process_args(ctx):
    # reset device by args
    #argdev = ctx.args.gpus or ctx.args.xpus or ctx.args.npus
    argdev = ctx.args.devices
    if argdev:
        ctx.node.device.labels = argdev.split(',')
        ctx.node.device.count = len(ctx.node.device.labels)
        ctx.logger.debug('Device reset by args {}'.format(argdev))


def collective_compatible(ctx):
    if 'PADDLE_TRAINER_ENDPOINTS' in ctx.envs:
        ctx.master = ctx.envs['PADDLE_TRAINER_ENDPOINTS'].split(',')[0]
    if 'DISTRIBUTED_TRAINER_ENDPOINTS' in ctx.envs:
        ctx.master = ctx.envs['DISTRIBUTED_TRAINER_ENDPOINTS'].split(',')[0]


def rewrite_host_ip(ctx):
    if ctx.args.host is not None and "." in ctx.args.host:
        ctx.logger.warning('Host ip reset to {}'.format(ctx.args.host))
        ctx.node.ip = ctx.args.host


enabled_plugins = [collective_compatible, rewrite_host_ip, process_args, log]
