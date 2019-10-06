# -*- coding: utf-8 -*-
#!/usr/bin/env python

# Copyright 2019 aiboy.wei Authors. All Rights Reserved.
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
# ==============================================================================
"""
Implementation of paper Searching for MobileNetV3, https://arxiv.org/abs/1905.02244
author: aiboy.wei@outlook.com
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

MobileNetV3_Small_Spec = [
    # Op            k    exp    out    SE     NL        s
    [ "ConvBnAct",  3,   False, 16,    False, "hswish", 2 ],
    [ "bneck",      3,   16,    16,    True,  "relu",   2 ],
    [ "bneck",      3,   72,    24,    False, "relu",   2 ],
    [ "bneck",      3,   88,    24,    False, "relu",   1 ],
    [ "bneck",      5,   96,    40,    True,  "hswish", 2 ],
    [ "bneck",      5,   240,   40,    True,  "hswish", 1 ],
    [ "bneck",      5,   240,   40,    True,  "hswish", 1 ],
    [ "bneck",      5,   120,   48,    True,  "hswish", 1 ],
    [ "bneck",      5,   144,   48,    True,  "hswish", 1 ],
    [ "bneck",      5,   288,   96,    True,  "hswish", 2 ],
    [ "bneck",      5,   576,   96,    True,  "hswish", 1 ],
    [ "bneck",      5,   576,   96,    True,  "hswish", 1 ],
    [ "ConvBnAct",  1,   False, 576,   True,  "hswish", 1 ],
    [ "pool",       7,   False, False, False, "None",   1 ],
    [ "ConvNBnAct", 1,   False, 1280,  False, "hswish", 1 ],
    [ "ConvNBnAct", 1,   False, 1000,  False, "None",   1 ],
]

MobileNetV3_Large_Spec = [
    # Op            k    exp    out    SE     NL        s
    [ "ConvBnAct",  3,   False, 16,    False, "hswish", 2 ],
    [ "bneck",      3,   16,    16,    False, "relu",   1 ],
    [ "bneck",      3,   64,    24,    False, "relu",   2 ],
    [ "bneck",      3,   72,    24,    False, "relu",   1 ],
    [ "bneck",      5,   72,    40,    True,  "relu",   2 ],
    [ "bneck",      5,   120,   40,    True,  "relu",   1 ],
    [ "bneck",      5,   120,   40,    True,  "relu",   1 ],
    [ "bneck",      3,   240,   80,    False, "hswish", 2 ],
    [ "bneck",      3,   200,   80,    False, "hswish", 1 ],
    [ "bneck",      3,   184,   80,    False, "hswish", 1 ],
    [ "bneck",      3,   184,   80,    False, "hswish", 1 ],
    [ "bneck",      3,   480,   112,   True,  "hswish", 1 ],
    [ "bneck",      3,   672,   112,   True,  "hswish", 1 ],
    [ "bneck",      5,   672,   160,   True,  "hswish", 2 ],
    [ "bneck",      5,   960,   160,   True,  "hswish", 1 ],
    [ "bneck",      5,   960,   160,   True,  "hswish", 1 ],
    [ "ConvBnAct",  1,   False, 960,   False, "hswish", 1 ],
    [ "pool",       7,   False, False, False, "None",   1 ],
    [ "ConvNBnAct", 1,   False, 1280,  False, "hswish", 1 ],
    [ "ConvNBnAct", 1,   False, 1000,  False, "None",   1 ],
]

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)

    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor

    return new_v

class Identity(tf.keras.layers.Layer):
    def __init__(self, name="Identity", **kwargs):
        super(Identity, self).__init__(name=name, **kwargs)

    def call(self, input):
        return input

    def get_config(self):
        base_config = super(Identity, self).get_config()
        return dict(list(base_config.items()))

class HardSigmoid(tf.keras.layers.Layer):
    def __init__(self, name="HardSigmoid", **kwargs):
        super(HardSigmoid, self).__init__(name=name, **kwargs)
        self.relu6 = tf.keras.layers.ReLU(max_value=6, name="ReLU6", **kwargs)

    def call(self, input):
        return self.relu6(input + 3.0) / 6.0

    def get_config(self):
        base_config = super(HardSigmoid, self).get_config()
        return dict(list(base_config.items()))

class HardSwish(tf.keras.layers.Layer):
    def __init__(self, name="HardSwish", **kwargs):
        super(HardSwish, self).__init__(name=name, **kwargs)
        self.relu6 = tf.keras.layers.ReLU(max_value=6, name="ReLU6", **kwargs)

    def call(self, input):
        return input * self.relu6(input + 3.0) / 6.0

    def get_config(self):
        base_config = super(HardSwish, self).get_config()
        return dict(list(base_config.items()))

_available_activation = {
            "relu": tf.keras.layers.ReLU(name="ReLU"),
            "relu6": tf.keras.layers.ReLU(max_value=6, name="ReLU6"),
            "hswish": HardSwish(),
            "hsigmoid": HardSigmoid(),
            "softmax": tf.keras.layers.Softmax(name="Softmax"),
            "None": Identity(),
        }

class SENet(tf.keras.layers.Layer):
    def __init__(self, reduction=4, l2=2e-4, name="SENet", **kwargs):
        super(SENet, self).__init__(name=name, **kwargs)
        self.reduction = reduction
        self.l2 = l2

    def build(self, input_shape):
        _, h, w, c = input_shape
        self.gap = tf.keras.layers.GlobalAveragePooling2D(name=f'AvgPool{h}x{w}')
        self.fc1 = tf.keras.layers.Dense(units=c//self.reduction, activation="relu", use_bias=False,
                                         kernel_regularizer=tf.keras.regularizers.l2(self.l2), name="Squeeze")
        self.fc2 = tf.keras.layers.Dense(units=c, activation=HardSigmoid(), use_bias=False,
                                         kernel_regularizer=tf.keras.regularizers.l2(self.l2), name="Excite")
        self.reshape = tf.keras.layers.Reshape((1, 1, c), name=f'Reshape_None_1_1_{c}')

        super().build(input_shape)

    def call(self, input):
        output = self.gap(input)
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.reshape(output)
        return input * output

    def get_config(self):
        config = {"reduction":self.reduction, "l2":self.l2}
        base_config = super(SENet, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class ConvBnAct(tf.keras.layers.Layer):
    def __init__(self, k, exp, out, SE, NL, s, l2, name="ConvBnAct", **kwargs):
        super(ConvBnAct, self).__init__(name=name, **kwargs)
        self.k = k
        self.exp = exp
        self.out = out
        self.se = SE
        self.nl = NL
        self.s = s
        self.l2 = l2
        self.conv2d = tf.keras.layers.Conv2D(filters=out, kernel_size=k, strides=s, activation=None, padding="same",
                                             kernel_regularizer=tf.keras.regularizers.l2(l2), name="conv2d", **kwargs)
        self.bn = tf.keras.layers.BatchNormalization(momentum=0.99, name="BatchNormalization", **kwargs)
        self.act = _available_activation[NL]

    def call(self, input):
        output = self.conv2d(input)
        output = self.bn(output)
        output = self.act(output)
        return output

    def get_config(self):
        config = {"k":self.k, "exp":self.exp, "out":self.out, "SE":self.se, "NL":self.nl, "s":self.s, "l2":self.l2}
        base_config = super(ConvBnAct, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class ConvNBnAct(tf.keras.layers.Layer):
    def __init__(self, k, exp, out, SE, NL, s, l2, name="ConvNBnAct", **kwargs):
        super(ConvNBnAct, self).__init__(name=name, **kwargs)
        self.k = k
        self.exp = exp
        self.out = out
        self.se = SE
        self.nl = NL
        self.s = s
        self.l2 = l2
        self.act = _available_activation[NL]
        self.fn = tf.keras.layers.Conv2D(filters=out, kernel_size=k, strides=s, activation=self.act, padding="same",
                                         kernel_regularizer=tf.keras.regularizers.l2(l2),name="conv2d", **kwargs)

    def call(self, input):
        output = self.fn(input)
        return output

    def get_config(self):
        config = {"k":self.k, "exp":self.exp, "out":self.out, "SE":self.se, "NL":self.nl, "s":self.s, "l2":self.l2}
        base_config = super(ConvNBnAct, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Pool(tf.keras.layers.Layer):
    def __init__(self, k, exp, out, SE, NL, s, l2, name="Pool", **kwargs):
        super(Pool, self).__init__(name=name, **kwargs)
        self.k = k
        self.exp = exp
        self.out = out
        self.se = SE
        self.nl = NL
        self.s = s
        self.l2 = l2
        self.gap = tf.keras.layers.AveragePooling2D(pool_size=(k, k), strides=1, name=f'AvgPool{k}x{k}', **kwargs)

    def call(self, input):
        output = self.gap(input)
        return output

    def get_config(self):
        config = {"k":self.k, "exp":self.exp, "out":self.out, "SE":self.se, "NL":self.nl, "s":self.s, "l2":self.l2}
        base_config = super(Pool, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class BottleNeck(tf.keras.layers.Layer):
    def __init__(self, k, exp, out, SE, NL, s, l2, name="BottleNeck", **kwargs):
        super(BottleNeck, self).__init__(name=name, **kwargs)
        self.k = k
        self.exp = exp
        self.out = out
        self.se = SE
        self.nl = NL
        self.s = s
        self.l2 = l2
        self.expand = ConvBnAct(k=1, exp=exp, out=exp, SE=SE, NL=NL, s=1, l2=l2, name="BottleNeckExpand", **kwargs)
        self.depthwise = tf.keras.layers.DepthwiseConv2D(
            kernel_size=k,
            strides=s,
            padding="same",
            use_bias=False,
            depthwise_regularizer=tf.keras.regularizers.l2(l2),
            name=f'Depthwise{k}x{k}',
            ** kwargs,
        )
        self.pointwise = tf.keras.layers.Conv2D(
            filters=out,
            kernel_size=1,
            strides=1,
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(l2),
            name=f'Pointwise1x1',
            ** kwargs,
        )
        self.bn_1 = tf.keras.layers.BatchNormalization(momentum=0.99, name="BatchNormalization_1", **kwargs)
        self.bn_2 = tf.keras.layers.BatchNormalization(momentum=0.99, name="BatchNormalization_2", **kwargs)

        if self.se:
            self.SeNet = SENet(name="SEBottleneck", l2=l2, **kwargs)

        self.act = _available_activation[NL]

    def call(self, input):
        output = self.expand(input)
        output = self.depthwise(output)
        output = self.bn_1(output)
        if self.se:
            output = self.SeNet(output)
        output = self.act(output)
        output = self.pointwise(output)
        output = self.bn_2(output)

        if self.s == 1 and self.exp == self.out:
            return input + output
        else:
            return output

    def get_config(self):
        config = {"k":self.k, "exp":self.exp, "out":self.out, "SE":self.se, "NL":self.nl, "s":self.s, "l2":self.l2}
        base_config = super(BottleNeck, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

_available_mobilenetv3_spec = {
            "small": MobileNetV3_Small_Spec,
            "large": MobileNetV3_Large_Spec,
        }

_available_operation = {
            "ConvBnAct":  ConvBnAct,
            "bneck":      BottleNeck,
            "pool":       Pool,
            "ConvNBnAct": ConvNBnAct,
        }

class CusReshape(tf.keras.layers.Layer):
    def __init__(self, out, name="Reshape", **kwargs):
        super(CusReshape, self).__init__(name=name, **kwargs)
        self.out = out
        self.reshape = tf.keras.layers.Reshape((out,), name=f'Reshape_None_{out}', **kwargs)

    def call(self, input):
        output = self.reshape(input)
        return output

    def get_config(self):
        config = {"out":self.out}
        base_config = super(CusReshape, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class CusDropout(tf.keras.layers.Layer):
    def __init__(self, dropout_rate, name="Dropout", **kwargs):
        super(CusDropout, self).__init__(name=name, **kwargs)
        self.dropout_rate = dropout_rate
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate, name=f'Dropout', **kwargs)

    def call(self, input):
        output = self.dropout(input)
        return output

    def get_config(self):
        config = {"dropout_rate":self.dropout_rate}
        base_config = super(CusDropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def MobileNetV3(type="large", input_shape=(224, 224, 3), classes_number=1000, width_multiplier=1.0,
                divisible_by=8, l2_reg=2e-5, dropout_rate=0.2, name="MobileNetV3"):
    spec = _available_mobilenetv3_spec[type]
    spec[-1][3] = classes_number  # bottlenet layer size or class numbers
    name = name + "_" + type

    inputs = tf.keras.layers.Input(shape=input_shape, name="inputs")

    for i, params in enumerate(spec):
        Op, k, exp, out, SE, NL, s = params
        inference_op = _available_operation[Op]

        if isinstance(exp, int):
            exp_ch = _make_divisible(exp * width_multiplier, divisible_by)
        else:
            exp_ch = None
        if isinstance(out, int):
            out_ch = _make_divisible(out * width_multiplier, divisible_by)
        else:
            out_ch = None
        if i == len(spec) - 1:  # fix output classes error.
            out_ch = classes_number

        op_name = f'{Op}_{i}'
        if i == 0:
            output = inference_op(k, exp_ch, out_ch, SE, NL, s, l2_reg, op_name)(inputs)
        else:
            output = inference_op(k, exp_ch, out_ch, SE, NL, s, l2_reg, op_name)(output)

        if (type == "small" and i == 14) or (type == "large" and i == 18):
            output = CusDropout(dropout_rate=dropout_rate)(output)

    outputs = CusReshape(out=classes_number)(output)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
    model.summary()

    return model

custom_objects = {
    "ConvBnAct" :  ConvBnAct,
    "BottleNeck":  BottleNeck,
    "Pool"      :  Pool,
    "ConvNBnAct":  ConvNBnAct,
    "CusReshape":  CusReshape,
    "CusDropout":  CusDropout,
}


if __name__ == "__main__":
    # if you use gpu device
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.compat.v2.config.experimental.set_memory_growth(gpu, True)

    model = MobileNetV3(type="small")

    model = MobileNetV3(type="large")
