# Copyright (c) 2021 Crystal Data Contributors
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

describe Num::NN do
  it "finds a simple convolution im2colgemm_conv2d" do
    input = [
      [1, 2, 0, 0],
      [5, 3, 0, 4],
      [0, 0, 0, 7],
      [9, 3, 0, 0],
    ].to_tensor.reshape([1, 1, 4, 4]).as_type(Float32)
    kernel = [
      [1, 1, 1],
      [1, 1, 0],
      [1, 0, 0],
    ].to_tensor.reshape([1, 1, 3, 3]).as_type(Float32)
    target = [
      [1, 8, 5, 0],
      [8, 11, 5, 4],
      [8, 17, 10, 11],
      [9, 12, 10, 7],
    ].to_tensor.reshape([1, 1, 4, 4]).as_type(Float32)
    bias = [0].to_tensor.reshape(1, 1, 1).as_type(Float32)

    result = Num::NN.im2colgemm_conv2d(input, kernel, bias, padding: {1, 1})
    Num::Testing.tensor_equal(result, target).should be_true
  end

  it "finds a simple convolution conv2d nnpack", tags: "nnpack" do
    input = [
      [1, 2, 0, 0],
      [5, 3, 0, 4],
      [0, 0, 0, 7],
      [9, 3, 0, 0],
    ].to_tensor.reshape([1, 1, 4, 4]).as_type(Float32)
    kernel = [
      [1, 1, 1],
      [1, 1, 0],
      [1, 0, 0],
    ].to_tensor.reshape([1, 1, 3, 3]).as_type(Float32)
    target = [
      [1, 8, 5, 0],
      [8, 11, 5, 4],
      [8, 17, 10, 11],
      [9, 12, 10, 7],
    ].to_tensor.reshape([1, 1, 4, 4]).as_type(Float32)
    bias = [0].to_tensor.reshape(1, 1, 1).as_type(Float32)

    result = Num::NN.conv2d(input, kernel, bias, padding: {1, 1})
    Num::Testing.tensor_equal(result, target, tolerance: 1e-3).should be_true
  end

  it "finds a strided convolution im2colgemm_conv2d" do
    input = [
      [
        [
          [2, 2, 0, 2, 1],
          [0, 1, 1, 0, 2],
          [1, 2, 1, 2, 1],
          [2, 2, 0, 0, 2],
          [2, 1, 1, 1, 2],
        ], [
          [2, 0, 1, 1, 1],
          [2, 2, 0, 0, 2],
          [2, 2, 1, 0, 0],
          [1, 1, 2, 2, 0],
          [2, 1, 1, 1, 0],
        ], [
          [0, 1, 2, 2, 0],
          [1, 1, 1, 1, 0],
          [2, 1, 2, 2, 0],
          [0, 2, 2, 2, 1],
          [0, 0, 2, 2, 1],
        ],
      ],
    ].to_tensor.as_type(Float32)

    kernel =
      [
        [
          [
            [-1, -1, -1],
            [1, 0, 1],
            [0, -1, 0],
          ], [
            [1, 0, -1],
            [1, -1, 1],
            [0, 1, 0],
          ], [
            [0, 0, 1],
            [-1, -1, -1],
            [-1, 0, 0],
          ],
        ], [
          [
            [0, 1, 0],
            [1, -1, -1],
            [1, 1, -1],
          ], [
            [-1, 0, 1],
            [-1, -1, 1],
            [1, 1, 0],
          ], [
            [0, 1, 1],
            [-1, 1, -1],
            [-1, -1, 0],
          ],
        ],
      ].to_tensor.as_type(Float32)

    target =
      [
        [
          [2, -2, 0],
          [-3, 2, -5],
          [-2, -1, 0],
        ], [
          [-7, 1, 0],
          [3, -3, 2],
          [1, 3, -2],
        ],
      ].to_tensor.as_type(Float32).reshape([1, 2, 3, 3])

    bias = [1, 0].to_tensor.as_type(Float32).reshape([2, 1, 1])

    result = Num::NN.im2colgemm_conv2d(input, kernel, bias, padding: {1, 1}, stride: {2, 2})
    Num::Testing.tensor_equal(result, target).should be_true
  end

  it "Finds the correct gradient for convolution forward + backward" do
    input = Tensor.random(0_f32...1_f32, [2, 3, 4, 5])
    kernel = Tensor.random(0_f32...1_f32, [2, 3, 3, 3])
    bias = Tensor.random(0_f32...1_f32, [2, 1, 1])
    padding = {1, 1}
    stride = {1, 1}

    output = Num::NN.conv2d(input, kernel, bias, padding, stride)

    dinput = input.as_type(Float64)
    dkernel = kernel.as_type(Float64)
    dbias = bias.as_type(Float64)

    target_grad_input = Num::NN.numerical_gradient(
      dinput,
      ->(x : Tensor(Float64, CPU(Float64))) { Num::NN.im2colgemm_conv2d(x, dkernel, dbias, padding, stride).sum }
    )

    target_grad_weight = Num::NN.numerical_gradient(
      dkernel,
      ->(w : Tensor(Float64, CPU(Float64))) { Num::NN.im2colgemm_conv2d(dinput, w, dbias, padding, stride).sum }
    )

    target_grad_bias = Num::NN.numerical_gradient(
      dbias,
      ->(b : Tensor(Float64, CPU(Float64))) { Num::NN.im2colgemm_conv2d(dinput, dkernel, b, padding, stride).sum }
    )

    grad_output = Tensor(Float32, CPU(Float32)).ones(output.shape)

    grad_input, grad_weight, grad_bias = Num::NN.im2colgemm_conv2d_gradient(
      input, kernel, bias, grad_output, padding, stride
    )

    Num::NN.mean_relative_error(grad_bias.as_type(Float64), target_grad_bias).should be < 1e-6
    Num::NN.mean_relative_error(grad_input.as_type(Float64), target_grad_input).should be < 1e-6
    Num::NN.mean_relative_error(grad_weight.as_type(Float64), target_grad_weight).should be < 1e-6
  end

  it "Finds the correct gradient for convolution forward + backward nnpack", tags: "nnpack" do
    input = Tensor.random(0_f32...1_f32, [2, 3, 4, 5])
    kernel = Tensor.random(0_f32...1_f32, [2, 3, 3, 3])
    bias = Tensor.random(0_f32...1_f32, [2, 1, 1])
    padding = {1, 1}
    stride = {1, 1}

    output = Num::NN.conv2d(input, kernel, bias, padding, stride)

    dinput = input.as_type(Float64)
    dkernel = kernel.as_type(Float64)
    dbias = bias.as_type(Float64)

    target_grad_input = Num::NN.numerical_gradient(
      dinput,
      ->(x : Tensor(Float64, CPU(Float64))) { Num::NN.im2colgemm_conv2d(x, dkernel, dbias, padding, stride).sum }
    )

    target_grad_weight = Num::NN.numerical_gradient(
      dkernel,
      ->(w : Tensor(Float64, CPU(Float64))) { Num::NN.im2colgemm_conv2d(dinput, w, dbias, padding, stride).sum }
    )

    target_grad_bias = Num::NN.numerical_gradient(
      dbias,
      ->(b : Tensor(Float64, CPU(Float64))) { Num::NN.im2colgemm_conv2d(dinput, dkernel, b, padding, stride).sum }
    )

    grad_output = Tensor(Float32, CPU(Float32)).ones(output.shape)

    grad_input, grad_weight, grad_bias = Num::NN.conv2d_backward(
      input, kernel, bias, grad_output, padding, stride
    )

    Num::NN.mean_relative_error(grad_bias.as_type(Float64), target_grad_bias).should be < 1e-6
    Num::NN.mean_relative_error(grad_input.as_type(Float64), target_grad_input).should be < 1e-6
    Num::NN.mean_relative_error(grad_weight.as_type(Float64), target_grad_weight).should be < 1e-6
  end
end
