# Copyright (c) 2020 Crystal Data Contributors
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

module Num::NN
  extend self

  # Tanh squashes a real-valued number to the range [-1, 1]. It’s non-linear.
  # But unlike Sigmoid, its output is zero-centered. Therefore, in practice
  # the tanh non-linearity is always preferred to the sigmoid nonlinearity.
  #
  # ## Arguments
  #
  # * x : `Tensor` - `Tensor` to activate
  #
  # ## Examples
  #
  # ```
  # a = [0.1, 0.34, 0.65].to_tensor
  # Num::NN.tanh(a) # => [0.099668, 0.327477, 0.57167 ]
  # ```
  def tanh(x : Tensor(U, OCL(U))) : Tensor(U, OCL(U)) forall U
    Num.tanh(x)
  end

  # :ditto:
  def tanh!(x : Tensor(U, OCL(U))) forall U
    Num.tanh!(x)
  end

  # Derivative of the Tanh function
  #
  # ## Arguments
  #
  # * x : `Tensor` - `Tensor` to derive
  # * cached : `Tensor` - Cached `Tensor` from activation
  #
  # ## Examples
  #
  # ```
  # a = [0.1, 0.34, 0.65].to_tensor
  # Num::NN.d_tanh(a) # => [0.990066, 0.892759, 0.673193]
  # ```
  def tanh_prime(gradient : Tensor(U, OCL(U)), cached : Tensor(U, OCL(U))) forall U
    call_opencl_kernel(
      U,
      TanhBackwardKernel,
      [Float32, Float64],
      gradient, cached
    )
  end

  # Sigmoid takes a real value as input and outputs another value
  # between 0 and 1. It’s easy to work with and has all the
  # nice properties of activation functions: it’s non-linear,
  # continuously differentiable, monotonic, and has a fixed
  # output range.
  #
  # ## Arguments
  #
  # * x : `Tensor` - `Tensor` to activate
  #
  # ## Examples
  #
  # ```
  # a = [0.1, 0.34, 0.65].to_tensor
  # puts Num::NN.sigmoid(a) # => [0.524979, 0.584191, 0.65701 ]
  # ```
  def sigmoid(x : Tensor(U, OCL(U))) forall U
    call_opencl_kernel(
      U,
      SigmoidKernel,
      [Float32, Float64],
      x
    )
  end

  # :ditto:
  def sigmoid!(x : Tensor(U, OCL(U))) forall U
    call_opencl_kernel(
      U,
      SigmoidInplaceKernel,
      [Float32, Float64],
      x
    )
  end

  # Derivative of the Sigmoid function
  #
  # ## Arguments
  #
  # * gradient : `Tensor` - `Tensor` to derive
  # * cached : `Tensor` - Cached `Tensor` from activation
  #
  # ## Examples
  #
  # ```
  # a = [0.1, 0.34, 0.65].to_tensor
  # puts Num::NN.d_sigmoid(a) # => [0.249376, 0.242912, 0.225348]
  # ```
  def sigmoid_prime(
    gradient : Tensor(U, OCL(U)),
    cached : Tensor(U, OCL(U))
  ) : Tensor(U, OCL(U)) forall U
    call_opencl_kernel(
      U,
      SigmoidBackwardKernel,
      [Float32, Float64],
      gradient, cached
    )
  end

  # ReLU activation function
  #
  # ## Arguments
  #
  # * x : `Tensor` - Argument to activate
  def relu(x : Tensor(U, OCL(U))) : Tensor(U, OCL(U)) forall U
    call_opencl_kernel(
      U,
      ReluKernel,
      [Float32, Float64],
      x
    )
  end

  # :ditto:
  def relu!(x : Tensor(U, OCL(U))) forall U
    call_opencl_kernel(
      U,
      ReluInplaceKernel,
      [Float32, Float64],
      x
    )
  end

  # Derivative of the ReLU activation function
  #
  # ## Arguments
  #
  # * gradient : `Tensor` - `Tensor` to derive
  # * cached : `Tensor` Cached `Tensor` from activation
  def relu_prime(
    gradient : Tensor(U, OCL(U)),
    cached : Tensor(U, OCL(U))
  ) : Tensor(U, OCL(U)) forall U
    call_opencl_kernel(
      U,
      ReluBackwardKernel,
      [Float32, Float64],
      cached, gradient
    )
  end

  # Leaky ReLU activation function
  #
  # ## Arguments
  #
  # * x : `Tensor` - Argument to activate
  def leaky_relu(x : Tensor(U, OCL(U))) : Tensor(U, OCL(U)) forall U
    call_opencl_kernel(
      U,
      LeakyReluKernel,
      [Float32, Float64],
      x
    )
  end

  # :ditto:
  def leaky_relu!(x : Tensor(U, OCL(U))) forall U
    call_opencl_kernel(
      U,
      LeakyReluInplaceKernel,
      [Float32, Float64],
      x
    )
  end

  # Leaky ReLU derivative
  #
  # ## Arguments
  #
  # * gradient : `Tensor` - `Tensor` to derive
  # * cached : `Tensor` - Cached `Tensor` from activation
  def leaky_relu_prime(
    gradient : Tensor(U, OCL(U)),
    cached : Tensor(U, OCL(U))
  ) : Tensor(U, OCL(U)) forall U
    call_opencl_kernel(
      U,
      LeakyReluBackwardKernel,
      [Float32, Float64],
      gradient, cached
    )
  end

  # Sigmoid cross entropy loss
  #
  # ## Arguments
  #
  # * input : `Tensor` - Predicted values
  # * target : `Tensor` - Truth values
  def sigmoid_cross_entropy(input : Tensor(U, OCL(U)), target : Tensor(U, OCL(U))) forall U
    batch_size = input.shape[0]
    result = call_opencl_kernel(
      U,
      SigmoidCrossEntropyKernel,
      [Float32, Float64],
      input, target
    )

    ones = Tensor(U, OCL(U)).ones_like(result)
    summed = result.dot(ones)
    result = summed / U.new(batch_size)
    result
  end
end
