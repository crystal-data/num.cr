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
  def tanh(x : Tensor(Float32, OCL(Float32))) : Tensor(Float32, OCL(Float32)) forall U
    Num.tanh(x)
  end

  # :ditto:
  def tanh!(x : Tensor(Float32, OCL(Float32))) forall U
    Num.tanh!(x)
  end

  # :nodoc:
  private def opencl_backwards_template(
    fn,
    a : Tensor(Float32, OCL(Float32)),
    b : Tensor(Float32, OCL(Float32))
  )
    result = Tensor(Float32, OCL(Float32)).zeros_like(a)

    Cl.args(
      fn, result.rank, result.size,
      result.data.shape, result.data.strides, result.offset, result.data.to_unsafe,
      a.data.shape, a.data.strides, a.offset, a.data.to_unsafe,
      b.data.shape, b.data.strides, b.offset, b.data.to_unsafe,
    )

    Cl.run(Num::ClContext.instance.queue, fn, result.size)
    result
  end

  # :nodoc:
  private def opencl_forwards(
    fn,
    a : Tensor(Float32, OCL(Float32))
  )
    result = Tensor(Float32, OCL(Float32)).zeros_like(a)

    Cl.args(
      fn, result.rank, result.size,
      result.data.shape, result.data.strides, result.offset, result.data.to_unsafe,
      a.data.shape, a.data.strides, a.offset, a.data.to_unsafe
    )

    Cl.run(Num::ClContext.instance.queue, fn, result.size)
    result
  end

  # :nodoc:
  private def opencl_forwards_inplace(
    fn,
    a : Tensor(Float32, OCL(Float32))
  )
    Cl.args(
      fn, a.rank, a.size,
      a.data.shape, a.data.strides, a.offset, a.data.to_unsafe
    )

    Cl.run(Num::ClContext.instance.queue, fn, a.size)
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
  def tanh_prime(gradient : Tensor(Float32, OCL(Float32)), cached : Tensor(Float32, OCL(Float32)))
    opencl_backwards_template(Num::OpenCLKernelCache.tanhBackwards, gradient, cached)
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
  def sigmoid(x : Tensor(Float32, OCL(Float32)))
    opencl_forwards(Num::OpenCLKernelCache.sigmoidForwards, x)
  end

  # :ditto:
  def sigmoid!(x : Tensor(Float32, OCL(Float32)))
    opencl_forwards_inplace(Num::OpenCLKernelCache.sigmoidForwardsInplace, x)
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
    gradient : Tensor(Float32, OCL(Float32)),
    cached : Tensor(Float32, OCL(Float32))
  ) : Tensor(Float32, OCL(Float32))
    opencl_backwards_template(
      Num::OpenCLKernelCache.sigmoidBackwards,
      gradient,
      cached
    )
  end

  # ReLU activation function
  #
  # ## Arguments
  #
  # * x : `Tensor` - Argument to activate
  def relu(x : Tensor(Float32, OCL(Float32))) : Tensor(Float32, OCL(Float32))
    opencl_forwards(Num::OpenCLKernelCache.reluForwards, x)
  end

  # :ditto:
  def relu!(x : Tensor(Float32, OCL(Float32))) : Tensor(Float32, OCL(Float32))
    opencl_forwards_inplace(Num::OpenCLKernelCache.reluForwardsInplace, x)
  end

  # Derivative of the ReLU activation function
  #
  # ## Arguments
  #
  # * gradient : `Tensor` - `Tensor` to derive
  # * cached : `Tensor` Cached `Tensor` from activation
  def relu_prime(
    gradient : Tensor(Float32, OCL(Float32)),
    cached : Tensor(Float32, OCL(Float32))
  ) : Tensor(Float32, OCL(Float32))
    opencl_backwards_template(
      Num::OpenCLKernelCache.reluBackwards,
      cached,
      gradient,
    )
  end

  # Leaky ReLU activation function
  #
  # ## Arguments
  #
  # * x : `Tensor` - Argument to activate
  def leaky_relu(x : Tensor(Float32, OCL(Float32))) : Tensor(Float32, OCL(Float32))
    opencl_forwards(Num::OpenCLKernelCache.leakyReluForwards, x)
  end

  # :ditto:
  def leaky_relu!(x : Tensor(Float32, OCL(Float32)))
    opencl_forwards_inplace(Num::OpenCLKernelCache.leakyReluForwardsInplace, x)
  end

  # Leaky ReLU derivative
  #
  # ## Arguments
  #
  # * gradient : `Tensor` - `Tensor` to derive
  # * cached : `Tensor` - Cached `Tensor` from activation
  def leaky_relu_prime(
    gradient : Tensor(Float32, OCL(Float32)),
    cached : Tensor(Float32, OCL(Float32))
  ) : Tensor(Float32, OCL(Float32))
    opencl_backwards_template(
      Num::OpenCLKernelCache.leakyReluBackwards,
      gradient,
      cached,
    )
  end

  # Sigmoid cross entropy loss
  #
  # ## Arguments
  #
  # * input : `Tensor` - Predicted values
  # * target : `Tensor` - Truth values
  def sigmoid_cross_entropy(input : Tensor(Float32, OCL(Float32)), target : Tensor(Float32, OCL(Float32)))
    batch_size = input.shape[0]
    result = opencl_backwards_template(
      Num::OpenCLKernelCache.sigmoidCrossEntropyLoss,
      input,
      target
    )

    ones = Tensor(Float32, OCL(Float32)).ones_like(result)
    summed = result.dot(ones)
    ones.data.finalize
    result = summed / batch_size
    summed.data.finalize
    result
  end

  # Mean squared error loss
  #
  # ## Arguments
  #
  # * input : `Tensor` - Predicted values
  # * target : `Tensor` - Truth values
  def mse(input : Tensor(Float32, OCL(Float32)), target : Tensor(Float32, OCL(Float32)))
    result = opencl_backwards_template(
      Num::OpenCLKernelCache.mseLoss,
      input,
      target
    )

    ones = Tensor(Float32, OCL(Float32)).ones_like(result)
    summed = result.dot(ones)
    ones.data.finalize
    result = summed / input.size
    summed.data.finalze
    result
  end
end
