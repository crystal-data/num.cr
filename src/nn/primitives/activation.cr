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
  # Arguments
  # ---------
  # *x* : Tensor
  #   Tensor to activate
  #
  # Examples
  # --------
  # ```
  # a = [0.1, 0.34, 0.65].to_tensor
  # Num::NN.tanh(a) # => [0.099668, 0.327477, 0.57167 ]
  # ```
  def tanh(x : Tensor(U)) : Tensor(U) forall U
    Num.tanh(x)
  end

  # :ditto:
  def tanh!(x : Tensor(U)) forall U
    Num.tanh!(x)
  end

  # Derivative of the Tanh function
  #
  # Arguments
  # ---------
  # *x* : Tensor
  #   Tensor to derive
  #
  # Examples
  # --------
  # ```
  # a = [0.1, 0.34, 0.65].to_tensor
  # Num::NN.d_tanh(a) # => [0.990066, 0.892759, 0.673193]
  # ```
  def tanh_prime(gradient : Tensor(U), cached : Tensor(U)) forall U
    cached.map(gradient) do |x, y|
      y * (1 - x * x)
    end
  end

  # Sigmoid takes a real value as input and outputs another value
  # between 0 and 1. It’s easy to work with and has all the
  # nice properties of activation functions: it’s non-linear,
  # continuously differentiable, monotonic, and has a fixed
  # output range.
  #
  # Arguments
  # ---------
  # *x* : Tensor
  #   Tensor to activate
  #
  # Examples
  # --------
  # ```
  # a = [0.1, 0.34, 0.65].to_tensor
  # puts Num::NN.sigmoid(a) # => [0.524979, 0.584191, 0.65701 ]
  # ```
  def sigmoid(x)
    x.map do |i|
      1 / (1 + Math.exp(-i))
    end
  end

  # :ditto:
  def sigmoid!(x : Tensor(U)) : Tensor(U) forall U
    x.map! do |i|
      1 / (1 + Math.exp(-i))
    end
  end

  # Derivative of the Sigmoid function
  #
  # Arguments
  # ---------
  # *x* : Tensor
  #   Tensor to derive
  #
  # Examples
  # --------
  # ```
  # a = [0.1, 0.34, 0.65].to_tensor
  # puts Num::NN.d_sigmoid(a) # => [0.249376, 0.242912, 0.225348]
  # ```
  def sigmoid_prime(gradient : Tensor(U), cached : Tensor(U)) : Tensor(U) forall U
    cached.map(gradient) do |x, y|
      x * (1 - x) * y
    end
  end

  # ReLU activation function
  #
  # Arguments
  # ---------
  # x : Tensor(U)
  #   Argument to activate
  #
  # Returns
  # -------
  # Tensor(U)
  #
  # Examples
  # --------
  def relu(x : Tensor(U)) : Tensor(U) forall U
    Num.max(U.new(0), x)
  end

  # :ditto:
  def relu!(x : Tensor(U)) : Tensor(U) forall U
    Num.max!(U.new(0), x)
  end

  # Derivative of the ReLU activation function
  #
  # Arguments
  # ---------
  #
  # Returns
  # -------
  # nil
  #
  # Examples
  # --------
  def relu_prime(gradient : Tensor(U), cached : Tensor(U)) : Tensor(U) forall U
    cached.map(gradient) do |x, y|
      if x <= 0
        U.new(0)
      else
        y
      end
    end
  end

  # Leaky ReLU activation function
  #
  # Arguments
  # ---------
  # x : Tensor(U)
  #   Argument to activate
  #
  # Returns
  # -------
  # Tensor(U)
  #
  # Examples
  # --------
  def leaky_relu(x : Tensor(U)) : Tensor(U) forall U
    x.map do |i|
      i > 0 ? i : U.new(i * 0.01)
    end
  end

  # :ditto:
  def leaky_relu!(x : Tensor(U)) forall U
    x.map! do |i|
      i > 0 ? i : i * 0.01
    end
  end

  # Brief description of leakyreluprime
  #
  # Arguments
  # ---------
  # gradient : Tensor(U)
  #   Gradient value
  # cached : Tensor(U)
  #   Stored value
  #
  # Returns
  # -------
  # Tensor(U)
  #
  # Examples
  # --------
  def leaky_relu_prime(gradient : Tensor(U), cached : Tensor(U)) : Tensor(U) forall U
    cached.map(gradient) do |x, y|
      if x <= 0
        y * 0.01
      else
        y
      end
    end
  end

  # Exponential linear unit activation
  #
  # Arguments
  # ---------
  # x : Tensor(U)
  #   Argument to activate
  #
  # Returns
  # -------
  # Tensor(U)
  #
  # Examples
  # --------
  def elu(x : Tensor(U), alpha = 0.01) : Tensor(U) forall U
    x.map do |i|
      if i > 0
        i
      else
        U.new(alpha * (Math::E ** i - 1))
      end
    end
  end

  # :ditto:
  def elu!(x : Tensor(U), alpha = 0.01) : Tensor(U) forall U
    x.map! do |i|
      if i > 0
        i
      else
        alpha * (Math::E ** i - 1)
      end
    end
  end

  # Derivative of the ELU activation
  #
  # Arguments
  # ---------
  # gradient : Tensor(U)
  #   Gradient value
  # cached : Tensor(U)
  #   Stored value
  #
  # Returns
  # -------
  # Tensor(U)
  #
  # Examples
  # --------
  def elu_prime(gradient : Tensor(U), cached : Tensor(U)) : Tensor(U) forall U
    cached.map(gradient) do |x, y|
      if x <= 0
        Math.exp(y)
      else
        U.new(1)
      end
    end
  end

  # Sigmoid cross entropy loss
  #
  # Arguments
  # ---------
  # input : Tensor(U)
  #   Predicted values
  # target : Tensor(U)
  #   Truth values
  #
  # Returns
  # -------
  # Tensor(U)
  #
  # Examples
  # --------
  def sigmoid_cross_entropy(input : Tensor(U), target : Tensor(U)) : U forall U
    batch_size = input.shape[0]
    result = input.map(target) do |x, y|
      -y * x + Math.max(x, U.new(0)) + Math.log1p(Math.exp(-x.abs))
    end
    result.sum / U.new(batch_size)
  end

  # Mean squared error loss
  #
  # Arguments
  # ---------
  # input : Tensor(U)
  #   Predicted values
  # target : Tensor(U)
  #   Truth values
  #
  # Returns
  # -------
  # Tensor(U)
  #
  # Examples
  # --------
  def mse(input : Tensor(U), target : Tensor(U)) : Tensor(U) forall U
    result = input.map(target) do |i, j|
      (i - j) ** 2
    end
    [U.new(result.mean)].to_tensor
  end

  def softmax(input : Tensor(U)) : Tensor(U) forall U
    exp_x = Num.exp(input - input.max(axis: -1, dims: true))
    exp_x / exp_x.sum(axis: -1, dims: true)
  end

  def softmax_prime(gradient : Tensor(U), cached : Tensor(U)) : Tensor(U) forall U
    soft = softmax(cached)
    gradient - soft
  end
end
