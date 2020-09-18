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
  def tanh(x : Tensor(U)) forall U
    Num.tanh(x)
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
  def d_tanh(x)
    x.map do |i|
      1 - (Math.tanh(i) ** 2)
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
  def d_sigmoid(x)
    sm = sigmoid(x)
    sm.map do |i|
      (1 - i) * i
    end
  end

  # Log loss, aka logistic loss or cross-entropy loss.
  #
  # Defined as the negative log-likelihood of a logistic model that returns
  # y probabilities for its training data y_true. The log loss is only
  # defined for two or more labels. For a single sample with true label
  # yt in {0,1} and estimated probability yp that yt = 1, the log loss is:
  #
  #   -log P(yt|yp) = -(yt log(yp) + (1 - yt) log(1 - yp))
  #
  # Arguments
  # ---------
  # *y* : Tensor
  #   Ground truth results
  # *a* : Tensor
  #   Actual results
  #
  # Examples
  # --------
  # ```
  # truth = [[0, 1, 1, 0]].to_tensor.as_type(Float64)
  # actual = [[0.0817076, 0.941659, 0.939411, 0.120442]].to_tensor
  # Num::NN.logloss(truth, actual)
  #
  # # [[0.0852394, 0.0601123, 0.062502 , 0.128336 ]]
  # ```
  def logloss(y, a)
    y.map(a) do |i, j|
      -(i * Math.log(j) + (1 - i)*Math.log(1 - j))
    end
  end

  # The derivative of logistic loss, used in back propogation
  #
  # Arguments
  # ---------
  # *y* : Tensor
  #   Ground truth results
  # *a* : Tensor
  #   Actual results
  #
  # Examples
  # --------
  # ```
  # truth = [[0, 1, 1, 0]].to_tensor.as_type(Float64)
  # actual = [[0.0817076, 0.941659, 0.939411, 0.120442]].to_tensor
  # Num::NN.d_logloss(truth, actual)
  #
  # # [[1.03625 , -1.0224 , -1.04178, 1.0528  ]]
  # ```
  def d_logloss(y, a)
    y.map(a) do |i, j|
      (j - i)/(j*(1 - j))
    end
  end
end
