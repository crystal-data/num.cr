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

module Num::NN
  extend self

  # Computes gradients of mean squared error loss
  #
  # ## Arguments
  #
  # * gradient : `Tensor` - `Tensor` gradient computed from MSE forwards
  # * cache : `Tensor` 4D - Cached `Tensor` from activation
  # * target : `Tensor` - `Tensor` truth values
  def mse_backwards(
    gradient : Tensor(U, CPU(U)),
    cache : Tensor(U, CPU(U)),
    target : Tensor(U, CPU(U))
  ) forall U
    norm = gradient.value * 2 / gradient.size
    result = cache.map(target) do |x, y|
      norm * (x - y)
    end
    [result]
  end

  # Computes gradients of sigmoid cross entropy loss
  #
  # ## Arguments
  #
  # * gradient : `Tensor` - `Tensor` gradient computed from SCE forwards
  # * cache : `Tensor` 4D - Cached `Tensor` from activation
  # * target : `Tensor` - `Tensor` truth values
  def sigmoid_cross_entropy_backwards(
    gradient : Tensor(U, CPU(U)),
    cache : Tensor(U, CPU(U)),
    target : Tensor(U, CPU(U))
  ) forall U
    grad = gradient.value
    batch_size = cache.shape[0]
    output = cache.map(target) do |x, y|
      grad * ((1 / (1 + Math.exp(-x))) - y) / batch_size
    end
    [output]
  end

  # Computes gradients of SmCE loss
  #
  # ## Arguments
  #
  # * gradient : `Tensor` - `Tensor` gradient computed from SmCE forwards
  # * cache : `Tensor` 4D - Cached `Tensor` from activation
  # * target : `Tensor` - `Tensor` truth values
  def softmax_cross_entropy_backward(
    gradient : Tensor(U, CPU(U)),
    cached : Tensor(U, CPU(U)),
    target : Tensor(U, CPU(U))
  ) forall U
    n = cached.shape[0]
    grad = gradient.value

    result = Tensor(U, CPU(U)).zeros_like(cached)

    n.times do |i|
      mx, sumexp = Num::NN.streaming_max_sumexp(cached[i])
      res_slice = result[i]

      res_slice.map!(cached[i], target[i]) do |_, y, z|
        grad * (Num::NN.stable_softmax(y, mx, sumexp) - z) / n
      end
    end
    [result]
  end
end
