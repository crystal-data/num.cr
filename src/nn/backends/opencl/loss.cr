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

  # Computes gradients of sigmoid cross entropy loss
  #
  # ## Arguments
  #
  # * gradient : `Tensor` - `Tensor` gradient computed from SCE forwards
  # * cache : `Tensor` 4D - Cached `Tensor` from activation
  # * target : `Tensor` - `Tensor` truth values
  def sigmoid_cross_entropy_backwards(
    gradient : Tensor(U, OCL(U)),
    cache : Tensor(U, OCL(U)),
    target : Tensor(U, OCL(U))
  ) forall U
    {% if U == Float32 %}
      singleton = Float32SigmoidCrossEntropyBackwardKernel.instance
      singleton.call(gradient, cache, target)
    {% elsif U == Float64 %}
      singleton = Float64SigmoidCrossEntropyBackwardKernel.instance
      singleton.call(gradient, cache, target)
    {% else %}
      \{% raise "Invalid Dtype" %}
    {% end %}
  end
end
