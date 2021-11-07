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
  # Compute numerical gradient for any function w.r.t. to an input Tensor,
  # useful for gradient checking, recommend using float64 types to assure
  # numerical precision. The gradient is calculated as:
  # (f(x + h) - f(x - h)) / (2*h)
  # where h is a small number, typically 1e-5
  # f(x) will be called for each input elements with +h and -h pertubation.
  # Iterate over all elements calculating each partial derivative
  def numerical_gradient(input : Tensor(U, CPU(U)), f : Proc(Tensor(U, CPU(U)), U), h : U = U.new(1e-5)) forall U
    result = Tensor(U, CPU(U)).new(input.shape)
    raw = result.to_unsafe
    x = input
    x.each_pointer_with_index do |ptr, i|
      orig_val = ptr.value
      ptr.value = orig_val + h
      fa = f.call(x)
      ptr.value = orig_val - h
      fb = f.call(x)
      ptr.value = orig_val
      raw[i] = (fa - fb) / (U.new(2.0) * h)
    end

    result
  end

  # Compute numerical gradient for any function w.r.t. to an input value,
  # useful for gradient checking, recommend using float64 types to assure
  # numerical precision. The gradient is calculated as:
  # (f(x + h) - f(x - h)) / (2*h)
  # where h is a small number, typically 1e-5.
  def numerical_gradient(input : Float, f : Proc(Float, Float), h : Float = 1e-5) : Float
    (f.call(input + h) - f.call(input - h)) / (2.0 * h)
  end

  # Mean relative error for Tensor, mean of the element-wise
  # |y_true - y|/max(|y_true|, |y|)
  # Normally the relative error is defined as |y_true - y| / |y_true|,
  # but here max is used to make it symmetric and to prevent dividing by zero,
  # guaranteed to return zero in the case when both values are zero.
  def mean_relative_error(y : Tensor(U, CPU(U)), y_true : Tensor(U, CPU(U))) forall U
    result = y.map(y_true) do |i, j|
      denom = {j.abs, i.abs}.max
      if denom == 0
        U.new(0)
      else
        (j - i).abs / denom
      end
    end
    result.mean
  end
end
