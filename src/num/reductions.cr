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

require "../tensor/tensor"
require "./math"

module Num
  extend self

  def sum(a : Tensor)
    a.iter.reduce(0) { |i, j| i + j.value }
  end

  def sum(a : Tensor, axis : Int32, keepdims = false)
    a.reduce_fast(axis, keepdims) { |i, j| i + j }
  end

  def prod(a : Tensor)
    a.iter.reduce(1) { |i, j| i * j.value }
  end

  def prod(a : Tensor, axis : Int32, keepdims = false)
    a.reduce_fast(axis, keepdims) { |i, j| i * j }
  end

  def all(a : Tensor)
    a.astype(Bool).iter.reduce(true) { |i, j| i & j.value }
  end

  def all(a : Tensor, axis : Int32, keepdims = false)
    a.astype(Bool).reduce_fast(axis, keepdims) { |i, j| i & j }
  end

  def any(a : Tensor)
    a.astype(Bool).iter.reduce(false) { |i, j| i | j.value }
  end

  def any(a : Tensor, axis : Int32, keepdims = false)
    a.astype(Bool).reduce_fast(axis, keepdims) { |i, j| i | j }
  end

  def mean(a : Tensor)
    divide(sum(a), a.size)
  end

  def mean(a : Tensor, axis : Int32, keepdims = false)
    divide(sum(a, axis, keepdims), a.shape[axis])
  end

  def std(a : Tensor)
    Math.sqrt(divide(sum(power(subtract(a, mean(a)), 2)), a.size))
  end

  def std(a : Tensor, axis : Int32, keepdims = false)
    sqrt(divide(sum(power(subtract(a, mean(a, axis, keepdims: true)), 2), axis, keepdims), a.shape[axis]))
  end

  def max(a : Tensor(U)) forall U
    mx = uninitialized U
    a.iter.each_with_index do |el, i|
      c = el.value
      if i == 0
        mx = c
      end
      if c > mx
        mx = c
      end
    end
    mx
  end

  def max(a : Tensor, axis : Int32, keepdims = false)
    a.reduce_fast(axis, keepdims) do |i, j|
      Math.max(i, j)
    end
  end
end
