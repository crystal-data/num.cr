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

require "../array/array"
require "../base/array"

class Tensor(T) < AnyArray(T)
  # Reduce a `Tensor` to a scalar by summing all of its values
  #
  # Example
  # ```
  # t = Tensor.range(1, 10)
  # puts t.sum
  # ```
  #
  # Output
  # ```
  # 45
  # ```
  def sum
    iter.reduce(0) { |i, j| i + j.value }
  end

  # Reduce a `Tensor`, summing along an axis of the `Tensor`
  #
  # Example
  # ```
  # t = Tensor(Int32).ones([3, 3])
  # puts t.sum(0)
  # ```
  #
  # Output
  # ```
  # [3, 3, 3]
  # ```
  def sum(axis : Int32, keepdims = false)
    reduce_fast(axis, keepdims) { |i, j| i + j }
  end

  # Reduce a `Tensor` to a scalar by multipling all of its values
  #
  # Example
  # ```
  # t = Tensor.range(1, 10)
  # puts t.prod
  # ```
  #
  # Output
  # ```
  # 362880
  # ```
  def prod
    iter.reduce(1) { |i, j| i * j.value }
  end

  # Reduce a `Tensor`, multiplying along an axis of the `Tensor`
  #
  # Example
  # ```
  # t = [[1, 2], [3, 4], [6, 7]].to_tensor
  # puts t.prod(1, keepdims: true)
  # ```
  #
  # Output
  # ```
  # [[2],
  #  [12],
  #  [42]]
  # ```
  def prod(axis : Int32, keepdims = false)
    reduce_fast(axis, keepdims) { |i, j| i * j }
  end

  # Reduce a `Tensor` to a boolean by checking the truthiness
  # of all values
  #
  # Example
  # ```
  # t = [True, False, True].to_tensor
  # puts t.all
  # ```
  #
  # Output
  # ```
  # False
  # ```
  def all
    astype(Bool).iter.reduce(true) { |i, j| i & j.value }
  end

  # Reduce a `Tensor` by checking the truthiness
  # of all values along an axis
  #
  # Example
  # ```
  # t = [[true, true], [false, false]].to_tensor
  # puts t.all(1)
  # ```
  #
  # Output
  # ```
  # [true, false]
  # ```
  def all(axis : Int32, keepdims = false)
    astype(Bool).reduce_fast(axis, keepdims) { |i, j| i & j }
  end

  # Reduce a `Tensor` to a boolean by checking the truthiness
  # of any value
  #
  # Example
  # ```
  # t = [True, False, True].to_tensor
  # puts t.any
  # ```
  #
  # Output
  # ```
  # True
  # ```
  def any
    astype(Bool).iter.reduce(false) { |i, j| i | j.value }
  end

  # Reduce a `Tensor` by checking the truthiness
  # of any values along an axis
  #
  # Example
  # ```
  # t = [[true, true], [false, false]].to_tensor
  # puts t.any(0)
  # ```
  #
  # Output
  # ```
  # [true, true]
  # ```
  def any(axis : Int32, keepdims = false)
    astype(Bool).reduce_fast(axis, keepdims) { |i, j| i | j }
  end

  # Reduce a `Tensor` to a scalar by finding the average value
  #
  # Example
  # ```
  # t = Tensor.range(1, 10)
  # puts t.mean
  # ```
  #
  # Output
  # ```
  # 5.0
  # ```
  def mean
    sum / size
  end

  # Reduce a `Tensor` along an axis by calculating the mean
  # along the provided axis
  #
  # Example
  # ```
  # t = Tensor.range(10).reshape(5, 2)
  # puts t.mean(1)
  # ```
  #
  # Output
  # ```
  # [0.5, 2.5, 4.5, 6.5, 8.5]
  # ```
  def mean(axis : Int32, keepdims = false)
    sum(axis, keepdims) / shape[axis]
  end

  # Reduce a `Tensor` to a scalar by finding the maximum value
  #
  # Example
  # ```
  # t = Tensor.range(1, 10)
  # puts t.max
  # ```
  #
  # Output
  # ```
  # 9
  # ```
  def max
    mx = uninitialized T
    iter.each_with_index do |el, i|
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

  # Reduce a `Tensor` by finding the maximum value along
  # an axis
  #
  # Example
  # ```
  # t = Tensor.range(10).reshape(5, 2)
  # puts t.max(1)
  # ```
  #
  # Output
  # ```
  # [1, 3, 5, 7, 9]
  # ```
  def max(axis : Int32, keepdims = false)
    reduce_fast(axis, keepdims) do |i, j|
      Math.max(i, j)
    end
  end

  # Reduce a `Tensor` to a scalar by finding the minimum value
  #
  # Example
  # ```
  # t = Tensor.range(1, 10)
  # puts t.min
  # ```
  #
  # Output
  # ```
  # 1
  # ```
  def min
    mx = uninitialized T
    iter.each_with_index do |el, i|
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

  # Reduce a `Tensor` by finding the minimum value along
  # an axis
  #
  # Example
  # ```
  # t = Tensor.range(10).reshape(5, 2)
  # puts t.min(1)
  # ```
  #
  # Output
  # ```
  # [0, 2, 4, 6, 8]
  # ```
  def min(axis : Int32, keepdims = false)
    reduce_fast(axis, keepdims) do |i, j|
      Math.min(i, j)
    end
  end
end
