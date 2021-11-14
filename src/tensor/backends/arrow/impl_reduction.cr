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

module Num
  extend self

  # Reduces a `Tensor` to a scalar by summing all of its
  # elements
  #
  # ## Arguments
  #
  # * a : `Tensor(U, ARROW(U))` - `Tensor` to reduce
  #
  # ## Examples
  #
  # ```
  # a = [1, 2, 3]
  # Num.sum(a) # => 6
  # ```
  @[Inline]
  def sum(a : Tensor(U, ARROW(U))) forall U
    a.reduce { |i, j| i + j }
  end

  # Reduces a `Tensor` along an axis, summing each view into
  # the `Tensor`
  #
  # ## Arguments
  #
  # * a : `Tensor(U, ARROW(U))` - `Tensor` to reduce
  # * axis : `Int` - Axis of reduction
  # * dims : `Bool` - Indicate if the axis of reduction should remain in the
  #   result
  #
  # ## Examples
  #
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # Num.sum(a, 0) # => [2, 4]
  # Num.sum(a, 1, dims: true)
  # # [[1],
  # #  [5]]
  # ```
  @[Inline]
  def sum(a : Tensor(U, ARROW(U)), axis : Int, dims : Bool = false) forall U
    a.reduce_axis(axis, dims) { |i, j| i + j }
  end

  # Reduces a `Tensor` to a scalar by multiplying all of its
  # elements
  #
  # ## Arguments
  #
  # * a : `Tensor(U, ARROW(U))` - `Tensor` to reduce
  #
  # ## Examples
  #
  # ```
  # a = [1, 2, 3]
  # Num.prod(a) # => 6
  # ```
  @[Inline]
  def prod(a : Tensor(U, ARROW(U))) forall U
    a.reduce { |i, j| i * j }
  end

  # Reduces a `Tensor` along an axis, multiplying each view into
  # the `Tensor`
  #
  # ## Arguments
  #
  # * a : `Tensor(U, ARROW(U))` - `Tensor` to reduce
  # * axis : `Int` - Axis of reduction
  # * dims : `Bool` - Indicate if the axis of reduction should remain in the
  #   result
  #
  # ## Examples
  #
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # Num.prod(a, 0) # => [0, 3]
  # Num.prod(a, 1, dims: true)
  # # [[0],
  # #  [6]]
  # ```
  @[Inline]
  def prod(a : Tensor(U, ARROW(U)), axis : Int, dims : Bool = false) forall U
    a.reduce_axis(axis, dims) { |i, j| i * j }
  end

  # Reduces a `Tensor` to a boolean by asserting the truthiness of
  # all elements
  #
  # ## Arguments
  #
  # * a : `Tensor(U, ARROW(U))` - `Tensor` to reduce
  #
  # ## Examples
  #
  # ```
  # a = [0, 2, 3]
  # Num.all(a) # => false
  # ```
  @[Inline]
  def all(a : Tensor(U, ARROW(U))) forall U
    result = a.as_type(Bool)
    result.reduce { |i, j| i & j }
  end

  # Reduces a `Tensor` along an axis, asserting the truthiness of all values
  # in each view into the `Tensor`
  #
  # ## Arguments
  #
  # * a : `Tensor(U, ARROW(U))` - `Tensor` to reduce
  # * axis : `Int` - Axis of reduction
  # * dims : `Bool` - Indicate if the axis of reduction should remain in the
  #   result
  #
  # ## Examples
  #
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # Num.all(a, 0) # => [false, true]
  # Num.all(a, 1, dims: true)
  # # [[false],
  # #  [ true]]
  # ```
  @[Inline]
  def all(a : Tensor(U, ARROW(U)), axis : Int, dims : Bool = false) forall U
    result = a.as_type(Bool)
    result.reduce_axis(axis, dims) { |i, j| i & j }
  end

  # Reduces a `Tensor` to a boolean by asserting the truthiness of
  # any element
  #
  # ## Arguments
  #
  # * a : `Tensor(U, ARROW(U))` - `Tensor` to reduce
  #
  # ## Examples
  #
  # ```
  # a = [0, 2, 3]
  # Num.any(a) # => true
  # ```
  @[Inline]
  def any(a : Tensor(U, ARROW(U))) forall U
    result = a.as_type(Bool)
    result.reduce { |i, j| i | j }
  end

  # Reduces a `Tensor` along an axis, asserting the truthiness of any values
  # in each view into the `Tensor`
  #
  # ## Arguments
  #
  # * a : `Tensor(U, ARROW(U))` - `Tensor` to reduce
  # * axis : `Int` - Axis of reduction
  # * dims : `Bool` - Indicate if the axis of reduction should remain in the
  #   result
  #
  # ## Examples
  #
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # Num.any(a, 0) # => [true, true]
  # Num.any(a, 1, dims: true)
  # # [[true],
  # #  [ true]]
  # ```
  @[Inline]
  def any(a : Tensor(U, ARROW(U)), axis : Int, dims : Bool = false) forall U
    result = a.as_type(Bool)
    result.reduce_axis(axis, dims) { |i, j| i | j }
  end

  # Reduces a `Tensor` to a scalar by finding the average
  #
  # ## Arguments
  #
  # * a : `Tensor(U, ARROW(U))` - `Tensor` to reduce
  #
  # ## Examples
  #
  # ```
  # a = [1, 2, 3]
  # Num.mean(a) # => 2.0
  # ```
  @[Inline]
  def mean(a : Tensor(U, ARROW(U))) forall U
    Num.sum(a) / a.size
  end

  # Reduces a `Tensor` along an axis, finding the average of each
  # view into the `Tensor`
  #
  # ## Arguments
  #
  # * a : `Tensor(U, ARROW(U))` - `Tensor` to reduce
  # * axis : `Int` - Axis of reduction
  # * dims : `Bool` - Indicate if the axis of reduction should remain in the
  #   result
  #
  # ## Examples
  #
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # Num.mean(a, 0) # => [1, 2]
  # Num.mean(a, 1, dims: true)
  # # [[0],
  # #  [2]]
  # ```
  @[Inline]
  def mean(a : Tensor(U, ARROW(U)), axis : Int, dims : Bool = false) forall U
    n = a.shape[axis]
    a.reduce_axis(axis, dims) { |i, j| i + j } / n
  end

  # Reduces a `Tensor` to a scalar by finding the maximum value
  #
  # ## Arguments
  #
  # * a : `Tensor(U, ARROW(U))` - `Tensor` to reduce
  #
  # ## Examples
  #
  # ```
  # a = [1, 2, 3]
  # Num.max(a) # => 3
  # ```
  @[Inline]
  def max(a : Tensor(U, ARROW(U))) forall U
    m = a.value
    a.each do |el|
      m = el if el > m
    end
    m
  end

  # Reduces a `Tensor` along an axis, finding the max of each
  # view into the `Tensor`
  #
  # ## Arguments
  #
  # * a : `Tensor(U, ARROW(U))` - `Tensor` to reduce
  # * axis : `Int` - Axis of reduction
  # * dims : `Bool` - Indicate if the axis of reduction should remain in the
  #   result
  #
  # ## Examples
  #
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # Num.max(a, 0) # => [2, 3]
  # Num.max(a, 1, dims: true)
  # # [[1],
  # #  [3]]
  # ```
  @[Inline]
  def max(a : Tensor(U, ARROW(U)), axis : Int, dims : Bool = false) forall U
    a.reduce_axis(axis, dims) { |i, j| Math.max(i, j) }
  end

  # Reduces a `Tensor` to a scalar by finding the minimum value
  #
  # ## Arguments
  #
  # * a : `Tensor(U, ARROW(U))` - `Tensor` to reduce
  #
  # ## Examples
  #
  # ```
  # a = [1, 2, 3]
  # Num.min(a) # => 3
  # ```
  @[Inline]
  def min(a : Tensor(U, ARROW(U))) forall U
    m = a.value
    a.each do |el|
      m = el if el < m
    end
    m
  end

  # Reduces a `Tensor` along an axis, finding the min of each
  # view into the `Tensor`
  #
  # ## Arguments
  #
  # * a : `Tensor(U, ARROW(U))` - `Tensor` to reduce
  # * axis : `Int` - Axis of reduction
  # * dims : `Bool` - Indicate if the axis of reduction should remain in the
  #   result
  #
  # ## Examples
  #
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # Num.min(a, 0) # => [0, 1]
  # Num.min(a, 1, dims: true)
  # # [[0],
  # #  [2]]
  # ```
  @[Inline]
  def min(a : Tensor(U, ARROW(U)), axis : Int, dims : Bool = false) forall U
    a.reduce_axis(axis, dims) { |i, j| Math.min(i, j) }
  end

  # Reduces a `Tensor` to a scalar by finding the standard deviation
  #
  # ## Arguments
  #
  # * a : `Tensor(U, ARROW(U))` - `Tensor` to reduce
  #
  # ## Examples
  #
  # ```
  # a = [1, 2, 3]
  # Num.std(a) # => 0.816496580927726
  # ```
  @[Inline]
  def std(a : Tensor(U, ARROW(U))) forall U
    avg = Num.mean(a)
    result = a.reduce(0) { |i, j| i + (j - avg) ** 2 }
    Math.sqrt(result / a.size)
  end

  # Reduces a `Tensor` along an axis, finding the std of each
  # view into the `Tensor`
  #
  # ## Arguments
  #
  # * a : `Tensor(U, ARROW(U))` - `Tensor` to reduce
  # * axis : `Int` - Axis of reduction
  # * dims : `Bool` - Indicate if the axis of reduction should remain in the
  #   result
  #
  # ## Examples
  #
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # Num.std(a, 0) # => [1, 1]
  # Num.std(a, 1, dims: true)
  # # [[0.707107],
  # #  [0.707107]]
  # ```
  @[Inline]
  def std(a : Tensor(U, ARROW(U)), axis : Int, dims : Bool = false) forall U
    result = at_axis_index(a, axis, 0, dims).as_type(Float64)
    idx = 0
    a.yield_along_axis(axis) do |ax|
      result[idx] = Num.std(ax)
      idx += 1
    end
    result
  end

  # Find the maximum index value of a Tensor
  #
  # ## Arguments
  #
  # * a : `Tensor(U, ARROW(U))` - `Tensor` to reduce
  #
  # ## Examples
  #
  # ```
  # a = [1, 10, 1].to_tensor
  # a.argmax # => 1
  # ```
  @[Inline]
  def argmax(a : Tensor(U, ARROW(U))) : Int32 forall U
    m = a.value
    idx = 0
    a.each_with_index do |el, i|
      if el > m
        m = el
        idx = i
      end
    end
    idx
  end

  # Find the maximum index value of a Tensor along
  # an axis
  #
  # ## Arguments
  #
  # * a : `Tensor(U, ARROW(U))` - `Tensor` to reduce
  # * axis : `Int` - Axis of reduction
  # * dims : `Bool` - Indicate if the axis of reduction should remain in the
  #   result
  #
  # ## Examples
  #
  # ```
  # a = [[2, 1], [1, 2]].to_tensor
  # puts a.argmax(1) # => [0, 1]
  # ```
  @[Inline]
  def argmax(a : Tensor(U, ARROW(U)), axis : Int, dims : Bool = false) forall U
    result = at_axis_index(a, axis, 0, dims).as_type(Int32)
    idx = 0
    a.yield_along_axis(axis) do |ax|
      result[idx] = Num.argmax(ax)
      idx += 1
    end
    result
  end

  # Find the minimum index value of a Tensor
  #
  # ## Arguments
  #
  # * a : `Tensor(U, ARROW(U))` - `Tensor` to reduce
  #
  # ## Examples
  #
  # ```
  # a = [10, 1, 10].to_tensor
  # a.argmin # => 1
  # ```
  @[Inline]
  def argmin(a : Tensor(U, ARROW(U))) : Int32 forall U
    m = a.value
    idx = 0
    a.each_with_index do |el, i|
      if el < m
        m = el
        idx = i
      end
    end
    idx
  end

  # Find the minimum index value of a Tensor along
  # an axis
  #
  # ## Arguments
  #
  # * a : `Tensor(U, ARROW(U))` - `Tensor` to reduce
  # * axis : `Int` - Axis of reduction
  # * dims : `Bool` - Indicate if the axis of reduction should remain in the
  #   result
  #
  # ## Examples
  #
  # ```
  # a = [[2, 1], [1, 2]].to_tensor
  # puts a.argmin(1) # => [1, 0]
  # ```
  @[Inline]
  def argmin(a : Tensor(U, ARROW(U)), axis : Int, dims : Bool = false) forall U
    result = at_axis_index(a, axis, 0, dims).as_type(Int32)
    idx = 0
    a.yield_along_axis(axis) do |ax|
      result[idx] = Num.argmin(ax)
      idx += 1
    end
    result
  end

  # Sorts a `Tensor`, treating it's elements like the `Tensor`
  # is flat.
  #
  # ## Arguments
  #
  # * a : `Tensor(U, ARROW(U))` - `Tensor` to sort
  #
  # ## Examples
  #
  # ```
  # a = [3, 2, 1].to_tensor
  # Num.sort(a) # => [1, 2, 3]
  # ```
  @[Inline]
  def sort(a : Tensor(U, ARROW(U))) forall U
    result = a.dup(Num::RowMajor)
    Slice.new(result.to_unsafe, result.size).sort!
    result
  end

  # Sorts a `Tensor`, treating it's elements like the `Tensor`
  # is flat.
  #
  # ## Arguments
  #
  # * a : `Tensor(U, ARROW(U))` - `Tensor` to sort
  # * block : `Proc(U, U, _)` - `Proc` to use to compare values
  #
  # ## Examples
  #
  # ```
  # a = [3, 2, 1].to_tensor
  # Num.sort(a) # => [1, 2, 3]
  # ```
  @[Inline]
  def sort(a : Tensor(U, ARROW(U)), &block : U, U -> _) forall U
    result = a.dup(Num::RowMajor)
    Slice.new(result.to_unsafe, result.size).sort!(&block)
    result
  end

  # Sorts a `Tensor` along an axis.
  #
  # ## Arguments
  #
  # * a : `Tensor(U, ARROW(U))` - `Tensor` to sort
  # * axis : `Int` - Axis along which to sort
  #
  # ## Examples
  #
  # ```
  # t = Tensor.random(0...10, [3, 3, 2])
  # puts Num.sort(t, axis: 1)
  #
  # # [[[1, 1],
  # #   [4, 5],
  # #   [5, 7]],
  # #
  # #  [[0, 0],
  # #   [2, 3],
  # #   [8, 4]],
  # #
  # #  [[2, 5],
  # #   [5, 7],
  # #   [5, 7]]]
  # ```
  @[Inline]
  def sort(a : Tensor(U, ARROW(U)), axis : Int) forall U
    result = a.dup(Num::RowMajor)
    result.yield_along_axis(axis) do |ax|
      ax[...] = Num.sort(ax)
    end
    result
  end

  # Sorts a `Tensor` along an axis.
  #
  # ## Arguments
  #
  # * a : `Tensor(U, ARROW(U))` - `Tensor` to sort
  # * axis : `Int` - Axis along which to sort
  # * block : `Proc(U, U, _)` - `Proc` to use to sort
  #
  # ## Examples
  #
  # ```
  # t = Tensor.random(0...10, [3, 3, 2])
  # puts Num.sort(t, axis: 1) { |i, j| i <=> j }
  # # [[[3, 1],
  # #   [4, 3],
  # #   [6, 8]],
  # #
  # #  [[4, 2],
  # #   [5, 3],
  # #   [9, 7]],
  # #
  # #  [[0, 1],
  # #   [2, 2],
  # #   [4, 9]]]
  # ```
  @[Inline]
  def sort(a : Tensor(U, ARROW(U)), axis : Int, &block : U, U -> _) forall U
    result = a.dup(Num::RowMajor)
    result.yield_along_axis(axis) do |ax|
      ax[...] = Num.sort(ax, &block)
    end
    result
  end

  # Asserts that two `Tensor`s are equal, allowing for small
  # margins of errors with floating point values using
  # an EPSILON value.
  #
  # ## Arguments
  #
  # * a : `Tensor(U, ARROW(U))` - LHS argument to compare
  # * b : `Tensor(V, ARROW(V))` - RHS argument to compare
  # * epsilon : `Number` - Allowed variance between numbers
  #
  # ## Examples
  #
  # ```
  # a = [0.0, 0.0, 0.0000000001].to_tensor
  # b = [0.0, 0.0, 0.0].to_tensor
  # Num.all_close(a, b) # => true
  # ```
  @[Inline]
  def all_close(
    a : Tensor(U, ARROW(U)),
    b : Tensor(V, ARROW(V)),
    epsilon = 1e-6
  ) : Bool forall U, V
    unless a.shape == b.shape
      return false
    end
    a.zip(b) do |i, j|
      m = (i - j).abs < epsilon
      unless m
        return false
      end
    end
    true
  end

  # Finds the difference between the maximum and minimum
  # elements of a `Tensor`
  #
  # ## Arguments
  #
  # * a : `Tensor(U, ARROW(U))` - `Tensor` to reduce
  #
  # ## Examples
  #
  # ```
  # a = [1, 2, 3]
  # Num.ptp(a) # => 2
  # ```
  @[Inline]
  def ptp(a : Tensor(U, ARROW(U))) forall U
    minimum = a.value
    maximum = a.value

    a.each do |e|
      if e < minimum
        minimum = e
      end
      if e > maximum
        maximum = e
      end
    end
    maximum - minimum
  end

  # Finds the difference between the maximum and minimum
  # elements of a `Tensor` along an axis
  #
  # ## Arguments
  #
  # * a : `Tensor(U, ARROW(U))` - `Tensor` to reduce
  # * axis : `Int` - Axis of reduction
  # * dims : `Bool` - Indicate if the axis of reduction should remain in the
  #   result
  #
  # ## Examples
  #
  # ```
  # a = [[3, 4], [1, 2], [6, 2]]
  # Num.ptp(a, 1) # [1, 1, 4]
  # ```
  @[Inline]
  def ptp(a : Tensor(U, ARROW(U)), axis : Int, dims : Bool = false) forall U
    Num.subtract(
      Num.max(a, axis, dims),
      Num.min(a, axis, dims)
    )
  end
end
