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

require "./operators"

module Num
  extend self

  # Reduces a `Tensor` to a scalar by summing all of its
  # elements
  #
  # Arguments
  # ---------
  # *a* : Tensor | Enumerable
  #   Argument to sum
  #
  # Examples
  # --------
  # ```
  # a = [1, 2, 3]
  # Num.sum(a) # => 6
  # ```
  def sum(a : Tensor | Enumerable)
    a_t = a.to_tensor
    a_t.iter.reduce(a_t.dtype.new(0)) do |i, j|
      i + j.value
    end
  end

  # Reduces a `Tensor` along an axis, summing each view into
  # the `Tensor`
  #
  # Arguments
  # ---------
  # *a* : Tensor | Enumerable
  #   Argument to sum
  # *axis* : Int
  #   Axis of summation
  # *dims* : Bool
  #   Indicate if the axis of reduction should remain in the result
  #
  # Examples
  # --------
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # Num.sum(a, 0) # => [2, 4]
  # Num.sum(a, 1, dims: true)
  # # [[1],
  # #  [5]]
  # ```
  def sum(a : Tensor | Enumerable, axis : Int, dims : Bool = false)
    a_t = a.to_tensor
    a_t.reduce_axis(axis, dims) do |i, j|
      i + j
    end
  end

  # Accumulate a `Tensor` as though it is flat.  Returning a one dimensional
  # result.
  #
  # Arguments
  # ---------
  # *a* : Tensor | Enumerable
  #   Tensor to accumulate.  Will be treated as one-dimensional
  #
  # Examples
  # --------
  # ```
  # a = [1, 2, 3]
  # Num.cumsum(a) # => [1, 3, 6]
  # ```
  def cumsum(a : Tensor | Enumerable)
    a.to_tensor.accumulate do |i, j|
      i + j
    end
  end

  # Accumulate's a `Tensor` along an axis, summing each view into
  # the `Tensor`
  #
  # Arguments
  # ---------
  # *a* : Tensor | Enumerable
  #   Argument to sum
  # *axis* : Int
  #   Axis of summation
  #
  # Examples
  # --------
  # ```
  # a = Tensor.new([3, 3, 2]) { |i| i }
  # Num.cumsum(a, 0)
  #
  # # [[[ 0,  1],
  # #   [ 2,  3],
  # #   [ 4,  5]],
  # #
  # #  [[ 6,  8],
  # #   [10, 12],
  # #   [14, 16]],
  # #
  # #  [[18, 21],
  # #   [24, 27],
  # #   [30, 33]]]
  # ```
  def cumsum(t : Tensor | Enumerable, axis : Int)
    a_t = t.to_tensor
    a_t.accumulate_axis(axis) do |i, j|
      i + j
    end
  end

  # Reduces a `Tensor` to a scalar by multiplying all of its
  # elements
  #
  # Arguments
  # ---------
  # *a* : Tensor | Enumerable
  #   Argument to reduce
  #
  # Examples
  # --------
  # ```
  # a = [1, 2, 3]
  # Num.prod(a) # => 6
  # ```
  def prod(a : Tensor | Enumerable)
    a_t = a.to_tensor
    a_t.iter.reduce(1) do |i, j|
      i * j.value
    end
  end

  # Reduces a `Tensor` along an axis, multiplying each view into
  # the `Tensor`
  #
  # Arguments
  # ---------
  # *a* : Tensor | Enumerable
  #   Argument to reduce
  # *axis* : Int
  #   Axis of reduction
  # *dims* : Bool
  #   Indicate if the axis of reduction should remain in the result
  #
  # Examples
  # --------
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # Num.prod(a, 0) # => [0, 3]
  # Num.prod(a, 1, dims: true)
  # # [[0],
  # #  [6]]
  # ```
  def prod(a : Tensor | Enumerable, axis : Int, dims : Bool = false)
    a_t = a.to_tensor
    a_t.reduce_axis(axis, dims) do |i, j|
      i * j
    end
  end

  # Reduces a `Tensor` to a boolean by asserting the truthiness of
  # all elements
  #
  # Arguments
  # ---------
  # *a* : Tensor | Enumerable
  #   Argument to reduce
  #
  # Examples
  # --------
  # ```
  # a = [0, 2, 3]
  # Num.all(a) # => false
  # ```
  def all(a : Tensor | Enumerable)
    a_t = a.to_tensor.as_type(Bool)
    a_t.iter.reduce(true) do |i, j|
      i & j.value
    end
  end

  # Reduces a `Tensor` along an axis, asserting the truthiness of all values
  # in each view into the `Tensor`
  #
  # Arguments
  # ---------
  # *a* : Tensor | Enumerable
  #   Argument to reduce
  # *axis* : Int
  #   Axis of reduction
  # *dims* : Bool
  #   Indicate if the axis of reduction should remain in the result
  #
  # Examples
  # --------
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # Num.all(a, 0) # => [false, true]
  # Num.all(a, 1, dims: true)
  # # [[false],
  # #  [ true]]
  # ```
  def all(a : Tensor | Enumerable, axis : Int, dims : Bool = false)
    a_t = a.to_tensor.as_type(Bool)
    a_t.reduce_axis(axis, dims) do |i, j|
      i & j
    end
  end

  # Reduces a `Tensor` to a boolean by asserting the truthiness of
  # any element
  #
  # Arguments
  # ---------
  # *a* : Tensor | Enumerable
  #   Argument to reduce
  #
  # Examples
  # --------
  # ```
  # a = [0, 2, 3]
  # Num.any(a) # => true
  # ```
  def any(a : Tensor | Enumerable)
    a_t = a.to_tensor
    a_t.iter.reduce(false) do |i, j|
      i | (j == 0)
    end
  end

  # Reduces a `Tensor` along an axis, asserting the truthiness of any values
  # in each view into the `Tensor`
  #
  # Arguments
  # ---------
  # *a* : Tensor | Enumerable
  #   Argument to reduce
  # *axis* : Int
  #   Axis of reduction
  # *dims* : Bool
  #   Indicate if the axis of reduction should remain in the result
  #
  # Examples
  # --------
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # Num.any(a, 0) # => [true, true]
  # Num.any(a, 1, dims: true)
  # # [[true],
  # #  [ true]]
  # ```
  def any(a : Tensor | Enumerable, axis : Int, dims : Bool = false)
    a_t = a.to_tensor.as_type(Bool)
    a_t.reduce_axis(axis, dims) do |i, j|
      i | j
    end
  end

  # Reduces a `Tensor` to a scalar by finding the average
  #
  # Arguments
  # ---------
  # *a* : Tensor | Enumerable
  #   Argument to reduce
  #
  # Examples
  # --------
  # ```
  # a = [1, 2, 3]
  # Num.mean(a) # => 2.0
  # ```
  def mean(a : Tensor | Enumerable)
    a_t = a.to_tensor
    Num.sum(a_t) / a_t.size
  end

  # Reduces a `Tensor` along an axis, finding the average of each
  # view into the `Tensor`
  #
  # Arguments
  # ---------
  # *a* : Tensor | Enumerable
  #   Argument to reduce
  # *axis* : Int
  #   Axis of reduction
  # *dims* : Bool
  #   Indicate if the axis of reduction should remain in the result
  #
  # Examples
  # --------
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # Num.mean(a, 0) # => [1, 2]
  # Num.mean(a, 1, dims: true)
  # # [[0],
  # #  [2]]
  # ```
  def mean(a : Tensor | Enumerable, axis : Int, dims : Bool = false)
    a_t = a.to_tensor
    n = a_t.shape[axis]

    a_t.reduce_axis(axis, dims) do |i, j|
      (i + j) / n
    end
  end

  # Reduces a `Tensor` to a scalar by finding the standard deviation
  #
  # Arguments
  # ---------
  # *a* : Tensor | Enumerable
  #   Argument to reduce
  #
  # Examples
  # --------
  # ```
  # a = [1, 2, 3]
  # Num.std(a) # => 0.816496580927726
  # ```
  def std(a : Tensor | Enumerable)
    a_t = a.to_tensor
    c = Num.mean(a_t)
    v = a_t.iter.reduce(0) do |i, j|
      i + (j.value - c)**2
    end
    Math.sqrt(v / a_t.size)
  end

  # Reduces a `Tensor` along an axis, finding the std of each
  # view into the `Tensor`
  #
  # Arguments
  # ---------
  # *a* : Tensor | Enumerable
  #   Argument to reduce
  # *axis* : Int
  #   Axis of reduction
  # *dims* : Bool
  #   Indicate if the axis of reduction should remain in the result
  #
  # Examples
  # --------
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # Num.std(a, 0) # => [1, 1]
  # Num.std(a, 1, dims: true)
  # # [[0.707107],
  # #  [0.707107]]
  # ```
  def std(a : Tensor | Enumerable, axis : Int, dims : Bool = false)
    a_t = a.to_tensor
    u = a_t.shape.dup
    if dims
      u[axis] = 1
    else
      u.delete_at(axis)
    end
    w = Tensor(Float64).new(u)
    v = w.unsafe_iter
    a_t.yield_along_axis(axis) do |a|
      v.next.value = Num.std(a)
    end
    w
  end

  # Reduces a `Tensor` to a scalar by finding the maximum value
  #
  # Arguments
  # ---------
  # *a* : Tensor | Enumerable
  #   Argument to reduce
  #
  # Examples
  # --------
  # ```
  # a = [1, 2, 3]
  # Num.max(a) # => 3
  # ```
  def max(a : Tensor | Enumerable)
    a_t = a.to_tensor
    m = a_t.value
    a_t.each_with_index do |el, i|
      if el > m
        m = el
      end
    end
    m
  end

  # Reduces a `Tensor` along an axis, finding the max of each
  # view into the `Tensor`
  #
  # Arguments
  # ---------
  # *a* : Tensor | Enumerable
  #   Argument to reduce
  # *axis* : Int
  #   Axis of reduction
  # *dims* : Bool
  #   Indicate if the axis of reduction should remain in the result
  #
  # Examples
  # --------
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # Num.max(a, 0) # => [2, 3]
  # Num.max(a, 1, dims: true)
  # # [[1],
  # #  [3]]
  # ```
  def max(a : Tensor | Enumerable, axis : Int, dims : Bool = false)
    a_t = a.to_tensor
    a_t.reduce_axis(axis, dims) do |i, j|
      Math.max(i, j)
    end
  end

  # Reduces a `Tensor` to a scalar by finding the minimum value
  #
  # Arguments
  # ---------
  # *a* : Tensor | Enumerable
  #   Argument to reduce
  #
  # Examples
  # --------
  # ```
  # a = [1, 2, 3]
  # Num.min(a) # => 3
  # ```
  def min(a : Tensor | Enumerable)
    a_t = a.to_tensor
    m = a_t.value
    a_t.each_with_index do |el, i|
      if el < m
        m = el
      end
    end
    m
  end

  # Reduces a `Tensor` along an axis, finding the min of each
  # view into the `Tensor`
  #
  # Arguments
  # ---------
  # *a* : Tensor | Enumerable
  #   Argument to reduce
  # *axis* : Int
  #   Axis of reduction
  # *dims* : Bool
  #   Indicate if the axis of reduction should remain in the result
  #
  # Examples
  # --------
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # Num.min(a, 0) # => [0, 1]
  # Num.min(a, 1, dims: true)
  # # [[0],
  # #  [2]]
  # ```
  def min(a : Tensor | Enumerable, axis : Int, dims : Bool = false)
    a_t = a.to_tensor
    a_t.reduce_axis(axis, dims) do |i, j|
      Math.min(i, j)
    end
  end

  # Sorts a `Tensor`, treating it's elements like the `Tensor`
  # is flat.
  #
  # Arguments
  # ---------
  # *a* : Tensor | Enumerable
  #   Argument to sort
  # *axis* : Int
  #
  # Examples
  # --------
  # ```
  # a = [3, 2, 1].to_tensor
  # Num.sort(a) # => [1, 2, 3]
  # ```
  def sort(a : Tensor | Enumerable)
    a_t = a.to_tensor
    ret = a_t.dup(Num::RowMajor)
    Slice.new(ret.to_unsafe, ret.size).sort!
    ret
  end

  # Sorts a `Tensor` along an axis.
  #
  # Arguments
  # ---------
  # *a* : Tensor | Enumerable
  #   Argument to sort
  # *axis* : Int
  #   Axis to sort along
  #
  # Examples
  # --------
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
  def sort(a : Tensor | Enumerable, axis : Int)
    a_t = a.to_tensor
    ret = a_t.dup(Num::RowMajor)
    ret.yield_along_axis(axis) do |view|
      view[...] = Num.sort(view)
    end
    ret
  end

  # Asserts that two `Tensor`s are equal, allowing for small
  # margins of errors with floating point values using
  # an EPSILON value.
  #
  # Arguments
  # ---------
  # *a* : Tensor | Enumerable
  #   First `Tensor` to compare
  # *b* : Tensor | Enumerable
  #   Second `Tensor` to compare
  # *epsilon* : Number
  #   Allowed variance between numbers
  #
  # Examples
  # --------
  # ```
  # a = [0.0, 0.0, 0.0000000001].to_tensor
  # b = [0.0, 0.0, 0.0].to_tensor
  # Num.all_close(a, b) # => true
  # ```
  def all_close(
    a : Tensor | Enumerable,
    b : Tensor | Enumerable,
    epsilon = 1e-6
  )
    unless a.size == b.size
      return false
    end
    a_t = a.to_tensor
    b_t = b.to_tensor
    a_t.map(b_t) do |i, j|
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
  # Arguments
  # ---------
  # *a* : Tensor to find peak to peak value
  #
  # Examples
  # --------
  # ```
  # a = [1, 2, 3]
  # Num.ptp(a) # => 2
  # ```
  def ptp(a : Tensor | Enumerable)
    a_t = a.to_tensor
    min = a_t.value
    max = a_t.value

    a_t.each do |e|
      if e < min
        min = e
      end
      if e > max
        max = e
      end
    end
    max - min
  end

  # Finds the difference between the maximum and minimum
  # elements of a `Tensor` along an axis
  #
  # Arguments
  # ---------
  # *a* : Tensor
  #   Argument to reduce
  # *axis* : Tensor
  #   Axis of reduction
  # *dims* : Bool
  #   Keep axis of reduction in output
  #
  #
  # Examples
  # --------
  # ```
  # a = [[3, 4], [1, 2], [6, 2]]
  # Num.ptp(a, 1) # [1, 1, 4]
  # ```
  def ptp(a : Tensor | Enumerable, axis : Int, dims : Bool = false)
    Num.subtract(
      Num.max(a, axis, dims),
      Num.min(a, axis, dims)
    )
  end

  # Returns a hash containing the count of each
  # unique element of a `Tensor`
  #
  # Arguments
  # ---------
  # *a* : Tensor
  #   Tensor to count
  # *axis* : Tens
  #
  #
  # Examples
  # --------
  # ```
  # a = [[3, 4], [2, 2]]
  # Num.value_counts(a) # => {3 => 1, 4 => 1, 2 => 2}
  # ```
  def value_counts(a : Tensor(U)) forall U
    counts = Hash(U, Int32).new
    a.each do |e|
      if counts.has_key?(e)
        counts[e] += 1
      else
        counts[e] = 1
      end
    end
    counts
  end
end

class Tensor(T)
  # Reduces a `Tensor` to a scalar by summing all of its
  # elements
  #
  # Arguments
  # ---------
  # *a* : Tensor | Enumerable
  #   Argument to sum
  #
  # Examples
  # --------
  # ```
  # a = [1, 2, 3]
  # Num.sum(a) # => 6
  # ```
  def sum
    Num.sum(self)
  end

  # Reduces a `Tensor` along an axis, summing each view into
  # the `Tensor`
  #
  # Arguments
  # ---------
  # *a* : Tensor | Enumerable
  #   Argument to sum
  # *axis* : Int
  #   Axis of summation
  # *dims* : Bool
  #   Indicate if the axis of reduction should remain in the result
  #
  # Examples
  # --------
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # Num.sum(a, 0) # => [2, 4]
  # Num.sum(a, 1, dims: true)
  # # [[1],
  # #  [5]]
  # ```
  def sum(axis : Int, dims : Bool = false)
    Num.sum(self, axis, dims)
  end

  # Accumulate a `Tensor` as though it is flat.  Returning a one dimensional
  # result.
  #
  # Arguments
  # ---------
  # *a* : Tensor | Enumerable
  #   Tensor to accumulate.  Will be treated as one-dimensional
  #
  # Examples
  # --------
  # ```
  # a = [1, 2, 3]
  # Num.cumsum(a) # => [1, 3, 6]
  # ```
  def cumsum
    Num.cumsum(self)
  end

  # Accumulate's a `Tensor` along an axis, summing each view into
  # the `Tensor`
  #
  # Arguments
  # ---------
  # *a* : Tensor | Enumerable
  #   Argument to sum
  # *axis* : Int
  #   Axis of summation
  #
  # Examples
  # --------
  # ```
  # a = Tensor.new([3, 3, 2]) { |i| i }
  # Num.cumsum(a, 0)
  #
  # # [[[ 0,  1],
  # #   [ 2,  3],
  # #   [ 4,  5]],
  # #
  # #  [[ 6,  8],
  # #   [10, 12],
  # #   [14, 16]],
  # #
  # #  [[18, 21],
  # #   [24, 27],
  # #   [30, 33]]]
  # ```
  def cumsum(axis : Int)
    Num.cumsum(self, axis)
  end

  # Reduces a `Tensor` to a scalar by multiplying all of its
  # elements
  #
  # Arguments
  # ---------
  # *a* : Tensor | Enumerable
  #   Argument to reduce
  #
  # Examples
  # --------
  # ```
  # a = [1, 2, 3]
  # Num.prod(a) # => 6
  # ```
  def prod
    Num.prod(self)
  end

  # Reduces a `Tensor` along an axis, multiplying each view into
  # the `Tensor`
  #
  # Arguments
  # ---------
  # *a* : Tensor | Enumerable
  #   Argument to reduce
  # *axis* : Int
  #   Axis of reduction
  # *dims* : Bool
  #   Indicate if the axis of reduction should remain in the result
  #
  # Examples
  # --------
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # Num.prod(a, 0) # => [0, 3]
  # Num.prod(a, 1, dims: true)
  # # [[0],
  # #  [6]]
  # ```
  def prod(axis : Int, dims : Bool = false)
    Num.prod(self, axis, dims)
  end

  # Reduces a `Tensor` to a boolean by asserting the truthiness of
  # all elements
  #
  # Arguments
  # ---------
  # *a* : Tensor | Enumerable
  #   Argument to reduce
  #
  # Examples
  # --------
  # ```
  # a = [0, 2, 3]
  # Num.all(a) # => false
  # ```
  def all
    Num.all(self)
  end

  # Reduces a `Tensor` along an axis, asserting the truthiness of all values
  # in each view into the `Tensor`
  #
  # Arguments
  # ---------
  # *a* : Tensor | Enumerable
  #   Argument to reduce
  # *axis* : Int
  #   Axis of reduction
  # *dims* : Bool
  #   Indicate if the axis of reduction should remain in the result
  #
  # Examples
  # --------
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # Num.all(a, 0) # => [false, true]
  # Num.all(a, 1, dims: true)
  # # [[false],
  # #  [ true]]
  # ```
  def all(axis : Int, dims : Bool = false)
    Num.all(self, axis, dims)
  end

  # Reduces a `Tensor` to a boolean by asserting the truthiness of
  # any element
  #
  # Arguments
  # ---------
  # *a* : Tensor | Enumerable
  #   Argument to reduce
  #
  # Examples
  # --------
  # ```
  # a = [0, 2, 3]
  # Num.any(a) # => true
  # ```
  def any
    Num.any(self)
  end

  # Reduces a `Tensor` along an axis, asserting the truthiness of any values
  # in each view into the `Tensor`
  #
  # Arguments
  # ---------
  # *a* : Tensor | Enumerable
  #   Argument to reduce
  # *axis* : Int
  #   Axis of reduction
  # *dims* : Bool
  #   Indicate if the axis of reduction should remain in the result
  #
  # Examples
  # --------
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # Num.any(a, 0) # => [true, true]
  # Num.any(a, 1, dims: true)
  # # [[true],
  # #  [ true]]
  # ```
  def any(axis : Int, dims : Bool = false)
    Num.any(self, axis, dims)
  end

  # Reduces a `Tensor` to a scalar by finding the average
  #
  # Arguments
  # ---------
  # *a* : Tensor | Enumerable
  #   Argument to reduce
  #
  # Examples
  # --------
  # ```
  # a = [1, 2, 3]
  # Num.mean(a) # => 2.0
  # ```
  def mean
    Num.mean(self)
  end

  # Reduces a `Tensor` along an axis, finding the average of each
  # view into the `Tensor`
  #
  # Arguments
  # ---------
  # *a* : Tensor | Enumerable
  #   Argument to reduce
  # *axis* : Int
  #   Axis of reduction
  # *dims* : Bool
  #   Indicate if the axis of reduction should remain in the result
  #
  # Examples
  # --------
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # Num.mean(a, 0) # => [1, 2]
  # Num.mean(a, 1, dims: true)
  # # [[0],
  # #  [2]]
  # ```
  def mean(axis : Int, dims : Bool = false)
    Num.mean(self, axis, dims)
  end

  # Reduces a `Tensor` to a scalar by finding the standard deviation
  #
  # Arguments
  # ---------
  # *a* : Tensor | Enumerable
  #   Argument to reduce
  #
  # Examples
  # --------
  # ```
  # a = [1, 2, 3]
  # Num.std(a) # => 0.816496580927726
  # ```
  def std
    Num.std(self)
  end

  # Reduces a `Tensor` along an axis, finding the std of each
  # view into the `Tensor`
  #
  # Arguments
  # ---------
  # *a* : Tensor | Enumerable
  #   Argument to reduce
  # *axis* : Int
  #   Axis of reduction
  # *dims* : Bool
  #   Indicate if the axis of reduction should remain in the result
  #
  # Examples
  # --------
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # Num.std(a, 0) # => [1, 1]
  # Num.std(a, 1, dims: true)
  # # [[0.707107],
  # #  [0.707107]]
  # ```
  def std(axis : Int, dims : Bool = false)
    Num.std(self, axis, dims)
  end

  # Reduces a `Tensor` to a scalar by finding the maximum value
  #
  # Arguments
  # ---------
  # *a* : Tensor | Enumerable
  #   Argument to reduce
  #
  # Examples
  # --------
  # ```
  # a = [1, 2, 3]
  # Num.max(a) # => 3
  # ```
  def max
    Num.max(self)
  end

  # Reduces a `Tensor` along an axis, finding the max of each
  # view into the `Tensor`
  #
  # Arguments
  # ---------
  # *a* : Tensor | Enumerable
  #   Argument to reduce
  # *axis* : Int
  #   Axis of reduction
  # *dims* : Bool
  #   Indicate if the axis of reduction should remain in the result
  #
  # Examples
  # --------
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # Num.max(a, 0) # => [2, 3]
  # Num.max(a, 1, dims: true)
  # # [[1],
  # #  [3]]
  # ```
  def max(axis : Int, dims : Bool = false)
    Num.max(self, axis, dims)
  end

  # Reduces a `Tensor` to a scalar by finding the minimum value
  #
  # Arguments
  # ---------
  # *a* : Tensor | Enumerable
  #   Argument to reduce
  #
  # Examples
  # --------
  # ```
  # a = [1, 2, 3]
  # Num.min(a) # => 3
  # ```
  def min
    Num.min(self)
  end

  # Reduces a `Tensor` along an axis, finding the min of each
  # view into the `Tensor`
  #
  # Arguments
  # ---------
  # *a* : Tensor | Enumerable
  #   Argument to reduce
  # *axis* : Int
  #   Axis of reduction
  # *dims* : Bool
  #   Indicate if the axis of reduction should remain in the result
  #
  # Examples
  # --------
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # Num.min(a, 0) # => [0, 1]
  # Num.min(a, 1, dims: true)
  # # [[0],
  # #  [2]]
  # ```
  def min(axis : Int, dims : Bool = false)
    Num.min(axis, dims)
  end

  # Finds the difference between the maximum and minimum
  # elements of a `Tensor`
  #
  # Arguments
  # ---------
  # *a* : Tensor to find peak to peak value
  #
  # Examples
  # --------
  # ```
  # a = [1, 2, 3]
  # Num.ptp(a) # => 2
  # ```
  def ptp
    Num.ptp(self)
  end

  # Finds the difference between the maximum and minimum
  # elements of a `Tensor` along an axis
  #
  # Arguments
  # ---------
  # *a* : Tensor
  #   Argument to reduce
  # *axis* : Tensor
  #   Axis of reduction
  # *dims* : Bool
  #   Keep axis of reduction in output
  #
  #
  # Examples
  # --------
  # ```
  # a = [[3, 4], [1, 2], [6, 2]]
  # Num.ptp(a, 1) # [1, 1, 4]
  # ```
  def ptp(axis : Int, dims : Bool = false)
    Num.ptp(self, axis, dims)
  end

  # Sorts a `Tensor`, treating it's elements like the `Tensor`
  # is flat.
  #
  # Arguments
  # ---------
  # *a* : Tensor | Enumerable
  #   Argument to sort
  # *axis* : Int
  #
  # Examples
  # --------
  # ```
  # a = [3, 2, 1].to_tensor
  # Num.sort(a) # => [1, 2, 3]
  # ```
  def sort
    Num.sort(self)
  end

  # Sorts a `Tensor` along an axis.
  #
  # Arguments
  # ---------
  # *a* : Tensor | Enumerable
  #   Argument to sort
  # *axis* : Int
  #   Axis to sort along
  #
  # Examples
  # --------
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
  def sort(axis : Int)
    Num.sort(self, axis)
  end

  # Returns a hash containing the count of each
  # unique element of a `Tensor`
  #
  # Arguments
  # ---------
  # *a* : Tensor
  #   Tensor to count
  # *axis* : Tens
  #
  #
  # Examples
  # --------
  # ```
  # a = [[3, 4], [2, 2]]
  # Num.value_counts(a) # => {3 => 1, 4 => 1, 2 => 2}
  # ```
  def value_counts
    Num.value_counts(self)
  end
end
