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

class Tensor(T, S)
  # Reduces a `Tensor` to a scalar by summing all of its
  # elements
  #
  # ## Examples
  #
  # ```
  # a = [1, 2, 3]
  # a.sum # => 6
  # ```
  def sum : T
    Num.sum(self)
  end

  # Reduces a `Tensor` along an axis, summing each view into
  # the `Tensor`
  #
  # ## Arguments
  #
  # * axis : `Int` - Axis of summation
  # * dims : `Bool` - Indicate if the axis of reduction should remain in the
  #   result
  #
  # ## Examples
  #
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # a.sum(0) # => [2, 4]
  # a.sum(1, dims: true)
  # # [[1],
  # #  [5]]
  # ```
  def sum(axis : Int, dims : Bool = false)
    Num.sum(self, axis, dims)
  end

  # Reduces a `Tensor` to a scalar by multiplying all of its
  # elements
  #
  # ## Examples
  #
  # ```
  # a = [1, 2, 3]
  # a.prod # => 6
  # ```
  def prod : T
    Num.prod(self)
  end

  # Reduces a `Tensor` along an axis, multiplying each view into
  # the `Tensor`
  #
  # ## Arguments
  #
  # * axis : `Int` - Axis of reduction
  # * dims : `Bool` - Indicate if the axis of reduction should remain in the
  #   result
  #
  # ## Examples
  #
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # a.prod(0) # => [0, 3]
  # a.prod(1, dims: true)
  # # [[0],
  # #  [6]]
  # ```
  def prod(axis : Int, dims : Bool = false)
    Num.prod(self, axis, dims)
  end

  # Reduces a `Tensor` to a boolean by asserting the truthiness of
  # all elements
  #
  # ## Examples
  #
  # ```
  # a = [0, 2, 3].to_tensor
  # a.all # => false
  # ```
  def all : Bool
    Num.all(self)
  end

  # Reduces a `Tensor` along an axis, asserting the truthiness of all values
  # in each view into the `Tensor`
  #
  # ## Arguments
  #
  # * axis : `Int` - Axis of reduction
  # * dims : `Bool` - Indicate if the axis of reduction should remain in the
  #   result
  #
  # ## Examples
  #
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # a.all(0) # => [false, true]
  # a.all(1, dims: true)
  # # [[false],
  # #  [ true]]
  # ```
  def all(axis : Int, dims : Bool = false)
    Num.all(self, axis, dims)
  end

  # Reduces a `Tensor` to a boolean by asserting the truthiness of
  # any element
  #
  # ## Examples
  #
  # ```
  # a = [0, 2, 3].to_tensor
  # a.any # => true
  # ```
  def any : Bool
    Num.any(self)
  end

  # Reduces a `Tensor` along an axis, asserting the truthiness of any values
  # in each view into the `Tensor`
  #
  # ## Arguments
  #
  # * axis : `Int` - Axis of reduction
  # * dims : `Bool` - Indicate if the axis of reduction should remain in the
  #   result
  #
  # ## Examples
  #
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # a.any(0) # => [true, true]
  # a.any(1, dims: true)
  # # [[true],
  # #  [ true]]
  # ```
  def any(axis : Int, dims : Bool = false)
    Num.any(self, axis, dims)
  end

  # Reduces a `Tensor` to a scalar by finding the average
  #
  # ## Examples
  #
  # ```
  # a = [1, 2, 3]
  # a.mean # => 2.0
  # ```
  def mean
    Num.mean(self)
  end

  # Reduces a `Tensor` along an axis, finding the average of each
  # view into the `Tensor`
  #
  # ## Arguments
  #
  # * axis : `Int` - Axis of reduction
  # * dims : `Bool` - Indicate if the axis of reduction should remain in the
  #   result
  #
  # ## Examples
  #
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # a.mean(0) # => [1, 2]
  # a.mean(1, dims: true)
  # # [[0],
  # #  [2]]
  # ```
  def mean(axis : Int, dims : Bool = false)
    Num.mean(self, axis, dims)
  end

  # Reduces a `Tensor` to a scalar by finding the maximum value
  #
  # ## Examples
  #
  # ```
  # a = [1, 2, 3]
  # a.max # => 3
  # ```
  def max : T
    Num.max(self)
  end

  # Reduces a `Tensor` along an axis, finding the max of each
  # view into the `Tensor`
  #
  # ## Arguments
  #
  # * axis : `Int` - Axis of reduction
  # * dims : `Bool` - Indicate if the axis of reduction should remain in the
  #   result
  #
  # ## Examples
  #
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # a.max(0) # => [2, 3]
  # a.max(1, dims: true)
  # # [[1],
  # #  [3]]
  # ```
  def max(axis : Int, dims : Bool = false)
    Num.max(self, axis, dims)
  end

  # Reduces a `Tensor` to a scalar by finding the minimum value
  #
  # ## Examples
  #
  # ```
  # a = [1, 2, 3]
  # Num.min(a) # => 3
  # ```
  def min : T
    Num.min(self, axis, dims)
  end

  # Reduces a `Tensor` along an axis, finding the min of each
  # view into the `Tensor`
  #
  # ## Arguments
  #
  # * axis : `Int` - Axis of reduction
  # * dims : `Bool` - Indicate if the axis of reduction should remain in the
  #   result
  #
  # ## Examples
  #
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # a.min(0) # => [0, 1]
  # a.min(1, dims: true)
  # # [[0],
  # #  [2]]
  # ```
  def min(axis : Int, dims : Bool = false)
    Num.min(self, axis, dims)
  end

  # Reduces a `Tensor` to a scalar by finding the standard deviation
  #
  # ## Examples
  #
  # ```
  # a = [1, 2, 3].to_tensor
  # a.std # => 0.816496580927726
  # ```
  def std : Float64
    Num.std(self)
  end

  # Reduces a `Tensor` along an axis, finding the std of each
  # view into the `Tensor`
  #
  # ## Arguments
  #
  # * axis : `Int` - Axis of reduction
  # * dims : `Bool` - Indicate if the axis of reduction should remain in the
  #   result
  #
  # ## Examples
  #
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # a.std(0) # => [1, 1]
  # a.std(1, dims: true)
  # # [[0.707107],
  # #  [0.707107]]
  # ```
  def std(axis : Int, dims : Bool = false)
    Num.std(self, axis, dims)
  end

  # Find the maximum index value of a Tensor
  #
  # ## Examples
  # ```
  # a = [1, 2, 3].to_tensor
  # a.argmax # => 2
  # ```
  def argmax : Int32
    Num.argmax(self)
  end

  # Find the maximum index value of a Tensor along
  # an axis
  #
  # ## Arguments
  #
  # * axis : `Int` - Axis of reduction
  # * dims : `Bool` - Indicate if the axis of reduction should remain in the
  #   result
  #
  # ## Examples
  #
  # ```
  # a = [[4, 2], [0, 1]].to_tensor
  # a.argmax(1) # => [0, 1]
  # ```
  def argmax(axis : Int, dims : Bool = false)
    Num.argmax(self, axis, dims)
  end

  # Find the minimum index value of a Tensor
  #
  # ## Examples
  # ```
  # a = [1, 2, 3].to_tensor
  # a.argmin # => 0
  # ```
  def argmin : Int32
    Num.argmin(self)
  end

  # Find the minimum index value of a Tensor along
  # an axis
  #
  # ## Arguments
  #
  # * axis : `Int` - Axis of reduction
  # * dims : `Bool` - Indicate if the axis of reduction should remain in the
  #   result
  #
  # ## Examples
  #
  # ```
  # a = [[4, 2], [0, 1]].to_tensor
  # a.argmin(1) # => [1, 0]
  # ```
  def argmin(axis : Int, dims : Bool = false)
    Num.argmin(self, axis, dims)
  end

  # Sorts a `Tensor`, treating it's elements like the `Tensor`
  # is flat.
  #
  # ## Examples
  #
  # ```
  # a = [3, 2, 1].to_tensor
  # a.sort # => [1, 2, 3]
  # ```
  def sort : Tensor(T, S)
    Num.sort(self)
  end

  # Sorts a `Tensor`, treating it's elements like the `Tensor`
  # is flat.  Sorts using criteria specified by a passed block
  #
  # ## Arguments
  #
  # * block : `Proc(T, T, _)` - Function used to sort
  #
  # ## Examples
  #
  # ```
  # a = [3, 2, 1].to_tensor
  # a.sort { |i, j| j - i } # => [3, 2, 1]
  # ```
  def sort(&block : T, T -> _)
    Num.sort(self, &block)
  end

  # Sorts a `Tensor` along an axis.
  #
  # ## Arguments
  #
  # * axis : `Int` - Axis of reduction
  #
  # ## Examples
  #
  # ```
  # t = Tensor.random(0...10, [3, 3, 2])
  # puts t.sort(axis: 1)
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
  def sort(axis : Int) : Tensor(T, S)
    Num.sort(self, axis)
  end

  # Sorts a `Tensor` along an axis.
  #
  # ## Arguments
  #
  # * axis : `Int` - Axis of reduction
  #
  # ## Examples
  #
  # ```
  # t = Tensor.random(0...10, [3, 3, 2])
  # puts t.sort(axis: 1) { |i, j| i <=> j }
  #
  # # [[[5, 3],
  # #   [6, 9],
  # #   [7, 9]],
  # #
  # #  [[0, 1],
  # #   [3, 2],
  # #   [8, 5]],
  # #
  # #  [[3, 1],
  # #   [4, 7],
  # #   [7, 8]]]
  # ```
  def sort(axis : Int, &block : T, T -> _)
    Num.sort(self, axis, &block)
  end

  # Asserts that two `Tensor`s are equal, allowing for small
  # margins of errors with floating point values using
  # an EPSILON value.
  #
  # ## Arguments
  #
  # * other : `Tensor` - `Tensor` to compare to `self`
  # * epsilon : `Float` - Margin of error to accept between elements
  #
  # ## Examples
  #
  # ```
  # a = [0.0, 0.0, 0.0000000001].to_tensor
  # b = [0.0, 0.0, 0.0].to_tensor
  # a.all_close(b)        # => true
  # a.all_close(b, 1e-12) # => false
  # ```
  def all_close(other : Tensor, epsilon : Float = 1e-6) : Bool
    Num.all_close(self, other, epsilon)
  end

  # Finds the difference between the maximum and minimum
  # elements of a `Tensor`
  #
  # ## Examples
  #
  # ```
  # a = [1, 2, 3].to_tensor
  # a.ptp # => 2
  # ```
  def ptp : T
    Num.ptp(self)
  end

  # Finds the difference between the maximum and minimum
  # elements of a `Tensor` along an axis
  #
  # ## Arguments
  #
  # * axis : `Int` - Axis of reduction
  # * dims : `Bool` - Indicate if the axis of reduction should remain in the
  #   result
  #
  # ## Examples
  #
  # ```
  # a = [[3, 4], [1, 2], [6, 2]].to_tensor
  # a.ptp(1) # [1, 1, 4]
  # ```
  def ptp(axis : Int, dims : Bool = false)
    Num.self(self, axis, dims)
  end
end
