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
  # ```crystal
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
  # ```crystal
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
  # ```crystal
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
  # ```crystal
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
  # ```crystal
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
  # ```crystal
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
  # *a* : Tensor
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

  # Reduces a `Tensor` to a scalar by finding the maximum value
  #
  # Arguments
  # ---------
  # *a* : Tensor
  #   Argument to reduce
  #
  # Examples
  # --------
  # ```
  # a = [1, 2, 3]
  # Num.max(a) # => 3
  # ```
  def max : T
    Num.max(self)
  end

  # Reduces a `Tensor` along an axis, finding the max of each
  # view into the `Tensor`
  #
  # Arguments
  # ---------
  # *a* : Tensor
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
  # *a* : Tensor
  #   Argument to reduce
  #
  # Examples
  # --------
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
  # Arguments
  # ---------
  # *a* : Tensor
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
    Num.min(self, axis, dims)
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
  def std : Float64
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

  # Find the maximum index value of a Tensor
  #
  # Arguments
  # ---------
  # a : Tensor | Enumerable
  #   Input tensor
  #
  # Returns
  # -------
  # Index of the maximum value
  #
  # Examples
  # --------
  def argmax : Int32
    Num.argmax(self)
  end

  # Find the maximum index value of a Tensor along
  # an axis
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
  # Returns
  # -------
  # Tensor(Int32, CPU(Int32))
  #
  # Examples
  # --------
  def argmax(axis : Int, dims : Bool = false)
    Num.argmax(self, axis, dims)
  end

  # Find the minimum index value of a Tensor
  #
  # Arguments
  # ---------
  # a : Tensor | Enumerable
  #   Input tensor
  #
  # Returns
  # -------
  # Index of the minimum value
  #
  # Examples
  # --------
  def argmin : Int32
    Num.argmin(self)
  end

  # Find the minimum index value of a Tensor along
  # an axis
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
  # Returns
  # -------
  # Tensor(Int32)
  #
  # Examples
  # --------
  def argmin(axis : Int, dims : Bool = false)
    Num.argmin(self, axis, dims)
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
  def sort : Tensor(T, S)
    Num.sort(self)
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
  def sort(&block : T, T -> _)
    Num.sort(self, &block)
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
  def sort(axis : Int) : Tensor(T, S)
    Num.sort(self, axis)
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
  def sort(axis : Int, &block : T, T -> _)
    Num.sort(self, axis, &block)
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
  def all_close(other : Tensor, epsilon = 1e-6) : Bool
    Num.all_close(self, other, epsilon)
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
  def ptp : T
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
    Num.self(self, axis, dims)
  end
end
