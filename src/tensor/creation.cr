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

require "./tensor"
require "./internal/random"

class Tensor(T)
  # Creates a `Tensor` of a provided shape, filled with 0.  The generic type
  # must be specified.
  #
  # Arguments
  # ---------
  # *shape*
  #   Shape of returned `Tensor`
  #
  # Examples
  # --------
  # ```
  # t = Tensor(Int8).zeros([3]) # => [0, 0, 0]
  # ```
  def self.zeros(shape : Array(Int)) : Tensor(T)
    self.new(shape, T.new(0))
  end

  # Creates a `Tensor` filled with 0, sharing the shape of another
  # provided `Tensor`
  #
  # Arguments
  # ---------
  # *t*
  #   `Tensor` to use for output shape
  #
  # Examples
  # --------
  # ```
  # t = Tensor.new([3]) &.to_f
  # u = Tensor(Int8).zeros_like(t) # => [0, 0, 0]
  # ```
  def self.zeros_like(t : Tensor) : Tensor(T)
    self.new(t.shape, T.new(0))
  end

  # Creates a `Tensor` of a provided shape, filled with 1.  The generic type
  # must be specified.
  #
  # Arguments
  # ---------
  # *shape*
  #   Shape of returned `Tensor`
  #
  # Examples
  # --------
  # ```
  # t = Tensor(Int8).ones([3]) # => [1, 1, 1]
  # ```
  def self.ones(shape : Array(Int)) : Tensor(T)
    self.new(shape, T.new(1))
  end

  # Creates a `Tensor` filled with 1, sharing the shape of another
  # provided `Tensor`
  #
  # Arguments
  # ---------
  # *t*
  #   `Tensor` to use for output shape
  #
  # Examples
  # --------
  # ```
  # t = Tensor.new([3]) &.to_f
  # u = Tensor(Int8).ones_like(t) # => [0, 0, 0]
  # ```
  def self.ones_like(t : Tensor) : Tensor(T)
    self.new(t.shape, T.new(1))
  end

  # Creates a flat `Tensor` containing a monotonically increasing
  # or decreasing range.  The generic type is inferred from
  # the inputs to the method
  #
  # Arguments
  # ---------
  # *start*
  #   Beginning value for the range
  # *stop*
  #   End value for the range
  # *step*
  #   Offset between values of the range
  #
  # Examples
  # --------
  # ```
  # Tensor.range(0, 5, 2)       # => [0, 2, 4]
  # Tensor.range(5, 0, -1)      # => [5, 4, 3, 2, 1]
  # Tensor.range(0.0, 3.5, 0.7) # => [0  , 0.7, 1.4, 2.1, 2.8]
  # ```
  def self.range(start : T, stop : T, step : T) : Tensor(T)
    if start > stop && step > 0
      raise Num::Internal::ValueError.new(
        "Range must return at least one value"
      )
    end

    r = stop - start
    n = (r / step).ceil.abs
    self.new([n.to_i]) do |i|
      T.new(start + i * step)
    end
  end

  # :ditto:
  def self.range(stop : T) : Tensor(T)
    self.range(T.new(0), stop, T.new(1))
  end

  # :ditto:
  def self.range(start : T, stop : T) : Tensor(T)
    self.range(start, stop, T.new(1))
  end

  # Return evenly spaced numbers over a specified interval.
  # Returns `num` evenly spaced samples, calculated over the
  # interval [`start`, `stop`].
  # The endpoint of the interval can optionally be excluded.
  #
  # ```
  # B.linspace(0, 1, 5) # => Tensor[0.0, 0.25, 0.5, 0.75, 1.0]
  #
  # B.linspace(0, 1, 5, endpoint: false) # => Tensor[0.0, 0.2, 0.4, 0.6, 0.8]
  # ```
  def self.linear_space(start : T, stop : T, num : Int = 50, endpoint = true)
    unless num > 0
      raise Num::Internal::ValueError.new(
        "Number of samples must be non-negative"
      )
    end
    divisor = endpoint ? num - 1 : num
    result = Tensor.range(T.new(num))
    delta = stop - start

    if num > 1
      step = delta / divisor
      if step == 0
        raise Num::Internal::ValueError.new(
          "Step cannot be 0"
        )
      end
    else
      step = delta
    end

    result.map! do |i|
      i * step + start
    end

    if endpoint && num > 1
      result[result.shape[0] - 1] = stop
    end

    result
  end

  # Return numbers spaced evenly on a log scale.
  # In linear space, the sequence starts at ``base ** start``
  # (`base` to the power of `start`) and ends with ``base ** stop``
  # (see `endpoint` below).
  #
  # ```
  # B.logspace(2.0, 3.0, num = 4) # => Tensor[100.0, 215.44346900318845, 464.15888336127773, 1000.0]
  # ```
  def self.logarithmic_space(
    start : T,
    stop : T,
    num = 50,
    endpoint = true,
    base : T = T.new(10.0)
  )
    result = Tensor.linear_space(start, stop, num: num, endpoint: endpoint)
    result.map! do |i|
      base ** i
    end
    result
  end

  # Return numbers spaced evenly on a log scale (a geometric progression).
  # This is similar to `logspace`, but with endpoints specified directly.
  # Each output sample is a constant multiple of the previous.
  #
  # ```
  # geomspace(1, 1000, 4) # => Tensor[1.0, 10.0, 100.0, 1000.0]
  # ```
  def self.geometric_space(start : T, stop : T, num = 50, endpoint = true)
    if start == 0 || stop == 0
      raise Num::Internal::ValueError.new(
        "Geometric sequence cannot include zero"
      )
    end

    out_sign = T.new(1.0)

    if start < 0 && stop < 0
      start, stop = -start, -stop
      out_sign = -out_sign
    end

    log_start = Math.log(start, T.new(10.0))
    log_stop = Math.log(stop, T.new(10.0))

    Tensor.logarithmic_space(
      log_start,
      log_stop,
      num: num,
      endpoint: endpoint,
      base: T.new(10.0)
    ) * out_sign
  end

  # Return a two-dimensional `Tensor` with ones along the diagonal,
  # and zeros elsewhere
  #
  # Arguments
  # ---------
  # *m* : Int
  #   Number of rows in the `Tensor`
  # *n* : Int?
  #   Number of columsn in the `Tensor`, defaults to `m` if nil
  # *offset* : Int
  #   Indicates which diagonal to fill with ones
  #
  # Examples
  # --------
  # ```
  # Tensor(Int32).eye(3, offset: -1)
  #
  # # [[0, 0, 0],
  # #  [1, 0, 0],
  # #  [0, 1, 0]]
  #
  # Tensor(Int8).eye(2)
  #
  # # [[1, 0],
  # #  [0, 1]]
  # ```
  def self.eye(m : Int, n : Int? = nil, offset : Int = 0)
    n = n.nil? ? m : n
    Tensor.new(m, n) do |i, j|
      i == j - offset ? T.new(1) : T.new(0)
    end
  end

  # Returns an identity `Tensor` with ones along the diagonal,
  # and zeros elsewhere
  #
  # Arguments
  # ---------
  # *n* : Number of rows and columns in output
  #
  # Examples
  # --------
  # ```
  # Tensor(Int8).identity(2)
  #
  # # [[1, 0],
  # #  [0, 1]]
  # ```
  def self.identity(n : Int)
    self.new(n, n) do |i, j|
      i == j ? T.new(1) : T.new(0)
    end
  end

  # Creates a two dimensional `Tensor` with a one-dimensional `Tensor`
  # placed along the diagonal
  #
  # Arguments
  # ---------
  # *a* : Tensor | Enumerable
  #   Input `Tensor`, must be one-dimensional
  #
  # Examples
  # --------
  # ```
  # Tensor.diag([1, 2, 3])
  #
  # # [[1, 0, 0],
  # #  [0, 2, 0],
  # #  [0, 0, 3]]
  # ```
  def self.diag(a : Tensor | Enumerable)
    a_t = a.to_tensor
    if a_t.rank > 1
      raise Num::Internal::ShapeError.new("Input must be one-dimensional")
    end
    s0 = a_t.shape[0]
    t = a_t.class.new([s0, s0])
    t.diagonal[...] = a_t
    t
  end

  # Generate a Vandermonde matrix.
  #
  # The columns of the output `Tensor` are powers of the input vector.
  # The order of the powers is determined by the increasing boolean
  # argument. Specifically, when increasing is False, the i-th output
  # column is the input vector raised element-wise to the power of
  # N - i - 1. Such a matrix with a geometric progression in each
  # row is named for Alexandre- Theophile Vandermonde.
  #
  # Arguments
  # ---------
  # *t* : Tensor(T)
  #   One dimensional input `Tensor`
  # *n* : Int
  #   Number of columns in the output, defaults to `t.size`
  # *increasing* : Bool
  #   Specifies the order of the powers in the output
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # Tensor.vandermonde(a)
  #
  # # [[1, 1, 1],
  # #  [4, 2, 1],
  # #  [9, 3, 1]]
  #
  # Tensor.vandermonde(a, 4, increasing: true)
  #
  # # [[ 1,  1,  1,  1],
  # #  [ 1,  2,  4,  8],
  # #  [ 1,  3,  9, 27]]
  # ```
  def self.vandermonde(
    t : Tensor | Enumerable,
    n : Int,
    increasing : Bool = false
  )
    a_t = t.to_tensor
    if a_t.rank > 1
      raise Num::Internal::ShapeError.new("Input must be one-dimensional")
    end

    a_t.class.new(a_t.size, n) do |i, j|
      a_t[i].value ** (increasing ? j : n - j - 1)
    end
  end

  # :ditto:
  def self.vandermonde(t : Tensor(T), increasing : Bool = false)
    vandermonde(t, t.size, increasing)
  end
end
