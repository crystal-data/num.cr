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
  # Create a `Tensor` containing random values between a range.
  # The dtype of the `Tensor` is inferred from the dtypes of the
  # range
  #
  # Example
  # ```
  # t = Tensor.random(1...5, [3])
  # puts t
  # ```
  #
  # Output
  # ```
  # [1, 4, 2]
  # ```
  def self.random(r : Range(T, T), _shape : Array(Int32))
    if _shape.size == 0
      Tensor(T).new(_shape)
    else
      new(_shape) { |_| Random.rand(r) }
    end
  end

  # Creates a `Tensor` of zeros with a specified shape
  #
  # Example
  # ```
  # t = Tensor(UInt8).zeros([5])
  # puts t
  # ```
  #
  # Output
  # ```
  # [0, 0, 0, 0, 0]
  # ```
  def self.zeros(shape : Array(Int32))
    Tensor(T).new(shape, T.new(0))
  end

  # Creates a `Tensor` of zeros with the same shape as
  # a passed `Tensor`
  #
  # Example
  # ```
  # a = Tensor(Float32).zeros([3])
  # b = Tensor(Float32).zeros_like(a)
  # puts a
  # ```
  #
  # Output
  # ```
  # [0, 0, 0]
  # ```
  def self.zeros_like(other : NumInternal::AnyTensor)
    Tensor(T).new(other.shape, T.new(0))
  end

  # Creates a `Tensor` of ones with a specified shape
  #
  # Example
  # ```
  # t = Tensor(UInt8).ones([5])
  # puts t
  # ```
  #
  # Output
  # ```
  # [1, 1, 1, 1, 1]
  # ```
  def self.ones(shape : Array(Int32))
    Tensor(T).new(shape, T.new(1))
  end

  # Creates a `Tensor` of ones with the same shape as
  # a passed `Tensor`
  #
  # Example
  # ```
  # a = Tensor(Float32).ones([3])
  # b = Tensor(Float32).ones_like(a)
  # puts a
  # ```
  #
  # Output
  # ```
  # [1, 1, 1]
  # ```
  def self.ones_like(other : NumInternal::AnyTensor)
    Tensor(T).new(other.shape, T.new(1))
  end

  # Creates a `Tensor` of a specified value with a specified shape
  #
  # Example
  # ```
  # t = Tensor(UInt8).full([5], 3.2)
  # puts t
  # ```
  #
  # Output
  # ```
  # [3.2, 3.2, 3.2, 3.2, 3.2]
  # ```
  def self.full(shape : Array(Int32), value : Number)
    Tensor(T).new(shape, T.new(value))
  end

  # Creates a `Tensor` of a specified value with the same shape as
  # a passed `Tensor`
  #
  # Example
  # ```
  # a = Tensor(Float32).full([3], 10)
  # b = Tensor(Float32).full_like(a, 2)
  # puts a
  # ```
  #
  # Output
  # ```
  # [2, 2, 2]
  # ```
  def self.full_like(other : NumInternal::AnyTensor, value : Number)
    Tensor(T).new(other.shape, T.new(value))
  end

  # Creates a Tensor from a provided start, stop and step.
  # The dtype of the Tensor is inferred from inputs
  #
  # Example
  # ```
  # a = Tensor.range(0, 5, 1)
  # puts a
  # ```
  #
  # Output
  # ```
  # [0, 1, 2, 3, 4]
  # ```
  def self.range(start : T, stop : T, step : T)
    if start > stop && step > 0
      raise NumInternal::ValueError.new("Range must return at at least one value")
    end
    r = (stop - start)
    num = (r / step).ceil.abs
    Tensor.new([Int32.new(num)]) { |i| T.new(start + (i * step)) }
  end

  # Creates a Tensor ranging from 0 to a provided stop value.
  # The dtype of the Tensor is inferred from inputs
  #
  # Example
  # ```
  # a = Tensor.range(3.0)
  # puts a
  # ```
  #
  # Output
  # ```
  # [0, 1, 2]
  # ```
  def self.range(stop : T)
    Tensor.range(T.new(0), stop, T.new(1))
  end

  # Creates a Tensor from a provided start and stop value
  # The dtype of the Tensor is inferred from inputs
  #
  # Example
  # ```
  # a = Tensor.range(2, 6)
  # puts a
  # ```
  #
  # Output
  # ```
  # [2, 3, 4, 5]
  # ```
  def self.range(start : T, stop : T)
    Tensor.range(start, stop, T.new(1))
  end

  # Returns evenly spaced numbers of a specified interval.
  # The endpoint of the interval can be optionally excluded
  #
  # Example
  # ```
  # a = Tensor(Float64).linear_space(2.0, 3.0, num: 5)
  # puts a
  # ```
  #
  # Output
  # ```
  # [2, 2.25, 2.5, 2.75, 3]
  # ```
  def self.linear_space(start : Number, stop : Number, num = 50, endpoint = true)
    raise NumInternal::ValueError.new "Number of samples must be non-negative" unless num > 0
    div = endpoint ? num - 1 : num
    start = start * 1.0
    stop = stop * 1.0
    y = Tensor.range(Float64.new(num))
    delta = stop - start
    if num > 1
      step = delta / div
      if step == 0
        raise NumInternal::ValueError.new "Cannot have a step of 0"
      end
      y = y * step
    else
      y = y * delta
    end
    y += start
    if endpoint && num > 1
      y[y.shape[0] - 1] = stop
    end
    y
  end

  # Returns evenly spaced numbers on a log scale
  # The sequence begins with `base ** start` and ends at `base ** step`
  #
  # Example
  # ```
  # a = Tensor(Float64).logarithmic_space(2.0, 3.0, num: 4)
  # puts a
  # ```
  #
  # Output
  # ```
  # [100, 215.443, 464.159, 1000]
  # ```
  def self.logarithmic_space(start, stop, num = 50, endpoint = true, base = 10.0)
    y = Tensor.linear_space(start, stop, num: num, endpoint: endpoint)
    base ** y
  end

  # Returns evenly spaced numbers on a log scale (geometric progression)
  # The sequence begins with `base ** start` and ends at `base ** step`
  #
  # Example
  # ```
  # a = Tensor(Float64).geometric_space(1, 1000, num: 4)
  # puts a
  # ```
  #
  # Output
  # ```
  # [1, 10, 100, 1000]
  # ```
  def self.geometric_space(start, stop, num = 50, endpoint = true)
    if start == 0 || stop == 0
      raise NumInternal::ValueError.new "Geometric sequence cannot include zero"
    end

    out_sign = 1.0

    if start < 0 && stop < 0
      start, stop = -start, -stop
      out_sign = -out_sign
    end

    log_start = Math.log(start, 10.0)
    log_stop = Math.log(stop, 10.0)

    Tensor.logarithmic_space(log_start, log_stop, num: num, endpoint: endpoint, base: 10.0) * out_sign
  end

  # Returns a Tensor from a provided range, the dtype of the `Tensor` is
  # inferred from the endpoints of the range
  #
  # Example
  # ```
  # a = Tensor.from_range(1..6)
  # puts a
  # ```
  #
  # Output
  # ```
  # [1, 2, 3, 4, 5, 6]
  # ```
  def self.from_range(rng : Range(T, T), step = 1)
    last = rng.excludes_end? ? rng.end : rng.end + T.new(1)
    self.range(rng.begin, last, T.new(step))
  end

  # Return a 2-D array with ones on the kth diagonal and zeros elsewhere.
  #
  # Example
  # ```
  # puts Tensor(Float64).eye(3, 4, k: 1)
  # ```
  #
  # Output
  # ```
  # [[0, 1, 0, 0],
  #  [0, 0, 1, 0],
  #  [0, 0, 0, 1]]
  # ```
  def self.eye(m : Int, n : Int? = nil, k : Int = 0)
    n = n.nil? ? m : n.as(Int32)
    Tensor.new(Int32.new(m), n) do |i, j|
      i == j - k ? T.new(1) : T.new(0)
    end
  end

  # Return an identity matrix
  #
  # Example
  # ```
  # puts Tensor(Float64).identity(2)
  # ```
  #
  # Output
  # ```
  # [[1, 0],
  #  [0, 1]]
  # ```
  def self.identity(n : Int)
    n32 = Int32.new(n)
    Tensor.new(n32, n32) do |i, j|
      i == j ? T.new(1) : T.new(0)
    end
  end

  # Return a matrix with a 1D `Tensor` along the diagonal
  #
  # Example
  # ```
  # a = Tensor(Float32).full([3], 5)
  # puts Tensor.diag(a)
  # ```
  #
  # Output
  # ```
  # [[5, 0, 0],
  #  [0, 5, 0],
  #  [0, 0, 5]]
  # ```
  def self.diag(a : Tensor(T), k : Int32 = 0)
    if a.ndims > 1
      raise "Only 1 dimensional Tensors are supported"
    end
    iter = NumInternal::UnsafeNDFlatIter.new(a)
    Tensor(T).new(a.shape[0], a.shape[0]) do |i, j|
      i == j - k ? iter.next.value : T.new(0)
    end
  end

  # Generate a Vandermonde matrix.
  #
  # The columns of the output matrix are powers of the input vector.
  # The order of the powers is determined by the increasing boolean argument.
  # Specifically, when increasing is False, the i-th output column is
  # the input vector raised element-wise to the power of N - i - 1.
  # Such a matrix with a geometric progression in each row is
  # named for Alexandre- Theophile Vandermonde.
  #
  # Example
  # ```
  # a = Tensor.range(1, 5)
  # puts Tensor.vander(a, 3)
  # ```
  #
  # Output
  # ```
  # [[1, 1, 1],
  #  [4, 2, 1],
  #  [9, 3, 1],
  #  [16, 4, 1]]
  # ```
  def self.vander(x : Tensor(T), n : Int32? = nil, increasing : Bool = false)
    if x.ndims > 1
      raise NumInternal::ShapeError.new("Vandermonde matrices must
        be initialized with a one-dimensional Tensor")
    end
    n = n.nil? ? x.size : n.as(Int32)
    Tensor(T).new(x.size, n) do |i, j|
      offset = increasing ? j : n - j - 1
      x[i].value ** offset
    end
  end
end
