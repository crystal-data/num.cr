require "../core/tensor"
require "../core/matrix"

# A module primarily responsible for `Tensor`
# and `Matrix` creation routines.
#
# This module should be namespaced as part of the
# external API to provide user facing methods
# for creation.
module Bottle::Internal::Numeric
  extend self

  # Initializes a `Tensor` with an uninitialized slice
  # of data.
  #
  # ```crystal
  # f = empty(5, dtype: Int32)
  # f # => Tensor[0, 0, 0, 0, 0]
  # ```
  def empty(size : Int32, dtype : U.class = Float64) forall U
    Tensor.new(
      Pointer(U).malloc(size),
      size,
      1,
      true
    )
  end

  # Initializes a `Matrix` with an uninitialized slice
  # of data.
  #
  # ```crystal
  # m = empty(3, 3, dtype: Int32)
  # m # => Matrix[[0, 0, 0], [0, 0, 0], [0, 0, 0]]
  # ```
  def empty(rows : Int32, cols : Int32, dtype : U.class = Float64) forall U
    Matrix.new(
      Pointer(U).malloc(rows * cols),
      rows,
      cols,
      cols,
      true
    )
  end

  # Initializes a `Tensor` with an uninitialized slice
  # of data that is the same size as a given
  # `Tensor`.
  #
  # ```crystal
  # t = Tensor.new [1, 2, 3]
  #
  # f = empty_like(t, dtype: Int32)
  # f # => Tensor[0, 0, 0]
  # ```
  def empty_like(other : Tensor, dtype : U.class = Float64) forall U
    Tensor.new(
      Pointer(U).malloc(other.size),
      size,
      1,
      true
    )
  end

  # Initializes a `Matrix` with an uninitialized slice
  # of data.  The number of rows and columns are inferred
  # from the passed `Matrix`
  #
  # ```crystal
  # m = Matrix.new [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
  #
  # e = empty_like(m, dtype: Int32)
  # e # => Matrix[[0, 0, 0], [0, 0, 0], [0, 0, 0]]
  # ```
  def empty_like(other : Matrix, dtype : U.class = Float64) forall U
    Matrix.new(
      Pointer(U).malloc(other.nrows * other.ncols),
      other.nrows,
      other.ncols,
      other.ncols,
      true
    )
  end

  # Return a `Matrix` with ones on the diagonal and
  # zeros elsewhere.
  #
  # ```
  # m = eye(3, dtype: Int32)
  #
  # m # => [[1, 0, 0], [0, 1, 0], 0, 0, 1]
  # ```
  def eye(m : Int32, n : Int32? = nil, k : Int32 = 0, dtype : U.class = Float64) forall U
    n = n.nil? ? m : n.as(Int32)
    Matrix.new(m, n) do |i, j|
      i == j - k ? U.new(1) : U.new(0)
    end
  end

  # Returns the identify matrix with dimensions
  # *m* by *m*
  #
  # ```
  # m = identity(3)
  #
  # m # => [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
  # ```
  def identity(n : Int32, dtype : U.class = Float64) forall U
    Matrix.new(n, n) do |i, j|
      i == j ? U.new(1) : U.new(0)
    end
  end

  # Initializes a `Tensor` of the given `size` and `dtype`,
  # filled with ones.
  #
  # ```crystal
  # f = ones(5, dtype: Int32)
  # f # => Tensor[1, 1, 1, 1, 1]
  # ```
  def ones(size : Int32, dtype : U.class = Float64) forall U
    Tensor.new(
      Pointer(U).malloc(size, U.new(1)),
      size,
      1,
      true
    )
  end

  # Initializes a `Matrix` with given rows and columns,
  # filled with ones.
  #
  # ```crystal
  # m = ones(3, 3, dtype: Int32)
  # m # => Matrix[[1, 1, 1], [1, 1, 1], [1, 1, 1]]
  # ```
  def ones(rows : Int32, cols : Int32, dtype : U.class = Float64) forall U
    Matrix.new(
      Pointer(U).malloc(rows * cols, U.new(1)),
      rows,
      cols,
      cols,
      true
    )
  end

  # Initializes a `Tensor` filled with ones, whose size
  # is inferred from a given `Tensor`
  #
  # ```crystal
  # t = Tensor.new [1, 2, 3]
  #
  # f = ones_like(t, dtype: Int32)
  # f # => Tensor[1, 1, 1]
  # ```
  def ones_like(other : Tensor, dtype : U.class = Float64) forall U
    Tensor.new(
      Pointer(U).malloc(other.size, U.new(1)),
      size,
      1,
      true
    )
  end

  # Initializes a `Matrix` filled with ones. The number of
  # rows and columns are inferred from the passed
  # `Matrix`
  #
  # ```crystal
  # m = Matrix.new [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
  #
  # e = ones_like(m, dtype: Int32)
  # e # => Matrix[[1, 1, 1], [1, 1, 1], [1, 1, 1]]
  # ```
  def ones_like(other : Matrix, dtype : U.class = Float64) forall U
    Matrix.new(
      Pointer(U).malloc(other.nrows * other.ncols, U.new(1)),
      other.nrows,
      other.ncols,
      other.ncols,
      true
    )
  end

  # Initializes a `Tensor` of the given `size` and `dtype`,
  # filled with zeros.
  #
  # ```crystal
  # f = zeros(5, dtype: Int32)
  # f # => Tensor[0, 0, 0, 0, 0]
  # ```
  def zeros(size : Int32, dtype : U.class = Float64) forall U
    Tensor.new(
      Pointer(U).malloc(size, U.new(0)),
      size,
      1,
      true
    )
  end

  # Initializes a `Matrix` with given rows and columns,
  # filled with zeros.
  #
  # ```crystal
  # m = zeros(3, 3, dtype: Int32)
  # m # => Matrix[[0, 0, 0], [0, 0, 0], [0, 0, 0]]
  # ```
  def zeros(rows : Int32, cols : Int32, dtype : U.class = Float64) forall U
    Matrix.new(
      Pointer(U).malloc(rows * cols, U.new(0)),
      rows,
      cols,
      cols,
      true
    )
  end

  # Initializes a `Tensor` filled with zeros, whose size
  # is inferred from a given `Tensor`
  #
  # ```crystal
  # t = Tensor.new [1, 2, 3]
  #
  # f = zeros_like(t, dtype: Int32)
  # f # => Tensor[0, 0, 0]
  # ```
  def zeros_like(other : Tensor, dtype : U.class = Float64) forall U
    Tensor.new(
      Pointer(U).malloc(other.size, U.new(0)),
      size,
      1,
      true
    )
  end

  # Initializes a `Matrix` filled with zeros. The number of
  # rows and columns are inferred from the passed
  # `Matrix`
  #
  # ```crystal
  # m = Matrix.new [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
  #
  # e = ones_like(m, dtype: Int32)
  # e # => Matrix[[0, 0, 0], [0, 0, 0], [0, 0, 0]]
  # ```
  def zeros_like(other : Matrix, dtype : U.class = Float64) forall U
    Matrix.new(
      Pointer(U).malloc(other.nrows * other.ncols, U.new(0)),
      other.nrows,
      other.ncols,
      other.ncols,
      true
    )
  end

  # Initializes a `Tensor` of the given `size` and `dtype`,
  # filled with the given value.
  #
  # ```crystal
  # f = full(5, 3, dtype: Int32)
  # f # => Tensor[3, 3, 3, 3, 3]
  # ```
  def full(size : Int32, x : Number, dtype : U.class = Float64) forall U
    Tensor.new(
      Pointer(U).malloc(size, U.new(x)),
      size,
      1,
      true
    )
  end

  # Initializes a `Matrix` with given rows and columns,
  # filled with the provided value.
  #
  # ```crystal
  # m = full(3, 3, -1, dtype: Int32)
  # m # => Matrix[[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]
  # ```
  def full(rows : Int32, cols : Int32, x : Number, dtype : U.class = Float64) forall U
    Matrix.new(
      Pointer(U).malloc(rows * cols, U.new(x)),
      rows,
      cols,
      cols,
      true
    )
  end

  # Initializes a `Tensor` filled with the provided value, whose size
  # is inferred from a given `Tensor`
  #
  # ```crystal
  # t = Tensor.new [1, 2, 3]
  #
  # f = full_like(t, -1, dtype: Int32)
  # f # => Tensor[-1, -1, -1]
  # ```
  def full_like(other : Tensor, x : Number, dtype : U.class = Float64) forall U
    Tensor.new(
      Pointer(U).malloc(other.size, U.new(x)),
      size,
      1,
      true
    )
  end

  # Initializes a `Matrix` filled with the given value. The number of
  # rows and columns are inferred from the passed
  # `Matrix`
  #
  # ```crystal
  # m = Matrix.new [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
  #
  # e = full_like(m, -1, dtype: Int32)
  # e # => Matrix[[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]
  # ```
  def full_like(other : Matrix, x : Number, dtype : U.class = Float64) forall U
    Matrix.new(
      Pointer(U).malloc(other.nrows * other.ncols, U.new(x)),
      other.nrows,
      other.ncols,
      other.ncols,
      true
    )
  end

  # Return evenly spaced values within a given interval.
  #
  # Values are generated within the half-open interval [start, stop)
  # (in other words, the interval including start but excluding stop).
  #
  # ```crystal
  # B.arange(1, 5) # => Tensor[1, 2, 3, 4]
  # ```
  def arange(start : Int32, stop : Int32, step : Int32 = 1, dtype : U.class = Int32) forall U
    r = stop - start
    num = r // step
    if stop <= start || !num
      raise "vrange must return at least one value"
    end
    Tensor.new(num) { |i| U.new(start + (i * step)) }
  end

  # Return evenly spaced values within a given interval.
  #
  # Values are generated within the half-open interval [start, stop)
  # (in other words, the interval including start but excluding stop).
  #
  # ```crystal
  # B.arange(5) # => Tensor[0, 1, 2, 3, 4]
  # ```
  def arange(stop : Int32, step : Int32 = 1, dtype : U.class = Int32) forall U
    arange(0, stop, step, dtype)
  end

  # Return evenly spaced numbers over a specified interval.
  # Returns `num` evenly spaced samples, calculated over the
  # interval [`start`, `stop`].
  # The endpoint of the interval can optionally be excluded.
  #
  # ```crystal
  # B.linspace(0, 1, 5) # => Tensor[0.0, 0.25, 0.5, 0.75, 1.0]
  #
  # B.linspace(0, 1, 5, endpoint: false) # => Tensor[0.0, 0.2, 0.4, 0.6, 0.8]
  # ```
  def linspace(start : Number, stop : Number, num = 50, endpoint = true)
    if num < 0
      raise "Number of samples, #{num}, must be non-negative"
    end
    div = endpoint ? num - 1 : num
    start = start * 1.0
    stop = stop * 1.0
    y = arange(num, dtype: Float64)
    delta = stop - start
    if num > 1
      step = delta / div
      if step == 0
        raise "Cannot have a step of 0"
      end
      y = y * step
    else
      y = y * delta
    end
    y += start
    if endpoint && num > 1
      y[y.size - 1] = stop
    end
    y
  end

  # Return numbers spaced evenly on a log scale.
  # In linear space, the sequence starts at ``base ** start``
  # (`base` to the power of `start`) and ends with ``base ** stop``
  # (see `endpoint` below).
  #
  # ```crystal
  # B.logspace(2.0, 3.0, num = 4) # => Tensor[100.0, 215.44346900318845, 464.15888336127773, 1000.0]
  # ```
  def logspace(start, stop, num = 50, endpoint = true, base = 10.0)
    y = linspace(start, stop, num: num, endpoint: endpoint)
    power(base, y)
  end

  # Return numbers spaced evenly on a log scale (a geometric progression).
  # This is similar to `logspace`, but with endpoints specified directly.
  # Each output sample is a constant multiple of the previous.
  #
  # ```
  # geomspace(1, 1000, 4) # => Tensor[1.0, 10.0, 100.0, 1000.0]
  # ```
  def geomspace(start, stop, num = 50, endpoint = true)
    if start == 0 || stop == 0
      raise "Geometric sequence cannot include zero"
    end

    out_sign = 1.0

    if start < 0 && stop < 0
      start, stop = -start, -stop
      out_sign = -out_sign
    end

    log_start = Math.log(start, 10.0)
    log_stop = Math.log(stop, 10.0)

    logspace(log_start, log_stop, num: num, endpoint: endpoint, base: 10.0) * out_sign
  end

  # Returns the diagonal of a Matrix
  # as a 1D `Tensor`.  Data is copied into
  # the new `Tensor`
  #
  # TODO: Support *k* offsets
  def diag(a : Matrix)
    Tensor.new(a.nrows) do |i|
      a[i, i]
    end
  end

  # Returns a Matrix with the given
  # `Tensor` set along the diagonal.
  #
  # TODO: Support *k* offsets
  def diag(a : Tensor(U)) forall U
    Matrix.new(a.size, a.size) do |i, j|
      i == j ? a[i] : U.new(0)
    end
  end

  # Returns a `Tensor` with ones at and below
  # the given diagonal and zeros elsewhere.
  #
  # ```
  # p tri(3, 5, 2, dtype: Int32)
  #
  # # Matrix[[  1  1  1  0  0]
  # #        [  1  1  1  1  0]
  # #        [  1  1  1  1  1]]
  # ```
  def tri(m : Int32, n : Int32?, k = 0, dtype : U.class = Float64) forall U
    n = n.nil? ? m.as(Int32) : n.as(Int32)
    Matrix.new(m, n) do |i, j|
      i >= j - k ? U.new(1) : U.new(0)
    end
  end

  # Lower triangle of an array.
  #
  # Return a copy of an `Matrix with elements
  # above the k-th diagonal zeroed.
  def tril(a : Matrix(U), k = 0) forall U
    Matrix.new(a.nrows, a.ncols) do |i, j|
      i >= j - k ? a[i, j] : U.new(0)
    end
  end

  # Upper triangle of an array.
  #
  # Return a copy of an `Matrix with elements
  # below the k-th diagonal zeroed.
  def triu(a : Matrix(U), k = 0) forall U
    Matrix.new(a.nrows, a.ncols) do |i, j|
      i <= j - k ? a[i, j] : U.new(0)
    end
  end

  # Generate a Vandermonde matrix.
  #
  # The columns of the output matrix are powers of the input vector.
  # The order of the powers is determined by the increasing boolean
  # argument. Specifically, when increasing is False, the
  # i-th output column is the input vector raised element-wise
  # to the power of N - i - 1. Such a matrix with a
  # geometric progression in each row is named for Alexandre- Theophile
  # Vandermonde.
  def vander(a : Tensor, n : Int32? = nil, increasing = false)
    n = n.nil? ? a.size.as(Int32) : n.as(Int32)
    Matrix.new(a.size, n) do |i, j|
      increasing ? a[i]**j : a[i]**(n - (j + 1))
    end
  end
end
