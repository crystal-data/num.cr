require "./tensor"

# A module primarily responsible for `Tensor`
# creation routines.
#
# This module should be namespaced as part of the
# external API to provide user facing methods
# for creation.
module Bottle::Creation
  extend self

  # Initializes a `Tensor` with an uninitialized slice
  # of data.
  #
  # ```crystal
  # f = empty(5, dtype: Int32)
  # f # => Tensor[0, 0, 0, 0, 0]
  # ```
  def empty(shape : Array(Int32), dtype : U.class = Float64) forall U
    Tensor(U).new(shape)
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
    Tensor(U).new(other.shape)
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
    Tensor.new(m, n) do |i, j|
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
    Tensor.new(n, n) do |i, j|
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
  def ones(shape : Array(Int32), dtype : U.class = Float64) forall U
    Tensor(U).new(shape) { |_| U.new(1) }
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
    Tensor(U).new(other.shape) { |_| U.new(1) }
  end

  # Initializes a `Tensor` of the given `size` and `dtype`,
  # filled with zeros.
  #
  # ```crystal
  # f = zeros(5, dtype: Int32)
  # f # => Tensor[0, 0, 0, 0, 0]
  # ```
  def zeros(shape : Array(Int32), dtype : U.class = Float64) forall U
    Tensor(U).new(shape) { |_| U.new(0) }
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
  def zeros_like(other : NDTensor, dtype : U.class = Float64) forall U
    Tensor(U).new(other.shape) { |_| U.new(0) }
  end

  # Initializes a `Tensor` of the given `size` and `dtype`,
  # filled with the given value.
  #
  # ```crystal
  # f = full(5, 3, dtype: Int32)
  # f # => Tensor[3, 3, 3, 3, 3]
  # ```
  def full(shape : Array(Int32), x : Number, dtype : U.class = Float64) forall U
    Tensor(U).new(shape) { |_| U.new(x) }
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
  def full_like(other : NDTensor, x : Number, dtype : U.class = Float64) forall U
    Tensor(U).new(other.shape) { |_| U.new(x) }
  end

  # Returns a Matrix with the given
  # `Tensor` set along the diagonal.
  #
  # TODO: Support *k* offsets
  def diag(a : Tensor(U), k : Int32 = 0) forall U
    if a.ndims > 1
      raise "Only 1 dimensional Tensors are supported"
    end
    Tensor.new(a.shape[0], a.shape[0]) do |i, j|
      i == j - k ? a[[i]] : U.new(0)
    end
  end

  # Return evenly spaced values within a given interval.
  #
  # Values are generated within the half-open interval [start, stop)
  # (in other words, the interval including start but excluding stop).
  #
  # ```crystal
  # B.arange(1, 5) # => Tensor[1, 2, 3, 4]
  # ```
  def arange(start : Int32, stop : Int32, step : Number = 1, dtype : U.class = Int32) forall U
    r = stop - start
    num = (r / step).ceil
    if stop <= start || !num
      raise "vrange must return at least one value"
    end
    Tensor.new([Int32.new(num)]) { |i| U.new(start + (i * step)) }
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
      y[[y.shape[0] - 1]] = stop
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

  def tri(n : Int32, m : Int32? = nil, k : Int32 = 0, dtype = U.class = Float64) forall U
    Tensor(U).new(n, m.nil? ? n : m.as(Int32)) do |i, j|
      i >= j - k ? U.new(1) : U.new(0)
    end
  end

  macro triangulars(uplo, op)
    def tri{{uplo}}(t : Tensor(U), k = 0) forall U
      t = t.dup('C')
      iter = t.unsafe_iter
      if t.ndims == 1
        t[0] = 0
        return t
      elsif t.ndims == 2
        return Tensor(U).new(t.shape[0], t.shape[1]) do |i, j|
          val = iter.next
          i {{op.id}} j - k ? val.value : U.new(0)
        end
      else
        t.matrix_iter.each do |n|
          niter = n.unsafe_iter
          n[...] = Tensor(U).new(n.shape[0], n.shape[1]) do |i, j|
            val = niter.next
            i {{op.id}} j - k ? val.value : U.new(0)
          end
        end
      end
      t
    end
  end

  triangulars(u, :<=)
  triangulars(l, :>=)

  def vander(x : Tensor(U), n : Int32? = nil, increasing : Bool = false) forall U
    if x.ndims > 1
      raise Exceptions::ShapeError.new("Vandermonde matrices must
        be initialized with a one-dimensional Tensor")
    end
    n = n.nil? ? x.size : n.as(Int32)
    Tensor(U).new(x.size, n) do |i, j|
      offset = increasing ? j : n - j - 1
      x[[i]] ** offset
    end
  end

  # Count number of occurrences of each value in array of non-negative ints.
  #
  # The number of bins (of size 1) is one larger than the largest value in x.
  # If minlength is specified, there will be at least this number of bins
  # in the output array (though it will be longer if necessary, depending
  # on the contents of x). Each bin gives the number of occurrences of its
  # index value in x. If weights is specified the input array is weighted
  # by it, i.e. if a value n is found at position i, out[n] += weight[i]
  # instead of out[n] += 1.
  #
  # ```
  # t = Tensor.random(0...10, [10])
  # t           # => Tensor([7, 2, 2, 7, 0, 7, 6, 6, 0, 6])
  # bincount(t) # => Tensor([2, 0, 2, 0, 0, 0, 3, 3, 0, 0])
  # ```
  def bincount(x : Tensor(Int32), min_count = 0)
    if x.ndims != 1
      raise Exceptions::ShapeError.new("Input must be 1-dimensional")
    end

    sz = Math.max(min_count, x.size)
    ret = zeros([sz], dtype: Int32)
    x.flat_iter.each do |p|
      ret[[p.value]] += 1
    end
    ret
  end

  # Count number of occurrences of each value in array of non-negative ints.
  #
  # The number of bins (of size 1) is one larger than the largest value in x.
  # If minlength is specified, there will be at least this number of bins
  # in the output array (though it will be longer if necessary, depending
  # on the contents of x). Each bin gives the number of occurrences of its
  # index value in x. If weights is specified the input array is weighted
  # by it, i.e. if a value n is found at position i, out[n] += weight[i]
  # instead of out[n] += 1.
  #
  # ```
  # t = Tensor.random(0...10, [10])
  # t           # => Tensor([7, 2, 2, 7, 0, 7, 6, 6, 0, 6])
  # bincount(t) # => Tensor([2, 0, 2, 0, 0, 0, 3, 3, 0, 0])
  # ```
  def bincount(x : Tensor(Int32), weights : Tensor(U), min_count = 0) forall U
    if x.ndims != 1
      raise Exceptions::ShapeError.new("Input must be 1-dimensional")
    end

    if x.shape != weights.shape
      raise "Weights do not match input"
    end

    sz = Math.max(min_count, x.size)
    ret = zeros([sz], dtype: U)
    x.flat_iter.zip(weights.flat_iter).each do |i, j|
      ret[[i.value]] += j.value
    end
    ret
  end
end
