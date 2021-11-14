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

class Tensor(T, S)
  # Initialize a Tensor from a storage instance, a shape, an order, and a
  # data type.  This should primarily be used by internal methods, since it
  # assumes the contiguity of the storage.
  #
  # The dtype is required to infer T without having it explicitly provided
  def initialize(@data : S, shape : Array(Int), order : Num::OrderType = Num::RowMajor, dtype : T.class = T)
    assert_types
    @shape = shape.map &.to_i
    @strides = Num::Internal.shape_to_strides(shape, order)
    @data.update_metadata(@shape, @strides)
    @size = @shape.product
    @offset = 0
    @flags = Num::ArrayFlags::All
    update_flags
  end

  # Initialize a Tensor from a storage instance, a shape, strides, an offset,
  # and a data type.  This should primarily be used by internal methods,
  # since it assumes the passed shape and strides correspond to the
  # storage provided.
  #
  # The dtype is required to infer T without having it explicitly provided
  def initialize(@data : S, @shape : Array(Int32), @strides : Array(Int32), @offset : Int32, dtype : T.class = T)
    @data.update_metadata(@shape, @strides)
    @size = @shape.product
    @flags = Num::ArrayFlags::All
    update_flags
  end

  # Initialize a Tensor from a storage instance, a shape, strides, an offset,
  # flags, and a data type.  This should primarily be used by internal methods,
  # since it assumes the passed shape and strides correspond to the
  # storage provided.
  #
  # The dtype is required to infer T without having it explicitly provided
  def initialize(@data : S, @shape : Array(Int32), @strides : Array(Int32), @offset : Int32, @flags : Num::ArrayFlags, dtype : T.class = T)
    @data.update_metadata(@shape, @strides)
    @size = @shape.product
    update_flags
  end

  # Private initialization method to allow Tensors to be created from Arrays,
  # without having to provide a specific generic type to the array.
  # Since the storage instance is S, having the array passed
  # allows Num to infer T
  private def initialize(@data : S, shape : Array(Int), from_array : Array(T))
    assert_types
    @shape = shape.map &.to_i
    @strides = Num::Internal.shape_to_strides(shape, Num::RowMajor)
    @size = @shape.product
    @offset = 0
    @flags = Num::ArrayFlags::All
    update_flags
  end

  # Initializes a Tensor onto a device with a provided shape and memory
  # layout.
  #
  # ## Examples
  #
  # ```
  # a = Tensor(Float32).new([3, 3, 2], device: OCL(Float32)) # => GPU Tensor
  # b = Tensor(Float32).new([2, 3, 4])                       # => CPU Tensor
  # ```
  def self.new(shape : Array(Int), order : Num::OrderType = Num::RowMajor)
    data = S.new(shape, order)
    new(data, shape, order, T)
  end

  # Initializes a Tensor onto a device with a provided shape and memory
  # layout, containing a specified value.
  #
  # ## Examples
  #
  # ```
  # a = Tensor.new([2, 2], 3.5) # => CPU Tensor filled with 3.5
  # ```
  def self.new(shape : Array(Int), value : T, device = CPU(T), order : Num::OrderType = Num::RowMajor)
    data = device.new(shape, order, value)
    new(data, shape, order, T)
  end

  # Creates a Tensor from a block onto a specified device. The type of the
  # Tensor is inferred from the return type of the block
  #
  # ## Examples
  #
  # ```
  # a = Tensor.new([3, 3, 2]) { |i| i } # => Int32 Tensor stored on a CPU
  # ```
  def self.new(shape : Array(Int), order : Num::OrderType = Num::RowMajor, device = CPU, &block : Int32 -> T)
    ptr = Pointer.malloc(shape.product) do |index|
      yield index
    end
    storage = device.new(ptr, shape, Num::Internal.shape_to_strides(shape, order))
    new(storage, shape, order, T)
  end

  # Creates a matrix Tensor from a block onto a specified device.  The type
  # of the Tensor is inferred from the return type of the block
  #
  # ## Examples
  #
  # ```
  # a = Tensor.new(3, 3) { |i, j| i / j } # => Float64 Tensor stored on a CPU
  # ```
  def self.new(m : Int, n : Int, device = CPU, &block : Int32, Int32 -> T)
    ptr = Pointer.malloc(m * n) do |idx|
      i = idx // n
      j = idx % n
      yield i, j
    end
    storage = device.new(ptr, [m, n], [n, 1])
    new(storage, [m, n], Num::RowMajor, T)
  end

  # Creates a Tensor from a standard library array onto a specified device.
  # The type of Tensor is inferred from the innermost element type, and
  # the Array's shape must be uniform along all subarrays.
  #
  # ## Examples
  #
  # ```
  # a = [[1, 2], [3, 4], [5, 6]]
  # Tensor.from_array(a, device: OCL) # => [3, 2] Tensor stored on a GPU
  # ```
  def self.from_array(a : Array, device = CPU)
    shape = Num::Internal.recursive_array_shape(a)
    flat = a.flatten
    storage = device.new(flat.to_unsafe, shape, Num::Internal.shape_to_strides(shape))
    new(storage, shape, from_array: flat)
  end

  # Creates a `Tensor` of a provided shape, filled with 0.  The generic type
  # must be specified.
  #
  # ## Arguments
  #
  # * shape : `Array(Int)` - shape of returned `Tensor`
  #
  # ## Examples
  #
  # ```
  # t = Tensor(Int8).zeros([3]) # => [0, 0, 0]
  # ```
  def self.zeros(shape : Array(Int)) : Tensor(T, S)
    self.new(S.new(shape, Num::RowMajor, T.new(0)), shape, Num::RowMajor, T)
  end

  # Creates a `Tensor` filled with 0, sharing the shape of another
  # provided `Tensor`
  #
  # ## Arguments
  #
  # * t : `Tensor` - `Tensor` to use for output shape
  #
  # ## Examples
  #
  # ```
  # t = Tensor(Int8, CPU(Int8)).new([3]) &.to_f
  # u = Tensor(Int8, CPU(Int8)).zeros_like(t) # => [0, 0, 0]
  # ```
  def self.zeros_like(t : Tensor) : Tensor(T, S)
    self.new(S.new(t.shape, Num::RowMajor, T.new(0)), t.shape, Num::RowMajor, T)
  end

  # Creates a `Tensor` of a provided shape, filled with 1.  The generic type
  # must be specified.
  #
  # ## Arguments
  #
  # * shape : `Array(Int)` - shape of returned `Tensor`
  #
  # ## Examples
  #
  # ```
  # t = Tensor(Int8, CPU(Int8)).ones([3]) # => [1, 1, 1]
  # ```
  def self.ones(shape : Array(Int)) : Tensor(T, S)
    self.new(S.new(shape, Num::RowMajor, T.new(1)), shape, Num::RowMajor, T)
  end

  # Creates a `Tensor` filled with 1, sharing the shape of another
  # provided `Tensor`
  #
  # ## Arguments
  #
  # * t : `Tensor` - `Tensor` to use for output shape
  #
  # ## Examples
  #
  # ```
  # t = Tensor(Int8, CPU(Int8)) &.to_f
  # u = Tensor(Int8, CPU(Int8)).ones_like(t) # => [0, 0, 0]
  # ```
  def self.ones_like(t : Tensor) : Tensor(T, S)
    self.new(S.new(t.shape, Num::RowMajor, T.new(1)), t.shape, Num::RowMajor, T)
  end

  # Creates a `Tensor` of a provided shape, filled with a value.  The generic type
  # is inferred from the value
  #
  # ## Arguments
  #
  # * shape : `Array(Int)` - shape of returned `Tensor`
  #
  # ## Examples
  #
  # ```
  # t = Tensor(Int8, CPU(Int8)).full([3], 1) # => [1, 1, 1]
  # ```
  def self.full(shape : Array(Int), value : Number) : Tensor(T, S)
    self.new(S.new(shape, Num::RowMajor, T.new(value)), shape, Num::RowMajor, T)
  end

  # Creates a `Tensor` filled with a value, sharing the shape of another
  # provided `Tensor`
  #
  # ## Arguments
  #
  # * t : `Tensor` - `Tensor` to use for output shape
  #
  # ## Examples
  #
  # ```
  # t = Tensor(Int8, CPU(Int8)) &.to_f
  # u = Tensor(Int8, CPU(Int8)).full_like(t, 3) # => [3, 3, 3]
  # ```
  def self.full_like(t : Tensor, value : Number) : Tensor(T, S)
    self.new(S.new(shape, Num::RowMajor, T.new(value)), shape, Num::RowMajor, T)
  end

  # Creates a flat `Tensor` containing a monotonically increasing
  # or decreasing range.  The generic type is inferred from
  # the inputs to the method
  #
  # ## Arguments
  #
  # * start : `T` - Beginning value for the range
  # * stop : `T` - End value for the range
  # * step : `T` - Offset between values of the range
  #
  # ## Examples
  #
  # ```
  # Tensor.range(0, 5, 2)       # => [0, 2, 4]
  # Tensor.range(5, 0, -1)      # => [5, 4, 3, 2, 1]
  # Tensor.range(0.0, 3.5, 0.7) # => [0  , 0.7, 1.4, 2.1, 2.8]
  # ```
  def self.range(start : T, stop : T, step : T, device = CPU)
    if start > stop && step > 0
      raise "Range must return at least one value"
    end

    r = stop - start
    n = (r / step).ceil.abs
    self.new([n.to_i], device: device) do |i|
      T.new(start + i * step)
    end
  end

  # :ditto:
  def self.range(stop : T, device = CPU)
    self.range(T.new(0), stop, T.new(1), device)
  end

  # :ditto:
  def self.range(start : T, stop : T, device = CPU)
    self.range(start, stop, T.new(1), device)
  end

  # Return evenly spaced numbers over a specified interval.
  # Returns `num` evenly spaced samples, calculated over the
  # interval [`start`, `stop`].
  # The endpoint of the interval can optionally be excluded.
  #
  # ## Arguments
  #
  # * start : `T` - Start of interval
  # * stop : `T` - End of interval
  # * num : `Int` - Number of samples
  # * endpoint : `Bool` - Indicates if endpoint of the interval should be
  #   included in the results
  # * device : `Num::Storage` - Backend for the `Tensor`
  #
  # ## Examples
  #
  # ```
  # Tensor.linspace(0_f32, 1_f32, 5) # => [0.0, 0.25, 0.5, 0.75, 1.0]
  #
  # Tensor.linspace(0_f32, 1_f32, 5, endpoint: false) # => [0.0, 0.2, 0.4, 0.6, 0.8]
  # ```
  def self.linear_space(
    start : T,
    stop : T,
    num : Int = 50,
    endpoint = true,
    device = CPU
  )
    unless num > 0
      raise Num::Exceptions::ValueError.new(
        "Number of samples must be non-negative"
      )
    end
    divisor = endpoint ? num - 1 : num
    result = Tensor.range(T.new(num), device: device)
    delta = stop - start

    if num > 1
      step = delta / divisor
      if step == 0
        raise Num::Exceptions::ValueError.new(
          "Step cannot be 0"
        )
      end
    else
      step = delta
    end

    Num.multiply!(result, step)
    Num.add!(result, start)

    if endpoint && num > 1
      result[-1] = stop
    end

    result
  end

  # Return numbers spaced evenly on a log scale.
  # In linear space, the sequence starts at ``base ** start``
  # (`base` to the power of `start`) and ends with ``base ** stop``
  # (see `endpoint` below).
  #
  # ## Arguments
  #
  # * start : `T` - Start of interval
  # * stop : `T` - End of interval
  # * num : `Int` - Number of samples
  # * endpoint : `Bool` - Indicates if endpoint should be included in the results
  # * device : `Num::Storage` - Backend for the `Tensor`
  #
  # ## Examples
  #
  # ```
  # Tensor.logarithmic_space(2.0, 3.0, num: 4)
  #
  # # [100    , 215.443, 464.159, 1000   ]
  # ```
  def self.logarithmic_space(
    start : T,
    stop : T,
    num = 50,
    endpoint = true,
    base : T = T.new(10.0),
    device = CPU
  )
    result = Tensor.linear_space(
      start,
      stop,
      num: num,
      endpoint: endpoint,
      device: device
    )
    Num.power(base, result)
  end

  # Return numbers spaced evenly on a log scale (a geometric progression).
  # This is similar to `logspace`, but with endpoints specified directly.
  # Each output sample is a constant multiple of the previous.
  #
  # ## Arguments
  #
  # * start : `T` - Start of interval
  # * stop : `T` - End of interval
  # * num : `Int` - Number of samples
  # * endpoint : `Bool` - Indicates if endpoint should be included in the results
  # * device : `Num::Storage`
  #
  # ## Examples
  #
  # ```
  # Tensor.geometric_space(1_f32, 1000_f32, 4) # => [1, 10, 100, 1000]
  # ```
  def self.geometric_space(
    start : T,
    stop : T,
    num : Int = 50,
    endpoint : Bool = true,
    device = CPU
  )
    if start == 0 || stop == 0
      raise Num::Exceptions::ValueError.new(
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
      base: T.new(10.0),
      device: device
    ) * out_sign
  end

  # Return a two-dimensional `Tensor` with ones along the diagonal,
  # and zeros elsewhere
  #
  # ## Arguments
  #
  # * m : `Int` - Number of rows in the returned `Tensor`
  # * n : `Int?` - Number of columns in the `Tensor`, defaults to `m`
  #   if nil
  # * offset : `Int` - Indicates which diagonal to fill with ones
  #
  # ## Examples
  #
  # ```
  # Tensor(Int8, CPU(Int8)).eye(3, offset: -1)
  #
  # # [[0, 0, 0],
  # #  [1, 0, 0],
  # #  [0, 1, 0]]
  #
  # Tensor(Int8, CPU(Int8)).eye(2)
  #
  # # [[1, 0],
  # #  [0, 1]]
  # ```
  def self.eye(m : Int, n : Int? = nil, offset : Int = 0)
    n = n.nil? ? m : n
    self.new(m, n, device: S) do |i, j|
      i == j - offset ? T.new(1) : T.new(0)
    end
  end

  # Returns an identity `Tensor` with ones along the diagonal,
  # and zeros elsewhere
  #
  # ## Arguments
  #
  # * n : `Int` - Number of rows and columns in output
  #
  # ## Examples
  #
  # ```
  # Tensor(Int8, CPU(Int8)).identity(2)
  #
  # # [[1, 0],
  # #  [0, 1]]
  # ```
  def self.identity(n : Int)
    self.new(n, n, device: S) do |i, j|
      i == j ? T.new(1) : T.new(0)
    end
  end

  # Deep-copies a `Tensor`.  If an order is provided, the returned
  # `Tensor`'s memory layout will respect that order.
  #
  # ## Arguments
  #
  # * order : `Num::OrderType` - Memory layout to use for the returned `Tensor`
  #
  # ## Examples
  #
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.dup # => [1, 2, 3]
  # ```
  def dup(order : Num::OrderType = Num::RowMajor)
    Num.dup(self, order)
  end

  # Return a shallow copy of a `Tensor`.  The underlying data buffer
  # is shared, but the `Tensor` owns its other attributes.  Changes
  # to a view of a `Tensor` will be reflected in the original `Tensor`
  #
  # ## Examples
  #
  # ```
  # a = Tensor(Int32, CPU(Int32)).new([3, 3])
  # b = a.view
  # b[...] = 99
  # a
  #
  # # [[99, 99, 99],
  # #  [99, 99, 99],
  # #  [99, 99, 99]]
  # ```
  def view : Tensor(T, S)
    Tensor(T, S).new(@data, @shape.dup, @strides.dup, @offset)
  end
end
