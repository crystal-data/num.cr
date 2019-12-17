require "../base/base"
require "../base/arrayprint"
require "../core/math"
require "../core/reductions"
require "./iter"
require "complex"
require "../libs/cblas"
require "../testing/testing"

class Num::Tensor(T) < Num::BaseArray(T)
  # Compile time checking of data types of a `Tensor` to ensure
  # mixing data types is not allowed, not are bad data types
  # allowed into the `Tensor`
  getter basetype = Tensor

  def basetype(t : U.class) forall U
    Tensor(U)
  end

  def to_unsafe
    {% if T == Complex %}
      buffer.unsafe_as(Pointer(LibCblas::ComplexDouble))
    {% else %}
      buffer
    {% end %}
  end

  def check_type
    {% unless T == Float32 || T == Float64 || T == Int16 || T == Int32 || \
                 T == Int8 || T == UInt16 || T == UInt32 || T == UInt64 || \
                 T == UInt8 || T == Bool || T == Complex %}
      {% raise "Bad dtype: #{T}. #{T} is not supported for Char Arrays" %}
    {% end %}
  end

  # A flexible method to create `Tensor`'s of arbitrary shapes
  # filled with random values of arbitrary types.  Since
  # Ranges can contain any dtype, the type of tensor is
  # inferred from the passed range, and a new `Tensor` is
  # sampled using the provided shape.
  #
  # ```
  # t = Tensor.random(0...10, [2, 2])
  # t # =>
  # Tensor([[5, 9],
  #         [3, 9]])
  # ```
  def self.random(r : Range(U, U), _shape : Array(Int32)) forall U
    if _shape.size == 0
      Tensor(U).new(_shape)
    else
      new(_shape) { |_| Random.rand(r) }
    end
  end

  def self.from_array(_shape : Array(Int32), _data : Array)
    flat = _data.flatten
    ptr = flat.to_unsafe
    Testing.assert_compatible_shape(_shape, flat.size)
    if _shape.size == 0
      Tensor(typeof(flat[0])).new(_shape)
    else
      new(_shape) { |i| ptr[i] }
    end
  end

  # Creates a string representation of a `Tensor`.  The implementation
  # of this is a bit of a mess, but I am happy with the results currently,
  # it could however be cleaned up to handle long floating point values
  # more precisely.
  def to_s(io)
    io << "Tensor(" << ArrayPrint.array2string(self, prefix: "Tensor(") << ")"
  end

  def inspect(io)
    to_s(io)
  end

  def pretty_print(pp)
    pp.text ArrayPrint.array2string(self, separator: " ")
  end

  def matrix_iter
    MatrixIter.new(self)
  end

  def apply_along_axis(axis = -1)
    if axis < 0
      axis += ndims
    end
    if axis < 0 || axis >= ndims
      raise Exceptions::ShapeError.new("Dimension out of range for Tensor")
    end
    if ndims == 1
      yield self
    else
      ts = shape.dup
      ts.delete_at(axis)
      iterations = ts.reduce { |i, j| i * j }
      buff = to_unsafe
      iterations.times do |_|
        yield Tensor(T).new(buff, [shape[axis]], [strides[axis]], flags, self, false)
        buff += strides[-1]
      end
    end
  end

  private def triu2d(a : Tensor(T), k)
    m, n = a.shape
    a.flat_iter_indexed do |el, idx|
      i = idx // n
      j = idx % n
      if i > j - k
        el.value = T.new(0)
      end
    end
  end

  private def tril2d(a : Tensor(T), k)
    m, n = a.shape
    a.flat_iter_indexed do |el, idx|
      i = idx // n
      j = idx % n
      if i < j - k
        el.value = T.new(0)
      end
    end
  end

  def triu!(k = 0)
    if ndims == 2
      triu2d(self, k)
    else
      matrix_iter.each do |subm|
        triu2d(subm, k)
      end
    end
  end

  def tril!(k = 0)
    if ndims == 2
      tril2d(self, k)
    else
      matrix_iter.each do |subm|
        tril2d(subm, k)
      end
    end
  end

  # Elementwise addition of a {{klass}}} to another equally
  # sized {{klass}}} or scalar
  def +(other)
    BMath.add(self, other)
  end

  def -
    self * -1
  end

  # Elementwise subtraction of a {{klass}}} to another equally
  # sized {{klass}}} or scalar
  def -(other)
    BMath.subtract(self, other)
  end

  # Elementwise multiplication of a {{klass}}} to another equally
  # sized {{klass}}} or scalar
  def *(other)
    BMath.multiply(self, other)
  end

  def **(other)
    BMath.power(self, other)
  end

  # Elementwise division of a {{klass}}} to another equally
  # sized {{klass}}} or scalar
  def /(other)
    BMath.divide(self, other)
  end

  def //(other)
    BMath.floordiv(self, other)
  end

  # Elementwise modulus of a {{klass}}} to another equally
  # sized {{klass}}} or scalar
  def %(other)
    BMath.modulo(self, other)
  end

  # Elementwise bitwise and of a {{klass}}} to another equally
  # sized {{klass}}} or scalar
  def &(other)
    Binary.bitwise_and(self, other)
  end

  # Elementwise bitwise or of a {{klass}}} to another equally
  # sized {{klass}}} or scalar
  def |(other)
    Binary.bitwise_or(self, other)
  end

  # Elementwise bitwise xor of a {{klass}}} to another equally
  # sized {{klass}}} or scalar
  def ^(other)
    Binary.bitwise_xor(self, other)
  end

  # Elementwise left shift of a {{klass}}} to another equally
  # sized {{klass}}} or scalar
  def <<(other)
    Binary.left_shift(self, other)
  end

  # Elementwise right shift of a {{klass}}} to another equally
  # sized {{klass}}} or scalar
  def >>(other)
    Binary.right_shift(self, other)
  end

  # Elementwise greater than of a {{klass}}} to another equally
  # sized {{klass}}} or scalar
  def >(other)
    BMath.greater(self, other)
  end

  # Elementwise greater equal than of a {{klass}}} to another equally
  # sized {{klass}}} or scalar
  def >=(other)
    BMath.greater_equal(self, other)
  end

  # Elementwise less than of a {{klass}}} to another equally
  # sized {{klass}}} or scalar
  def <(other)
    BMath.less(self, other)
  end

  # Elementwise less equals of a {{klass}}} to another equally
  # sized {{klass}}} or scalar
  def <=(other)
    BMath.less_equal(self, other)
  end

  # Elementwise equals of a {{klass}}} to another equally
  # sized {{klass}}} or scalar
  def ==(other)
    BMath.equal(self, other)
  end

  def cumsum(axis : Int32)
    self.accumulate_fast(axis) { |i, j| i.value += j.value }
  end

  # Computes the total sum of a Tensor
  #
  # ```
  # v = Tensor.new [1, 2, 3, 4]
  # sum(v) # => 10
  # ```
  def sum
    Statistics.sum(self)
  end

  # Computes the total sum of a Tensor
  #
  # ```
  # v = Tensor.new [1, 2, 3, 4]
  # sum(v) # => 10
  # ```
  def sum(axis : Int32)
    Statistics.sum(self, axis)
  end

  # Computes the average of all Tensor values
  #
  # ```
  # v = Tensor.new [1, 2, 3, 4]
  # mean(v) # => 2.5
  # ```
  def mean
    Statistics.mean(self)
  end

  # Computes the average of all Tensor values
  #
  # ```
  # v = Tensor.new [1, 2, 3, 4]
  # mean(v) # => 2.5
  # ```
  def mean(axis : Int32)
    Statistics.mean(self, axis)
  end

  # Computes the standard deviation of a Tensor
  #
  # ```
  # v = Tensor.new [1, 2, 3, 4]
  # std(v) # => 1.118
  # ```
  def std
    Statistics.std(self)
  end

  # Computes the median value of a Tensor
  #
  # ```
  # v = Tensor.new [1, 2, 3, 4]
  # median(v) # => 2.5
  # ```
  def median
    Statistics.median(self)
  end

  # Computes the maximum value of a Tensor
  #
  # ```
  # v = Tensor.new [1, 2, 3, 4]
  # max(v) # => 4
  # ```
  def max
    Statistics.max(self)
  end

  # Computes the maximum value of a Tensor
  #
  # ```
  # v = Tensor.new [1, 2, 3, 4]
  # max(v) # => 4
  # ```
  def max(axis : Int32)
    Statistics.max(self, axis)
  end

  # Computes the minimum value of a Tensor
  #
  # ```
  # v = Tensor.new [1, 2, 3, 4]
  # min(v) # => 1
  # ```
  def min
    Statistics.min(self)
  end

  # Computes the minimum value of a Tensor
  #
  # ```
  # v = Tensor.new [1, 2, 3, 4]
  # min(v) # => 1
  # ```
  def min(axis : Int32)
    Statistics.min(self, axis)
  end

  # Computes the "peak to peak" of a Tensor (max - min)
  #
  # ```
  # v = Tensor.new [1, 2, 3, 4]
  # v.ptp # => 3
  # ```
  def ptp
    max - min
  end

  # Computes the "peak to peak" of a Tensor (max - min)
  #
  # ```
  # v = Tensor.new [1, 2, 3, 4]
  # v.ptp # => 3
  # ```
  def ptp(axis : Int32)
    max(axis) - min(axis)
  end

  def real
    {% if T == Complex %}
      ret = Tensor(Float64).new(shape)
      ret.flat_iter.zip(flat_iter) do |i, j|
        i.value = j.value.real
      end
      ret
    {% else %}
      raise Exceptions::TypeError.new("Tensor is not complex")
    {% end %}
  end

  def imag
    {% if T == Complex %}
      ret = Tensor(Float64).new(shape)
      ret.flat_iter.zip(flat_iter) do |i, j|
        i.value = j.value.imag
      end
      ret
    {% else %}
      raise Exceptions::TypeError.new("Tensor is not complex")
    {% end %}
  end
end
