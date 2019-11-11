require "../base/base"
require "../base/print"
require "../core/ufunc"
require "./statistics"
require "../arrayops/math"
require "../arrayops/binary"

class Bottle::Tensor(T) < Bottle::BaseArray(T)
  # Compile time checking of data types of a `Tensor` to ensure
  # mixing data types is not allowed, not are bad data types
  # allowed into the `Tensor`
  getter basetype = Tensor

  protected def check_type
    {% unless T == Float32 || T == Float64 || T == Int16 || T == Int32 || \
                 T == Int8 || T == UInt16 || T == UInt32 || T == UInt64 || T == UInt8 %}
      {% raise "Bad dtype: #{T}. #{T} is not supported for Char Arrays" %}
    {% end %}
  end

  # Creates a string representation of a `Tensor`.  The implementation
  # of this is a bit of a mess, but I am happy with the results currently,
  # it could however be cleaned up to handle long floating point values
  # more precisely.
  def to_s(io)
    printer = ToString::BasePrinter.new(self, io, "Tensor")
    printer.print
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

  # Elementwise addition of a Tensor to another equally sized Tensor
  #
  # ```
  # f1 = Tensor.new [1.0, 2.0, 3.0]
  # f2 = Tensor.new [2.0, 4.0, 6.0]
  # f1 + f2 # => [3.0, 6.0, 9.0]
  # ```
  def +(other)
    BMath.add(self, other)
  end

  # Elementwise subtraction of a Tensor to another equally sized Tensor
  #
  # ```
  # f1 = Tensor.new [1.0, 2.0, 3.0]
  # f2 = Tensor.new [2.0, 4.0, 6.0]
  # f1 - f2 # => [-1.0, -2.0, -3.0]
  # ```
  def -(other)
    BMath.subtract(self, other)
  end

  # Elementwise multiplication of a Tensor to another equally sized Tensor
  #
  # ```
  # f1 = Tensor.new [1.0, 2.0, 3.0]
  # f2 = Tensor.new [2.0, 4.0, 6.0]
  # f1 * f2 # => [3.0, 8.0, 18.0]
  # ```
  def *(other)
    BMath.multiply(self, other)
  end

  # Elementwise division of a Tensor to another equally sized Tensor
  #
  # ```
  # f1 = Tensor.new [1.0, 2.0, 3.0]
  # f2 = Tensor.new [2.0, 4.0, 6.0]
  # f1 / f2 # => [0.5, 0.5, 0.5]
  # ```
  def /(other)
    BMath.divide(self, other)
  end

  def &(other)
    Binary.bitwise_and(self, other)
  end

  def |(other)
    Binary.bitwise_or(self, other)
  end

  def ^(other)
    Binary.bitwise_xor(self, other)
  end

  def <<(other)
    Binary.left_shift(self, other)
  end

  def >>(other)
    Binary.right_shift(self, other)
  end
end

include Bottle::Statistics

t = Bottle::Tensor.new([3, 2, 2]) { |i| i }
puts t << t
