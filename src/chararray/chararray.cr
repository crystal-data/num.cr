require "../base/base"

class Bottle::CharArray(T) < Bottle::BaseArray(T)
  # Compile time checking of data types of a `Tensor` to ensure
  # mixing data types is not allowed, not are bad data types
  # allowed into the `Tensor`
  protected def check_type
    {% unless T == Char || T == String %}
      {% raise "Bad dtype: #{T}. #{T} is not supported for Char Arrays" %}
    {% end %}
  end

  getter basetype = CharArray

  # Creates a string representation of a `Tensor`.  The implementation
  # of this is a bit of a mess, but I am happy with the results currently,
  # it could however be cleaned up to handle long floating point values
  # more precisely.
  def to_s(io)
    maxlength = 0
    {% if T == String %}
      maxlength = max.size
    {% else %}
      maxlength = 1
    {% end %}
    printer = ToString::BasePrinter.new(self, io, "CharArray", maxlength)
    printer.print
  end

  def max
    mx = self[[0]]
    flat_iter.each do |i|
      s = i.value
      if s > mx
        mx = s
      end
    end
    mx
  end
end
