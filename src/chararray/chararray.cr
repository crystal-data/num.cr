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
end
