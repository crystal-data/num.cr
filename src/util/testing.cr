require "../base/base"

module Bottle::Internal::Comparison
  extend self

  # Asserts that two equally shaped `Tensor`s are equal within a provided
  # tolerance.  Useful for floating point comparison where direct equality might
  # not work
  #
  # ```
  # t = Tensor.new([2, 2, 3]) { |i| i * 1.0 }
  # tf = t + 0.00000000001
  # allclose(t, tf) # => true
  # ```
  def allclose(a : BaseArray(U), b : BaseArray(U), rtol = 1e-5, atol = 1e-8) forall U
    if a.shape != b.shape
      raise "Shape of arguments must match"
    end
    iter_a = a.flat_iter
    iter_b = b.flat_iter

    if (rtol > 0)
      iter_a.zip(iter_b) do |i, j|
        c = !((i.value - j.value).abs > atol + rtol * j.value.abs)
        return false unless c
      end
    else
      iter_a.zip(iter_b) do |i, j|
        c = (i.value - j.value).abs > atol
        return false unless c
      end
    end
    true
  end
end

module Bottle::Internal::Testing
  extend self

  def assert_compatible_shape(a : Array(Int32), b : Int32)
    toraise = false
    if a.size == 0
      if b != 0
        toraise = true
      end
    else
      sz = a.reduce { |i, j| i * j }
      toraise = true unless b == sz
    end
    if toraise
      raise Exceptions::ShapeError.new("An array of size #{b} cannot go
        into a Tensor of shape #{a}")
    end
  end
end
