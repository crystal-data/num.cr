require "../core/ndtensor"

module Bottle::Internal::Comparison
  extend self

  def allclose(a : Tensor(U), b : Tensor(U), rtol=1e-5, atol=1e-8) forall U
    if a.shape != b.shape
      raise "Shape of arguments must match"
    end
    iter_a = a.flat_iter
    iter_b = b.flat_iter

    if (rtol > 0)
      iter_a.zip(iter_b) do |i, j|
        c = (i.value - j.value).abs > atol + rtol * j.value.abs
        return false unless !c
      end
    else
      iter_a.zip(iter_b) do |i, j|
        c = (i.value - j.value).abs > atol
        return false unless !c
      end
    end
    true
  end
end
