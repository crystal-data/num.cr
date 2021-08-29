require "spec"
require "../src/api"

module Num::Testing
  extend self

  def tensor_equal(a : Tensor(U, CPU(U)), b : Tensor(V, CPU(V)), tolerance = 1e-6) forall U, V
    return false unless a.shape == b.shape
    a.zip(b) do |i, j|
      return false unless (i - j).abs <= tolerance
    end
    true
  end
end
