require "spec"
require "../src/api"

module Num::Testing
  extend self

  def tensor_equal(a : Tensor(U, CPU(U)), b : Tensor(V, CPU(V))) forall U, V
    return false unless a.shape == b.shape
    a.zip(b) do |i, j|
      return false unless (i - j).abs < 1e-6
    end
    true
  end
end
