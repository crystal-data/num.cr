require "./api"
require "benchmark"

module Bottle
  extend self
  VERSION = "0.2.1"
end

include Bottle

t = Tensor.new([4, 2, 2]) { |i| i * 1.0 }

def dot(a : Tensor(U), b : Tensor(U)) forall U
  if a.shape[...-2] != b.shape[...-2]
    raise "Shapes not aligned"
  end

  newshape = a.shape[...-2].dup
  newshape += [a.shape[-2], b.shape[-2]]

  dest = Tensor(U).new(newshape)

  a.matrix_iter.zip(b.matrix_iter, dest.matrix_iter) do |i, j, k|
    B.matmul(i, j, dest: k)
  end
  dest
end
