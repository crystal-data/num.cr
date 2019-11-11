require "./api"

module Bottle
  extend self
  VERSION = "0.2.2"
end

include Bottle

t = Tensor.new([2, 2]) { |i| i * 1.0 }
puts B.matmul(t, t)
