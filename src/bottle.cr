require "./api"
require "benchmark"

module Bottle
  extend self
  VERSION = "0.2.0"
end

include Bottle

t = Tensor.new([2, 2, 3]) { |i| Float64.new(i) }
puts B.mean(t, 1)
