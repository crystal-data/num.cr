require "./api"
require "benchmark"

module Bottle
  extend self
  VERSION = "0.2.5"
end

include Bottle

t = Tensor.new([3, 2, 2]) { |i| i * 1.0 }
t = t.astype(Float64)

puts t.inv
