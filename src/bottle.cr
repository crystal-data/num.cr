require "./api"
require "benchmark"

module Bottle
  extend self
  VERSION = "0.2.0"
end

include Bottle

t = Tensor.new([2, 2, 3]) { |i| i * 1.0 }

puts B.inv(t[0, ..., ...2])
