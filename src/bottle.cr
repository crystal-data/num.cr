require "./api"
require "complex"
require "benchmark"

module Bottle
  extend self
  VERSION = "0.2.5"
end

include Bottle

a = Tensor.from_array [[1.8, 2.88, 2.05, -0.89], [5.25, -2.95, -0.95, -3.80], [1.58, -2.68, -2.9, -1.04], [-1.11, -0.66, -0.59, 0.8]]

puts B.det(a)
