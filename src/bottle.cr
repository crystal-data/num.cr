require "./vector/*"
require "./matrix/*"
require "benchmark"

module Bottle
  VERSION = "0.1.0"
end


m = Matrix.new [[1, 2, 3], [4, 11, 6], [18, 2, 2]]
puts m.idxmax
