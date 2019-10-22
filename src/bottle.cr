require "./core/flask"
require "./core/jug"
require "benchmark"

module Bottle
  extend self
  VERSION = "0.1.1"
end

include Bottle

j = Jug.new [[1.8, 2.88, 2.05, -0.89], [5.25, -2.95, -0.95, -3.80], [1.58, -2.69, -2.90, -1.04], [-1.11, -0.66, -0.59, 0.8]]
puts LinAlg.lu(j)
