require "./vector/*"
require "./matrix/*"
require "./core/bottle/*"

module Bottle
  include Bottle::Core
  extend self
  VERSION = "0.1.1"
end

m = Matrix.new [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
puts m.cumsum(1)
