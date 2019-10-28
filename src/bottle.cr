require "./api"

module Bottle
  extend self
  VERSION = "0.2.0"
end

include Bottle

t = Tensor.new [1.0, 2.0, 3.0, 4.0, 5.0]
m = Matrix.new [[1.0, 2.0], [3.0, 4.0]]

puts t[1...]

puts m[0]

puts m[..., 1]

puts m[1..., 1...]
