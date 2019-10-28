require "./api"

module Bottle
  extend self
  VERSION = "0.2.0"
end

include Bottle

t = Tensor.new [1.0, 2.0, 3.0, 4.0, 5.0]
m = Matrix.new [[1.0, 2.0], [3.0, 4.0]]

puts B.dot(t, t)

puts B.matmul(m, m)

puts B.inv(m)

puts B.norm(t)
