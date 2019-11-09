require "./api"

module Bottle
  extend self
  VERSION = "0.2.2"
end

include Bottle

t = Tensor.from_array([2, 2], [1, 2, 3, 4])
t1 = Tensor.from_array([2, 2], [2, 3, 4, 5])

puts B.column_stack([t, t1])
