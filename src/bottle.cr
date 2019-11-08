require "./api"

module Bottle
  extend self
  VERSION = "0.2.1"
end

include Bottle

t = Tensor.from_array([4], [1, 2, 3, 5])
puts B.vander(t, 3)
