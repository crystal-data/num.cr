require "./api"

module Bottle
  extend self
  VERSION = "0.2.1"
end

include Bottle

t = Tensor.new([3, 2, 2]) { |i| i }
t[1] = 5
puts t
