require "./api"

module Bottle
  extend self
  VERSION = "0.2.2"
end

include Bottle

t = Tensor.new([2, 2, 3]) { |i| i }
puts t % 2 == 0
