require "./api"

module Bottle
  extend self
  VERSION = "0.2.2"
end

include Bottle

t = Tensor.new([3, 2, 2]) { |i| i }
puts t[..., ..., 1...].flags
