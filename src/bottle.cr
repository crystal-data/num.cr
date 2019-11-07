require "./api"

module Bottle
  extend self
  VERSION = "0.2.1"
end

include Bottle

t = Tensor.new([2, 2, 3]) { |i| i }
view = t[..., 1]
puts view.flags
