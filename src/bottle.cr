require "./api"

module Bottle
  extend self
  VERSION = "0.2.5"
end

include Bottle
t = Tensor.new([2, 2]) { |i| i }
puts t
