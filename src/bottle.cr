require "./api"
require "benchmark"

module Bottle
  extend self
  VERSION = "0.2.5"
end

include Bottle

a = Tensor.new([3, 2, 2]) { |i| i }

puts a
