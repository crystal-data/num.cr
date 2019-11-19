require "./api"

module Bottle
  extend self
  VERSION = "0.2.2"
end

include Bottle

x = B.arange(10)
puts x[2...5]
puts x[...-7]

y = B.arange(35).reshape([5, 7])
puts y[1..., 2...4]
