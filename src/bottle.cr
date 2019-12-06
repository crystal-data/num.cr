require "./api"

module Bottle
  extend self
  VERSION = "0.2.5"
end

include Bottle

puts B.leslie([0.1, 2.0, 1.0, 0.1], [0.2, 0.8, 0.7])
