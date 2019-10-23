require "./core/vector"

module Bottle
  extend self
  VERSION = "0.1.1"
end

include Bottle

v = Vector.new [1, 2, 3, 4]
puts B.median(v)
