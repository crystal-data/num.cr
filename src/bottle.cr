require "./core/vector"
require "./api/distributions"
require "./api/vectorprint"

module Bottle
  extend self
  VERSION = "0.1.1"
end

include Bottle

v = B.vrange(4)

puts v
