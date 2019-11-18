require "./api"

module Bottle
  extend self
  VERSION = "0.2.2"
end

include Bottle

puts B.zeros([3, 4])
puts B.ones([2, 3, 4], dtype: UInt8)
puts B.empty([2, 3])
