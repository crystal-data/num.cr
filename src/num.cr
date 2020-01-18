require "./api"

module Num
  VERSION = "0.2.6"
end

t = Tensor.new([10]) { |i| i }
r = t.map2(t) { |i, j| i + j }
puts r
