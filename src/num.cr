require "./api"

module Num
  VERSION = "0.2.6"
end

include Num

a = N.arange(6).reshape(3,2)
b = N.arange(12).reshape(4,3)

puts N.find_output_shape(["ab", "bc"], [[2, 3], [3, 4]], "ac")
