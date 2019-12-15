require "./api"

module Num
  VERSION = "0.2.6"
end

include Num

t = N.arange(9).reshape(3, 3)
puts N.rot90(t)
