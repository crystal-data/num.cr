require "./api"

module Num
  VERSION = "0.2.6"
end

a = Num.zeros([3, 3])
b = Num.ones([3, 3])
puts Num.concatenate([a, b], 1)
