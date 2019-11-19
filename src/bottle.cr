require "./api"

module Bottle
  extend self
  VERSION = "0.2.2"
end

include Bottle

a = B.arange(12).reshape([3, 4])
b = a  # no copy of the tensors data is made

c = a.dup_view()

puts a.buffer == c.buffer
puts c.flags.own_data?

c = c.reshape([2, 6])
c[[0, 4]] = 12345
puts a

s = a[..., 1...3]
s[...] = 10

puts a

d = a.dup
puts d.buffer == a.buffer

d[[0, 0]] = 9999
puts a
