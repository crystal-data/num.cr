require "./core/flask"
require "./core/jug"

module Bottle
  extend self
  VERSION = "0.1.1"
end

n = 10
stride = 2
slice = Slice.new(n) { |i| i }
f = Flask.new slice, n // stride, stride
result = f[1...]
puts result
