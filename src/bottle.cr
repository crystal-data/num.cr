require "./core/vector"
require "./core/matrix"
require "./linalg/*"
require "./blas/*"
require "benchmark"

module Bottle
  extend self
  VERSION = "0.1.1"
end

include Bottle

n = 100000

arr = (0...n).map { |i| Float64.new(i) }
vec = Vector.new arr
sli = Slice.new(n) { |i| Float64.new(i) }

puts arr.sum
puts vec.sum
puts sli.sum

Benchmark.ips do |i|
  i.report("array sum") { arr.sum }
  i.report("vector sum") { vec.sum }
  i.report("slice sum") { sli.sum }
end
