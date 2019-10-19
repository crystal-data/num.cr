require "./vector/*"
require "./matrix/*"
require "./core/bottle/*"
require "benchmark"

module Bottle
  include Bottle::Core
  extend self
  VERSION = "0.1.1"
end

def naive(arr : Array, b : Array)
  arr.map_with_index { |e, i| e +  b[i]}
end

a = (0...10000000).map { |e| 3.0 }
v = Vector.new a

Benchmark.ips do |b|
  b.report("crystal") { naive(a, a) }
  b.report("bottle") { v + v}
end
