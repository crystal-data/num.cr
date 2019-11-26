require "benchmark"
require "../src/bottle"

include Bottle
small = B.zeros([10, 10])
medium = B.zeros([500, 500])
large = B.zeros([3000, 3000])
