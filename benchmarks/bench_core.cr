require "benchmark"
require "../src/bottle"

include Bottle
r100 = (0...100).to_a
l = [B.arange(1000), B.arange(1000)]
l10xl10 = B.ones([10, 10])

Benchmark.ips do |bench|
  bench.report("time_tensor_empty") { Tensor(Int32).new([] of Int32) }
  bench.report("time_tensor_scalar") { Tensor.new(1) }
  bench.report("time_tensor_1el") { Tensor.from_array [1] }
  bench.report("time_tensor_100") { Tensor.from_array r100 }
  bench.report("time_vstack") { B.vstack(l) }
  bench.report("time_dstack") { B.dstack(l) }
  bench.report("time_arange_100") { B.arange(100) }
  bench.report("time_zeros_100") { B.zeros([100]) }
  bench.report("time_ones_100") { B.ones([100]) }
  bench.report("time_empty_100") { B.empty([100]) }
  bench.report("time_eye_100") { B.eye(100) }
  bench.report("time_identity_100") { B.identity(100) }
  bench.report("time_eye_3000") { B.eye(3000) }
  bench.report("time_identity_3000") { B.identity(3000) }
  bench.report("time_triu_10x10") { B.triu(l10xl10) }
  bench.report("time_tril_10x10") { B.tril(l10xl10) }
end

amid = B.ones([50000])
bmid = B.ones([50000])
alarge = B.ones([1000000])
blarge = B.ones([1000000])

Benchmark.ips do |bench|
  bench.report("time_mid") { amid * 2 + bmid }
  bench.report("time_mid2") { amid + bmid - 2 }
  bench.report("time_large") { alarge * 2 + blarge }
  bench.report("time_large2") { alarge + blarge - 2 }
end
