require "benchmark"
require "../src/num"

include Num
r100 = (0...100).to_a
l = [Num.arange(1000), Num.arange(1000)]
l10xl10 = Num.ones([10, 10])

# Benchmark.ips do |bench|
#   bench.report("time_tensor_empty") { Tensor(Int32).new([] of Int32) }
#   bench.report("time_tensor_scalar") { Tensor.new(1) }
#   bench.report("time_tensor_1el") { Tensor.from_array [1] }
#   bench.report("time_tensor_100") { Tensor.from_array r100 }
#   bench.report("time_vstack") { N.vstack(l) }
#   bench.report("time_dstack") { N.dstack(l) }
#   bench.report("time_arange_100") { N.arange(100) }
#   bench.report("time_zeros_100") { N.zeros([100]) }
#   bench.report("time_ones_100") { N.ones([100]) }
#   bench.report("time_empty_100") { N.empty([100]) }
#   bench.report("time_eye_100") { N.eye(100) }
#   bench.report("time_identity_100") { N.identity(100) }
#   bench.report("time_eye_3000") { N.eye(3000) }
#   bench.report("time_identity_3000") { N.identity(3000) }
#   bench.report("time_triu_10x10") { N.triu(l10xl10) }
#   bench.report("time_tril_10x10") { N.tril(l10xl10) }
#   bench.report("time_kron_10x10") { N.kron(l10xl10, l10xl10) }
# end

amid = Num.ones([50000])
bmid = Num.ones([50000])
alarge = Num.ones([1000000])
blarge = Num.ones([1000000])

Benchmark.ips do |bench|
  bench.report("time_mid") { amid * 2 + bmid }
  bench.report("time mid map") { amid.map2(bmid) { |i, j| i * 2 + j } }
  bench.report("time_mid2") { amid + bmid - 2 }
  bench.report("time mid2 map") { amid.map2(bmid) { |i, j| i + j - 2} }
  bench.report("time_large") { alarge * 2 + blarge }
  bench.report("time large map") { alarge.map2(blarge) { |i, j| i * 2 + j } }
  bench.report("time_large2") { alarge + blarge - 2 }
  bench.report("time large map") { alarge.map2(blarge) { |i, j| i + j - 2 } }
end
