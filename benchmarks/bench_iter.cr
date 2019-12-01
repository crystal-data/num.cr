require "../src/bottle"
require "benchmark"

puts "*******************NDITERATION***********************"

def test_iter(n)
  t = Bottle::Tensor.random(0.0...1.0, [n, n])

  puts "*********N = #{n}*************"
  Benchmark.ips do |bench|
    bench.report("REDUCTION") { t.sum }
    bench.report("REDUCE ALONG AXIS") { t.sum(0) }
    bench.report("ELEMENTWISE") { t + t }
    bench.report("ACCUMULATE") { t.cumsum(0) }
    if n < 1000
      bench.report("OUTER") { t.bc?(2) + t }
    end
  end
end

test_iter 10
test_iter 100
test_iter 500
test_iter 1000
test_iter 2000

# *********N = 10*************
#         REDUCTION   8.23M (121.45ns) (± 2.57%)   16.0B/op        fastest
# REDUCE ALONG AXIS 288.17k (  3.47µs) (± 1.07%)   3.4kB/op  28.57× slower
#       ELEMENTWISE   1.25M (801.23ns) (± 1.66%)   1.2kB/op   6.60× slower
#        ACCUMULATE 253.70k (  3.94µs) (± 2.73%)  5.21kB/op  32.46× slower
#             OUTER 105.03k (  9.52µs) (± 0.92%)  8.78kB/op  78.39× slower
# *********N = 100*************
#         REDUCTION 109.95k (  9.10µs) (± 0.89%)   16.0B/op         fastest
# REDUCE ALONG AXIS  20.21k ( 49.49µs) (± 1.66%)  29.1kB/op    5.44× slower
#       ELEMENTWISE  34.92k ( 28.64µs) (± 0.97%)  78.3kB/op    3.15× slower
#        ACCUMULATE  15.62k ( 64.01µs) (± 1.27%)   184kB/op    7.04× slower
#             OUTER 138.93  (  7.20ms) (± 0.75%)  7.63MB/op  791.35× slower
# *********N = 500*************
#         REDUCTION   4.48k (223.01µs) (± 1.40%)   16.0B/op          fastest
# REDUCE ALONG AXIS   1.34k (745.44µs) (± 2.35%)   142kB/op     3.34× slower
#       ELEMENTWISE   1.43k (697.61µs) (± 7.29%)  1.91MB/op     3.13× slower
#        ACCUMULATE 811.82  (  1.23ms) (± 3.92%)  3.95MB/op     5.52× slower
#             OUTER 964.10m (  1.04s ) (±26.51%)  0.93GB/op  4651.03× slower
# *********N = 1000*************
#         REDUCTION   1.11k (899.44µs) (± 1.08%)   16.0B/op        fastest
# REDUCE ALONG AXIS 351.33  (  2.85ms) (± 2.36%)   282kB/op   3.16× slower
#       ELEMENTWISE 340.13  (  2.94ms) (± 5.37%)  7.63MB/op   3.27× slower
#        ACCUMULATE 178.39  (  5.61ms) (± 8.38%)  15.5MB/op   6.23× slower
# *********N = 2000*************
#         REDUCTION 274.34  (  3.65ms) (± 1.98%)   17.0B/op        fastest
# REDUCE ALONG AXIS  92.82  ( 10.77ms) (± 2.50%)   563kB/op   2.96× slower
#       ELEMENTWISE  83.87  ( 11.92ms) (± 2.15%)  30.5MB/op   3.27× slower
#        ACCUMULATE  48.40  ( 20.66ms) (± 5.07%)  61.5MB/op   5.67× slower
