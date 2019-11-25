require "../src/bottle"

def test_iter(n)
  t = Bottle::Tensor.random(0.0...1.0, [n, n])

  puts "*********N = #{n}*************"
  Benchmark.ips do |bench|
    bench.report("REDUCTION") { t.sum }
    bench.report("REDUCE ALONG AXIS") { t.sum(0) }
    bench.report("ELEMENTWISE") { t + t }
  end
end

test_iter 10
test_iter 100
test_iter 500
test_iter 1000
test_iter 2000

# *********N = 10*************
#         REDUCTION   8.15M (122.66ns) (± 4.61%)   16.0B/op        fastest
# REDUCE ALONG AXIS 322.73k (  3.10µs) (± 4.21%)  3.39kB/op  25.26× slower
#       ELEMENTWISE   1.59M (628.17ns) (± 4.47%)   1.2kB/op   5.12× slower
# *********N = 100*************
#         REDUCTION 110.68k (  9.03µs) (± 1.20%)   16.0B/op        fastest
# REDUCE ALONG AXIS  19.04k ( 52.52µs) (± 0.91%)  29.2kB/op   5.81× slower
#       ELEMENTWISE  28.07k ( 35.62µs) (± 5.62%)  78.3kB/op   3.94× slower
# *********N = 500*************
#         REDUCTION   4.47k (223.78µs) (± 0.77%)   16.0B/op        fastest
# REDUCE ALONG AXIS   1.37k (727.98µs) (± 1.34%)   142kB/op   3.25× slower
#       ELEMENTWISE   1.43k (698.81µs) (± 1.30%)  1.91MB/op   3.12× slower
# *********N = 1000*************
#         REDUCTION   1.12k (892.50µs) (± 1.88%)   16.0B/op        fastest
# REDUCE ALONG AXIS 359.43  (  2.78ms) (± 1.60%)   282kB/op   3.12× slower
#       ELEMENTWISE 338.38  (  2.96ms) (± 1.46%)  7.63MB/op   3.31× slower
# *********N = 2000*************
#         REDUCTION 278.11  (  3.60ms) (± 0.92%)   16.0B/op        fastest
# REDUCE ALONG AXIS  94.69  ( 10.56ms) (± 1.18%)   563kB/op   2.94× slower
#       ELEMENTWISE  83.43  ( 11.99ms) (± 1.35%)  30.5MB/op   3.33× slower
