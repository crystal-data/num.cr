require "benchmark"
require "../../src/num"

def test_iter(n)
  t = Tensor.random(0.0...1.0, [n, n]).dup('F')

  Benchmark.ips do |bench|
    bench.report("Sort axis 0 #{n}x#{n}") { Num.sort(t, 0) }
    bench.report("Sort axis 1 #{n}x#{n}") { Num.sort(t, 1) }
  end
end

test_iter 10
test_iter 100
test_iter 500
test_iter 1000
test_iter 2000

# THere is SO much room for improvement of sorting, I did pretty much the
# brute force slice approach, and it's still only about 20% slower than numpy,
# which I consider a huge win.
#
#
# Tensor([ 3,  5,  6,  6,  7,  8,  9,  9,  9,  9])
# *********N = 10*************
# Sort axis 0  60.08k ( 16.65µs) (± 6.19%)  16.1kB/op   1.01× slower
# Sort axis 1  60.72k ( 16.47µs) (± 3.73%)  18.2kB/op        fastest
# *********N = 100*************
# Sort axis 0   1.17k (852.21µs) (± 1.90%)  564kB/op   1.09× slower
# Sort axis 1   1.28k (781.04µs) (± 2.13%)  676kB/op        fastest
# *********N = 500*************
# Sort axis 0  40.84  ( 24.49ms) (± 4.28%)  10.1MB/op   1.19× slower
# Sort axis 1  48.62  ( 20.57ms) (± 5.42%)  12.0MB/op        fastest
# *********N = 1000*************
# Sort axis 0   7.49  (133.54ms) (± 5.29%)  39.3MB/op   1.41× slower
# Sort axis 1  10.57  ( 94.65ms) (± 6.71%)  47.0MB/op        fastest
# *********N = 2000*************
# Sort axis 0   1.91  (522.20ms) (± 2.26%)  154MB/op   1.43× slower
# Sort axis 1   2.74  (364.64ms) (± 2.52%)  186MB/op        fastest
