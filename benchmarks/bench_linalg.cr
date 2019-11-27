require "../src/bottle"
require "benchmark"

puts "*******************LINEAR ALGEBRA***********************"

def test_linalg(n)
  t = Bottle::Tensor.random(0.0...1.0, [n, n])

  puts "*********N = #{n}*************"
  Benchmark.ips do |bench|
    bench.report("QR") { t.qr }
    bench.report("SVD") { t.svd }
    bench.report("INV") { t.inv }
    bench.report("EIGVALSH") { t.eigvalsh }
    bench.report("EIGVALS") { t.eigvals }
    bench.report("NORM") { t.norm }
    bench.report("DET") { t.det }
    bench.report("HESSENBERG") { t.hessenberg }
    bench.report("MATMUL") { t.matmul(t) }
  end
end

test_linalg 10
test_linalg 100
test_linalg 500
test_linalg 1000
test_linalg 2000

# *********N = 10*************
#         QR 123.28k (  8.11µs) (± 2.19%)  4.97kB/op   8.13× slower
#        SVD  25.09k ( 39.86µs) (± 1.22%)  7.48kB/op  39.95× slower
#        INV 281.80k (  3.55µs) (± 3.14%)  1.53kB/op   3.56× slower
#   EIGVALSH  63.27k ( 15.81µs) (± 0.65%)  3.26kB/op  15.84× slower
#    EIGVALS  52.47k ( 19.06µs) (± 1.86%)   5.7kB/op  19.10× slower
#       NORM 560.35k (  1.78µs) (± 1.80%)  1.48kB/op   1.79× slower
#        DET 395.00k (  2.53µs) (± 3.11%)  1.64kB/op   2.54× slower
# HESSENBERG 111.07k (  9.00µs) (± 1.56%)  6.27kB/op   9.02× slower
#     MATMUL   1.00M (997.74ns) (± 1.89%)   1.5kB/op        fastest
# *********N = 100*************
#         QR   1.66k (602.37µs) (± 1.08%)   236kB/op   16.38× slower
#        SVD 425.30  (  2.35ms) (± 1.28%)   392kB/op   63.93× slower
#        INV   3.16k (316.43µs) (±11.52%)  78.7kB/op    8.60× slower
#   EIGVALSH   1.56k (642.49µs) (± 1.59%)   158kB/op   17.47× slower
#    EIGVALS 268.69  (  3.72ms) (± 0.56%)   242kB/op  101.19× slower
#       NORM   9.32k (107.25µs) (± 2.86%)  78.3kB/op    2.92× slower
#        DET   5.36k (186.42µs) (±25.93%)  78.9kB/op    5.07× slower
# HESSENBERG   1.40k (715.72µs) (± 0.95%)   314kB/op   19.46× slower
#     MATMUL  27.19k ( 36.78µs) (± 1.97%)  78.3kB/op         fastest
# *********N = 500*************
#         QR  68.12  ( 14.68ms) (± 2.89%)  5.73MB/op   5.57× slower
#        SVD  19.95  ( 50.13ms) (± 1.11%)   9.5MB/op  19.01× slower
#        INV 124.98  (  8.00ms) (±11.77%)  1.91MB/op   3.03× slower
#   EIGVALSH  73.95  ( 13.52ms) (± 1.37%)  3.81MB/op   5.13× slower
#    EIGVALS   6.74  (148.45ms) (± 0.78%)  5.74MB/op  56.30× slower
#       NORM 362.39  (  2.76ms) (± 3.90%)  1.91MB/op   1.05× slower
#        DET 219.40  (  4.56ms) (±11.26%)  1.91MB/op   1.73× slower
# HESSENBERG  43.45  ( 23.02ms) (± 1.78%)  7.64MB/op   8.73× slower
#     MATMUL 379.24  (  2.64ms) (± 8.09%)  1.91MB/op        fastest
# *********N = 1000*************
#         QR  10.20  ( 98.01ms) (± 3.76%)  22.9MB/op   7.07× slower
#        SVD   2.46  (406.45ms) (± 1.81%)  38.1MB/op  29.33× slower
#        INV  20.78  ( 48.12ms) (± 3.94%)  7.63MB/op   3.47× slower
#   EIGVALSH  13.36  ( 74.87ms) (± 7.76%)  15.3MB/op   5.40× slower
#    EIGVALS   1.22  (816.76ms) (± 1.46%)  22.9MB/op  58.93× slower
#       NORM  72.15  ( 13.86ms) (± 3.76%)  7.63MB/op        fastest
#        DET  41.34  ( 24.19ms) (± 7.60%)  7.63MB/op   1.75× slower
# HESSENBERG   5.66  (176.73ms) (± 2.18%)  30.5MB/op  12.75× slower
#     MATMUL  49.82  ( 20.07ms) (± 3.11%)  7.63MB/op   1.45× slower
# *********N = 2000*************
#         QR   1.24  (807.90ms) (± 0.45%)  91.6MB/op   9.62× slower
#        SVD 205.56m (  4.86s ) (± 1.50%)   153MB/op  57.90× slower
#        INV   2.89  (346.36ms) (± 1.77%)  30.5MB/op   4.12× slower
#   EIGVALSH   1.16  (864.09ms) (± 0.50%)  61.0MB/op  10.28× slower
#    EIGVALS 193.42m (  5.17s ) (± 7.95%)  91.6MB/op  61.53× slower
#       NORM  11.90  ( 84.02ms) (±15.23%)  30.5MB/op        fastest
#        DET   5.80  (172.28ms) (±20.76%)  30.5MB/op   2.05× slower
# HESSENBERG 318.37m (  3.14s ) (± 3.34%)   122MB/op  37.38× slower
#     MATMUL   5.67  (176.31ms) (± 5.78%)  30.5MB/op   2.10× slower
