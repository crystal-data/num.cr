crystal build matmul_num.cr --release -Dopenblas
nim compile -d:openmp -d:native -d:danger matmul_arraymancer.nim

crystal build elementwise_num.cr --release
nim compile -d:openmp -d:native -d:danger elementwise_arraymancer.nim
