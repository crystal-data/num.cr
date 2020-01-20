require "../../src/num"
require "benchmark"

def test_iter(n)
  t = Num::Tensor.random(0.0...1.0, [n, n])

  Benchmark.ips do |bench|
    bench.report("split axis 0") { Num::N.split(t, n//2, 0) }
    bench.report("split axis 1") { Num::N.split(t, n//2, 1) }
    bench.report("hsplit") { Num::N.hsplit(t, n//2) }
    bench.report("vsplit") { Num::N.vsplit(t, n//2) }
    bench.report("repeat flat") { Num::N.repeat(t, 5) }
    bench.report("repeat axis 0") { Num::N.repeat(t, 5, 0) }
    bench.report("repeat axis 1") { Num::N.repeat(t, 5, 1) }
  end
end

test_iter 10
test_iter 100
test_iter 500
