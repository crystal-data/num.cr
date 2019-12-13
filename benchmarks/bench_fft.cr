require "../src/num"
require "benchmark"
require "complex"

def test_fftw(n)
  t = Num::Tensor.random(0.0...1.0, [n, n])
  tc = Num::Tensor.new(n, n) { |i, j| Complex.new(i, j) }
  ift = Num::N.rfft(t)
  ift2 = Num::N.rfft2(t)

  Benchmark.ips do |bench|
    bench.report("FFT #{n}x#{n}") { Num::N.fft(tc) }
    bench.report("RFFT #{n}x#{n}") { Num::N.rfft(t) }
    bench.report("IRFFT #{n}x#{n}") { Num::N.irfft(ift) }
    bench.report("RFFT2 #{n}x#{n}") { Num::N.rfft2(t) }
    bench.report("IRFFT2 #{n}x#{n}") { Num::N.irfft2(ift2) }
    bench.report("RFFTN #{n}x#{n}") { Num::N.rfftn(t) }
  end
end

test_fftw 10
test_fftw 100
test_fftw 500
test_fftw 1000
test_fftw 2000
