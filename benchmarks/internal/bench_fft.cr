require "../../src/num"
require "benchmark"
require "complex"

def test_fftw(n)
  t = Tensor.random(0.0...1.0, [n, n])
  tc = Tensor.new(n, n) { |i, j| Complex.new(i, j) }
  ift = Num.rfft(t)
  ift2 = Num.rfft2(t)

  Benchmark.ips do |bench|
    bench.report("FFT #{n}x#{n}") { Num.fft(tc) }
    bench.report("RFFT #{n}x#{n}") { Num.rfft(t) }
    bench.report("IRFFT #{n}x#{n}") { Num.irfft(ift) }
    bench.report("RFFT2 #{n}x#{n}") { Num.rfft2(t) }
    bench.report("IRFFT2 #{n}x#{n}") { Num.irfft2(ift2) }
    bench.report("RFFTN #{n}x#{n}") { Num.rfftn(t) }
  end
end

test_fftw 10
test_fftw 100
test_fftw 500
test_fftw 1000
test_fftw 2000
