require "../src/bottle"
require "benchmark"
require "complex"

def test_fftw(n)
  t = Bottle::Tensor.random(0.0...1.0, [n, n])
  tc = Bottle::Tensor.new(n, n) { |i, j| Complex.new(i, j) }
  ift = Bottle::B.rfft(t)
  ift2 = Bottle::B.rfft2(t)

  Benchmark.ips do |bench|
    bench.report("FFT #{n}x#{n}") { Bottle::B.fft(tc) }
    bench.report("RFFT #{n}x#{n}") { Bottle::B.rfft(t) }
    bench.report("IRFFT #{n}x#{n}") { Bottle::B.irfft(ift) }
    bench.report("RFFT2 #{n}x#{n}") { Bottle::B.rfft2(t) }
    bench.report("IRFFT2 #{n}x#{n}") { Bottle::B.irfft2(ift2) }
    bench.report("RFFTN #{n}x#{n}") { Bottle::B.rfftn(t) }
  end
end

test_fftw 10
test_fftw 100
test_fftw 500
test_fftw 1000
test_fftw 2000
