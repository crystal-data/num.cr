require "../tensor/tensor"
require "../libs/fftw"

module Bottle::FFT
  def fft(t : Tensor(Float64))
    if !t.flags.contiguous?
      t = t.dup
    end

    outshape = t.shape.dup
    if t.ndims == 1
      outshape[0] = outshape[0] // 2 + 1
    else
      outshape[-1] = outshape[-1] // outshape[-2] + 1
    end

    inptr = Pointer(Float64).malloc(t.size)
    size = outshape.reduce { |i, j| i * j }
    outptr = Pointer(LibFFTW::FftwComplex).malloc(size)

    plan = LibFFTW.fftw_plan_dft_r2c(t.ndims, t.shape.to_unsafe, inptr, outptr, UInt64.new(0))
    t.buffer.copy_to(inptr, t.size)

    LibFFTW.fftw_execute(plan)

    ret = Tensor(Complex).new(outshape) do |i|
      real, imag = outptr[i]
      Complex.new(real, imag)
    end

    LibFFTW.fftw_destroy_plan(plan)
    ret
  end
end
