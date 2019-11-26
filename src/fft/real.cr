require "../tensor/tensor"
require "../libs/fftw"
require "../libs/cblas"
require "complex"

module Bottle::FFT
  def fftn(t : Tensor)
    t = t.astype(Float64)
    outshape = t.shape.dup
    if t.ndims == 1
      outshape[0] = outshape[0] // 2 + 1
    else
      outshape[-1] = outshape[-1] // outshape[-2] + 1
    end
    inptr = Pointer(Float64).malloc(t.size)
    outptr = Tensor(Complex).new(outshape)
    plan = LibFFTW.fftw_plan_dft_r2c(t.ndims, t.shape.to_unsafe, inptr, outptr.to_unsafe, UInt64.new(0))
    t.buffer.copy_to(inptr, t.size)
    LibFFTW.fftw_execute(plan)
    LibFFTW.fftw_destroy_plan(plan)
    outptr
  end
end
