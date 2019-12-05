require "../tensor/tensor"
require "../libs/fftw"
require "../libs/cblas"
require "complex"

module Bottle::FFT
  def upscale_and_shape(t : Tensor, ptrsize : Int32)
    t = t.astype(Float64)
    ns = t.shape.dup
    ns[-1] = ns[-1] // 2 + 1
    cpl = Tensor(Complex).new(ns)
    cplu = cpl.to_unsafe
    inptr = Pointer(Float64).malloc(ptrsize)
    {t, ns, cpl, cplu, inptr}
  end

  def rfft(tensor : Tensor, flags = 0_u64)
    tensor, _, cpl, cplu, inptr = upscale_and_shape(tensor, tensor.shape[-1])
    plan = LibFFTW.fftw_plan_dft_r2c_1d(tensor.shape[-1], inptr, cplu, flags)
    case tensor.ndims
    when 1
      tensor.buffer.copy_to(inptr, tensor.shape[-1])
      LibFFTW.fftw_execute(plan)
    else
      iterations = tensor.shape[...-1].reduce { |i, j| i * j }
      s0 = tensor.shape[-1]
      buff = tensor.buffer
      iterations.times do |_|
        LibFFTW.fftw_execute_dft_r2c(plan, buff, cplu)
        buff += s0
        cplu += cpl.strides[-1] * cpl.shape[-1]
      end
    end
    LibFFTW.fftw_destroy_plan(plan)
    cpl
  end

  def fft(tensor : Tensor, flags = 0_u64)
    rfft(tensor, flags)
  end

  def fft(tensor : Tensor(Complex), flags = 0_u64)
    cpl = Tensor(Complex).new(tensor.shape)
    cplu = cpl.to_unsafe
    inptr = Pointer(LibCblas::ComplexDouble).malloc(tensor.shape[-1])
    plan = LibFFTW.fftw_plan_dft_1d(tensor.shape[-1], inptr, cplu, flags)

    case tensor.ndims
    when 1
      tensor.to_unsafe.copy_to(inptr, tensor.shape[-1])
      LibFFTW.fftw_execute(plan)
    else
      iterations = tensor.shape[...-1].reduce { |i, j| i * j }
      s0 = tensor.shape[-1]
      buff = tensor.to_unsafe
      iterations.times do |_|
        LibFFTW.fftw_execute_dft(plan, buff, cplu)
        buff += s0
        cplu += cpl.strides[-1] * cpl.shape[-1]
      end
    end
    LibFFTW.fftw_destroy_plan(plan)
    cpl
  end

  def rfft2(tensor : Tensor, flags = 0_u64)
    m, n = tensor.shape[-2...]
    tensor, _, cpl, cplu, inptr = upscale_and_shape(tensor, m * n)
    plan = LibFFTW.fftw_plan_dft_r2c_2d(m, n, inptr, cplu, flags)

    case tensor.ndims
    when 2
      tensor.buffer.copy_to(inptr, m * n)
      LibFFTW.fftw_execute(plan)
    else
      iterations = tensor.shape[...-2].reduce { |i, j| i * j }
      buff = tensor.buffer
      iterations.times do |_|
        LibFFTW.fftw_execute_dft_r2c(plan, buff, cplu)
        buff += m * n
        cplu += cpl.shape[-2] * cpl.strides[-2]
      end
    end
    cpl
  end

  def rfftn(tensor : Tensor)
    tensor, _, cpl, cplu, inptr = upscale_and_shape(tensor, tensor.size)
    plan = LibFFTW.fftw_plan_dft_r2c(
      tensor.ndims, tensor.shape.to_unsafe, inptr, cplu, UInt64.new(0))
    tensor.buffer.copy_to(inptr, tensor.size)
    LibFFTW.fftw_execute(plan)
    LibFFTW.fftw_destroy_plan(plan)
    cpl
  end
end
