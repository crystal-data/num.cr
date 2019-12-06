require "../tensor/tensor"
require "../libs/fftw"
require "../libs/cblas"
require "complex"

module Bottle::FFT
  def upscale_and_shape(t : Tensor, ptrsize : Int32, n : Int32? = nil)
    t = t.astype(Float64)
    ns = t.shape.dup
    if n.nil?
      ns[-1] = ns[-1] // 2 + 1
    else
      ns[-1] = n.as(Int32)
    end
    cpl = Tensor(Complex).new(ns)
    cplu = cpl.to_unsafe
    inptr = Pointer(Float64).malloc(ptrsize)
    {t, ns, cpl, cplu, inptr}
  end

  # Compute the one-dimensional discrete Fourier Transform for real input.
  #
  # This function computes the one-dimensional n-point discrete Fourier
  # Transform (DFT) of a real-valued array by means of an efficient
  # algorithm called the Fast Fourier Transform (FFT).
  #
  # For N-Dimensional tensors, this computes the one dimension fourier
  # transform along the final axis.  PR's welcome to extend this behavior
  # to any axis of the tensor.
  def rfft(tensor : Tensor, n : Int32? = nil, flags = 0_u64)
    # Casts the input to floating point and ensures that the data
    # is contiguous, as well as computes the output shape of the
    # resulting Complex tensor.
    tensor, _, cpl, cplu, inptr = upscale_and_shape(tensor, tensor.shape[-1], n)

    # Allocate a single plan, which will be executed on each appropriately
    # sized input of the tensor.
    plan = LibFFTW.fftw_plan_dft_r2c_1d(tensor.shape[-1], inptr, cplu, flags)

    # If the tensor is one dimensional, the calculation only needs to run once,
    # this could be done through the iteration method, but this saves time.
    case tensor.ndims
    when 1
      tensor.buffer.copy_to(inptr, tensor.shape[-1])
      LibFFTW.fftw_execute(plan)
    else
      # Calculate and permute along the last axis.  To extend this to higher
      # dimensions, changes will need to be made because there is no
      # guarantee the memory layout will be contiguous along any other
      # axis, in fact, it most likely will not be.
      iterations = tensor.shape[...-1].reduce { |i, j| i * j }
      s0 = tensor.shape[-1]
      buff = tensor.buffer
      iterations.times do |_|
        # Execute the plan once on each permutation of the axis,
        # as well as increment the pointers passed as arguments.
        LibFFTW.fftw_execute_dft_r2c(plan, buff, cplu)
        buff += s0
        cplu += cpl.strides[-1] * cpl.shape[-1]
      end
    end
    # Cleanup, complex tensor now contains the proper output.
    LibFFTW.fftw_destroy_plan(plan)
    cpl
  end

  def irfft(tensor : Tensor(Complex), n : Int32? = nil, flags = 0_u64)
    tensor = tensor.dup unless tensor.flags.contiguous?
    outshape = tensor.shape.dup
    dim = n.nil? ? tensor.shape[-1] : n.as(Int32)
    outshape[-1] = dim
    dbl = Tensor(Float64).new(outshape)
    dblu = dbl.to_unsafe
    inptr = Pointer(LibCblas::ComplexDouble).malloc(dim)
    plan = LibFFTW.fftw_plan_dft_c2r_1d(dim, inptr, dblu, flags)

    case tensor.ndims
    when 1
      tensor.to_unsafe.copy_to(inptr, dim)
      LibFFTW.fftw_execute(plan)
    else
      iterations = tensor.shape[...-1].reduce { |i, j| i * j }
      s0 = tensor.shape[-1]
      buff = tensor.to_unsafe
      iterations.times do |_|
        LibFFTW.fftw_execute_dft_c2r(plan, buff, dblu)
        buff += s0
        dblu += dbl.strides[-1] * dbl.shape[-1]
      end
    end
    dbl/dim
  end

  # Compute the one-dimensional discrete Fourier Transform.
  #
  # This function computes the one-dimensional n-point discrete Fourier
  # Transform (DFT) with the efficient Fast Fourier Transform (FFT)
  # algorithm [CT].
  #
  # If real input is passed, cast to float and return the real fourier
  # transform.
  def fft(tensor : Tensor, n : Int32? = nil, flags = 0_u64)
    rfft(tensor, n, flags)
  end

  # Compute the one-dimensional discrete Fourier Transform.
  #
  # This function computes the one-dimensional n-point discrete Fourier
  # Transform (DFT) with the efficient Fast Fourier Transform (FFT)
  # algorithm [CT].
  #
  # For N-Dimensional tensors, this computes the one dimension fourier
  # transform along the final axis.  PR's welcome to extend this behavior
  # to any axis of the tensor.
  def fft(tensor : Tensor(Complex), n : Int32? = nil, flags = 0_u64)
    # Creates a complex output tensor, as well as allocates
    # an appropriately sized output based on the passed
    # dimensions.
    outdim = n.nil? ? tensor.shape[-1] : n.as(Int32)
    outshape = tensor.shape.dup
    outshape[-1] = outdim
    cpl = Tensor(Complex).new(outshape)
    cplu = cpl.to_unsafe

    inptr = Pointer(LibCblas::ComplexDouble).malloc(outshape[-1])
    plan = LibFFTW.fftw_plan_dft_1d(outshape[-1], inptr, cplu, flags)

    # If the tensor is one dimensional, the calculation only needs to run once,
    # this could be done through the iteration method, but this saves time.
    case tensor.ndims
    when 1
      tensor.to_unsafe.copy_to(inptr, outshape[-1])
      LibFFTW.fftw_execute(plan)
    else
      # Calculate and permute along the last axis.  To extend this to higher
      # dimensions, changes will need to be made because there is no
      # guarantee the memory layout will be contiguous along any other
      # axis, in fact, it most likely will not be.
      iterations = tensor.shape[...-1].reduce { |i, j| i * j }
      s0 = tensor.shape[-1]
      buff = tensor.to_unsafe
      iterations.times do |_|
        # Execute the plan once on each permutation of the axis,
        # as well as increment the pointers passed as arguments.
        LibFFTW.fftw_execute_dft(plan, buff, cplu)
        buff += s0
        cplu += cpl.strides[-1] * cpl.shape[-1]
      end
    end
    # Cleanup and return the appropriately sized complex tensor
    LibFFTW.fftw_destroy_plan(plan)
    cpl
  end

  def rfft2(tensor : Tensor, s : Tuple(Int32, Int32)? = nil, flags = 0_u64)
    if s.nil?
      m, n = tensor.shape[-2...]
      tensor = tensor.astype(Float64)
    else
      m, n = s
      dim = [...] * (tensor.ndims - 2)
      dim += [...m, ...n]
      tensor = tensor.slice(dim).astype(Float64)
    end
    ns = tensor.shape.dup
    ns[-2...] = [m, s.nil? ? n // 2 + 1 : n]
    cpl = Tensor(Complex).new(ns)
    cplu = cpl.to_unsafe
    inptr = Pointer(Float64).malloc(m * n)
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

  def irfft2(tensor : Tensor(Complex), s : Tuple(Int32, Int32)? = nil, flags = 0_u64)
    if s.nil?
      m, n = tensor.shape[-2...]
    else
      m, n = s
    end
    ns = tensor.shape.dup
    ns[-2...] = [m, n]
    dbl = Tensor(Float64).new(ns)
    dblu = dbl.to_unsafe
    inptr = Pointer(LibCblas::ComplexDouble).malloc(m * n)
    plan = LibFFTW.fftw_plan_dft_c2r_2d(m, n, inptr, dblu, flags)

    case tensor.ndims
    when 2
      tensor.to_unsafe.copy_to(inptr, m * n)
      LibFFTW.fftw_execute(plan)
    else
      iterations = tensor.shape[...-2].reduce { |i, j| i * j }
      buff = tensor.to_unsafe
      iterations.times do |_|
        LibFFTW.fftw_execute_dft_c2r(plan, buff, dblu)
        buff += tensor.shape[-2] * tensor.strides[-2]
        dblu += m * n
      end
    end
    dbl / (m * n)
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
