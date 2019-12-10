require "../base"

@[Flags]
enum Num::Internal::ArrayFlags
  # Contiguous really means C-style contiguious.  The
  # contiguous part means that there are no 'skipped
  # elements'.  That is, that a flat_iter over the array will
  # touch every location in memory from the location of the
  # first element to that of the last element.  The C-style
  # part means that the data is laid out such that the last index
  # is the fastest varying as one scans though the array's
  # memory.
  Contiguous
  # Fortran really means Fortran-style contiguious.  The
  # contiguous part means that there are no 'skipped
  # elements'.  That is, that a flat_iter over the array will
  # touch every location in memory from the location of the
  # first element to that of the last element.  The Fortran-style
  # part means that the data is laid out such that the first index
  # is the fastest varying as one scans though the array's
  # memory.
  Fortran
  # OwnData indicates if this array is the owner of the data
  # pointed to by its .ptr property.  If not then this is a
  # view onto some other array's data.
  OwnData
  # Some views into arrays are created using stride tricks
  # and aren't safe to write to, since many locations may
  # be sharing the same memory
  Write
end

module Num::Internal::FlagChecks
  extend self

  # Asserts if a `Tensor` is fortran contiguous, otherwise known
  # as stored in column major order.  This is not the default
  # layout for `Tensor`'s, but can provide performance benefits
  # when passing to LaPACK routines since otherwise the
  # `Tensor` must be transposed in memory.
  def is_fortran_contiguous(ndims, shape, strides)
    # Empty Tensors are always both c-contig and f-contig
    return true unless ndims != 0

    # one-dimensional `Tensors` can be both c and f contiguous,
    # but not for multi-strided arrays
    if ndims == 1
      return shape[0] == 1 || strides[0] == 1
    end

    # Otherwise, have to compute based on a fixed
    # stride offset
    sd = 1
    ndims.times do |i|
      dim = shape[i]
      return true unless dim != 0
      return false unless strides[i] == sd
      sd *= dim
    end
    true
  end

  # Asserts if a `Tensor` is c contiguous, otherwise known
  # as stored in row major order.  This is the default memory
  # storage for NDArray
  def is_contiguous(ndims, shape, strides)
    # Empty Tensors are always both c-contig and f-contig
    return true unless ndims != 0

    # one-dimensional `Tensors` can be both c and f contiguous,
    # but not for multi-strided arrays
    if ndims == 1
      return shape[0] == 1 || strides[0] == 1
    end

    # Otherwise, have to compute based on a fixed
    # stride offset
    sd = 1
    (ndims - 1).step(to: 0, by: -1) do |i|
      dim = shape[i]
      return true unless dim != 0
      return false unless strides[i] == sd
      sd *= dim
    end
    true
  end

  def update_flags(arr : BaseArray, flags, mask, ndims, write = true)
    if mask & ArrayFlags::Fortran
      if is_fortran_contiguous(arr.ndims, arr.shape, arr.strides)
        flags |= ArrayFlags::Fortran

        # mutually exclusive
        if ndims > 1
          flags &= ~ArrayFlags::Contiguous
        end
      else
        flags &= ~ArrayFlags::Fortran
      end
    end

    if mask & ArrayFlags::Contiguous
      if is_contiguous(arr.ndims, arr.shape, arr.strides)
        flags |= ArrayFlags::Contiguous

        # mutually exclusive
        if ndims > 1
          flags &= ~ArrayFlags::Fortran
        end
      else
        flags &= ~ArrayFlags::Contiguous
      end
    end

    if write
      flags |= ArrayFlags::Write
    end
    flags
  end
end
