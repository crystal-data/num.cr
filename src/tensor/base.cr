require "./flags"
require "complex"

class BaseArray(T)
  @buffer : Pointer(T)
  @shape : Array(Int32)
  @strides : Array(Int32)
  @ndims : Int32
  @flags : Bottle::Internal::TensorFlags
  @size : Int32

  protected def check_type
    {% unless T == Float32 || T == Float64 || T == Bool || T == Int8 || T == Int16 || \
                 T == Int32 || T == Int64 || T == UInt8 || T == UInt16 || T == UInt32 || \
                 T == UInt64 || T == String || T == Char || T == Complex %}
      {% raise "Bad dtype: #{T}. #{T} is not supported by Bottle" %}
    {% end %}
  end

  def initialize(shape : Array(Int32),
                 order : Bottle::Internal::TensorFlags = Bottle::Internal::TensorFlags::Contiguous,
                 ptr : Pointer(T)? = nil)
    check_type
    @ndims = shape.size

    # Empty NDArrays are both allowed, and they have
    # a shape of [0] and a stride of [1].  This
    # is inferred from an empty array being passed
    # to the constructor.
    if @ndims == 0
      @shape = [0]
      @strides = [1]

      # Otherwise, shape is directly copied from the
      # array that was passed to the constructor.
      # Strides will always share dimensionality
      # with the shape of an NDArray.
    else
      @shape = shape.clone
      @strides = [0] * shape.size
    end
    sz = 1

    # For empty arrays, since no elements are initialized
    # nothing special has to be done about the memory
    # allocation, but strides must be calculated differently.
    case order
    # For Fortran ordered arrays strides are calculated from
    # the beginning of the shape to the end, with strides
    # monotonically increasing.
    when Bottle::Internal::TensorFlags::Fortran
      @ndims.times do |i|
        @strides[i] = sz
        sz *= @shape[i]
      end
      # Otherwise, row major order is chosen and strides are
      # calculated from the reversed shape, monotonically
      # decreasing.
    else
      @ndims.times do |i|
        @strides[@ndims - i - 1] = sz
        sz *= @shape[@ndims - i - 1]
      end
    end

    @size = sz

    # Memory allocation for empty arrays is consistent
    # regardless of order, and this method will always
    # return an NDArray that owns its own data.
    @buffer = ptr.nil? ? Pointer(T).malloc(@size) : ptr
    @flags = order | Bottle::Internal::TensorFlags::OwnData
    update_flags(Bottle::Internal::TensorFlags::All)
  end

  # Asserts if a `Tensor` is fortran contiguous, otherwise known
  # as stored in column major order.  This is not the default
  # layout for `Tensor`'s, but can provide performance benefits
  # when passing to LaPACK routines since otherwise the
  # `Tensor` must be transposed in memory.
  def is_fortran_contiguous
    # Empty Tensors are always both c-contig and f-contig
    return true unless @ndims != 0

    # one-dimensional `Tensors` can be both c and f contiguous,
    # but not for multi-strided arrays
    if @ndims == 1
      return @shape[0] == 1 || @strides[0] == 1
    end

    # Otherwise, have to compute based on a fixed
    # stride offset
    sd = 1
    @ndims.times do |i|
      dim = @shape[i]
      return true unless dim != 0
      return false unless @strides[i] == sd
      sd *= dim
    end
    true
  end

  # Asserts if a `Tensor` is c contiguous, otherwise known
  # as stored in row major order.  This is the default memory
  # storage for NDArray
  def is_contiguous
    # Empty Tensors are always both c-contig and f-contig
    return true unless @ndims != 0

    # one-dimensional `Tensors` can be both c and f contiguous,
    # but not for multi-strided arrays
    if @ndims == 1
      return @shape[0] == 1 || @strides[0] == 1
    end

    # Otherwise, have to compute based on a fixed
    # stride offset
    sd = 1
    (@ndims - 1).step(to: 0, by: -1) do |i|
      dim = @shape[i]
      return true unless dim != 0
      return false unless @strides[i] == sd
      sd *= dim
    end
    true
  end

  # Updates a `Tensor`'s flags by determining its
  # memory layout.  Multidimension tensors cannot be
  # both c and f contiguous, but this needs to be checked.
  #
  # This method should really only be called by internal
  # methods, or once stride tricks are exposed.
  protected def update_flags(flagmask)
    if flagmask & Bottle::Internal::TensorFlags::Fortran
      if is_fortran_contiguous
        @flags |= Bottle::Internal::TensorFlags::Fortran

        # mutually exclusive
        if @ndims > 1
          @flags &= ~Bottle::Internal::TensorFlags::Contiguous
        end
      else
        @flags &= ~Bottle::Internal::TensorFlags::Fortran
      end
    end

    if flagmask & Bottle::Internal::TensorFlags::Contiguous
      if is_contiguous
        @flags |= Bottle::Internal::TensorFlags::Contiguous

        # mutually exclusive
        if @ndims > 1
          @flags &= ~Bottle::Internal::TensorFlags::Fortran
        end
      else
        @flags &= ~Bottle::Internal::TensorFlags::Contiguous
      end
    end
  end
end
