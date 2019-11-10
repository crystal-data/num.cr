require "./flags"
require "./helpers"
require "./baseiter"
require "./print"

class Bottle::BaseArray(T)
  include Internal
  @buffer : Pointer(T)
  @shape : Array(Int32)
  @strides : Array(Int32)
  @ndims : Int32
  @flags : ArrayFlags
  @size : Int32
  @base : Pointer(T)? = nil

  getter shape
  getter strides
  getter ndims
  getter flags
  getter size

  def initialize(_shape : Array(Int32),
                 order : ArrayFlags = ArrayFlags::Contiguous,
                 ptr : Pointer(T)? = nil)
    @ndims = _shape.size

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
      @shape = _shape.clone
      @strides = [0] * _shape.size
    end
    sz = 1

    # For empty arrays, since no elements are initialized
    # nothing special has to be done about the memory
    # allocation, but strides must be calculated differently.
    case order
    # For Fortran ordered arrays strides are calculated from
    # the beginning of the shape to the end, with strides
    # monotonically increasing.
    when ArrayFlags::Fortran
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
    @flags = order | ArrayFlags::OwnData
    update_flags(ArrayFlags::All)
  end

  # Internal method to create tensors from low level libraries.
  # This does no validation on inputs and is very unsafe unless
  # called by the library.
  #
  # Should not be used by the external API.
  def initialize(@buffer, @shape, @strides, @flags, @base)
    check_type
    @ndims = @shape.size
    @size = @shape.reduce { |i, j| i * j }
    update_flags(ArrayFlags::All)
  end

  # Updates a `Tensor`'s flags by determining its
  # memory layout.  Multidimension tensors cannot be
  # both c and f contiguous, but this needs to be checked.
  #
  # This method should really only be called by internal
  # methods, or once stride tricks are exposed.
  protected def update_flags(flagmask)
    if flagmask & ArrayFlags::Fortran
      if BaseHelpers.is_obj_fortran(self)
        @flags |= ArrayFlags::Fortran

        # mutually exclusive
        if ndims > 1
          @flags &= ~ArrayFlags::Contiguous
        end
      else
        @flags &= ~ArrayFlags::Fortran
      end
    end

    if flagmask & ArrayFlags::Contiguous
      if BaseHelpers.is_obj_contiguous(self)
        @flags |= ArrayFlags::Contiguous

        # mutually exclusive
        if ndims > 1
          @flags &= ~ArrayFlags::Fortran
        end
      else
        @flags &= ~ArrayFlags::Contiguous
      end
    end
  end

  def to_s(io)
    printer = ToString::BasePrinter.new(self, io)
    printer.print
  end

  def flat_iter
    SafeNDIter.new(self)
  end

  def unsafe_iter
    UnsafeNDIter.new(self)
  end
end
