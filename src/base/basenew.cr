require "./flags"
require "./print"
require "../core/assemble"
require "../core/exceptions"
require "../iter/flat"
require "../iter/nd"
require "../iter/axes"
require "../iter/index"
require "../iter/permute"

abstract class Bottle::BaseArray(T)
  include Internal

  # Buffer pointing to the start of the array's data buffer
  getter buffer : Pointer(T)

  # Array of the array's dimensions
  property shape : Array(Int32)

  # Number of steps in each dimension to traverse the array
  property strides : Array(Int32)

  # Number of array dimensions
  getter ndims : Int32

  # Information about the memory layout of the array
  getter flags : ArrayFlags

  # The total number of elements in the array
  getter size : Int32

  # Base object if memory is from another array
  getter base : BaseArray(T)? = nil

  def initialize(_shape : Array(Int32),
                 order : ArrayFlags = ArrayFlags::Contiguous,
                 ptr : Pointer(T)? = nil)
    check_type
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

end
