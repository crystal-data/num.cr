require "../base"

module Num::StrideTricks
  extend self

  def view(arr : BaseArray(T), dtype : U.class) forall T, U
    if T == U
      return arr.dup_view
    end

    tsize = T == Bool ? 1 : sizeof(T)
    usize = U == Bool ? 1 : sizeof(U)

    newsize = tsize / usize
    newlast = arr.shape[-1] * newsize
    intlast = Int32.new newlast

    if newlast != intlast || intlast == 0
      raise "Cannot view array as #{U}"
    end

    newshape = arr.shape.dup
    newstrides = arr.strides.dup
    newflags = arr.flags.dup
    ptr = arr.buffer.unsafe_as(Pointer(U))

    newshape[-1] = Int32.new(newlast)

    sz = 1
    arr.ndims.times do |i|
      newstrides[arr.ndims - i - 1] = sz
      sz *= newshape[arr.ndims - i - 1]
    end

    arr.basetype(U).new(ptr, newshape, newstrides, newflags, nil)
  end

  # Determines if two shapes are broadcastable against each other.
  # The rules for checking this property are well defined:
  #
  # Two dimensions are compatible if:
  #   - they are equal
  #   - one of them is equal to 1
  #
  # If the axes of the array are different lengths, dimensions of
  # size one can be appended to one or the other in order to make
  # the arrays broadcastable against each other and satisfy the
  # rules for broadcastable dimensions.
  def broadcastable(arr : BaseArray, other : BaseArray)
    # Fast track instances where the two shapes already match, no
    # need in pointlessly calculating the same shape that
    # already exists
    return [] of Int32 unless arr.shape != other.shape

    sz = arr.shape.size
    osz = other.shape.size

    # If the sizes already match, the rules are well defined
    # to make a broadcast.
    if sz == osz
      # Check the shapes, return the new shape, both arrays will
      # be broadcasted, so this can't be used for in-place operations,
      # only one of the arrays can be broadcasted in that case.
      if broadcast_equal(arr.shape, other.shape)
        return broadcastable_shape(arr.shape, other.shape)
      end
    else
      # Both of these paths prepend ones to the smaller shape
      # in order to match broadcasting rules.
      if sz > osz
        othershape = [1] * (sz - osz) + other.shape
        if broadcast_equal(arr.shape, othershape)
          return broadcastable_shape(arr.shape, othershape)
        end
      else
        selfshape = [1] * (osz - sz) + arr.shape
        if broadcast_equal(selfshape, other.shape)
          return broadcastable_shape(selfshape, other.shape)
        end
      end
    end
    # If no broadcasting is possible, raise a ShapeError.  No other
    # result makes sense, operation has to fail.
    raise Exceptions::ShapeError.new("Shapes #{arr.shape} and #{other.shape} are not broadcastable")
  end

  # Broadcasts an array to a new shape. A readonly view on the original array
  # with the given shape. It is typically not contiguous. Furthermore,
  # more than one element of a broadcasted array may refer to a single
  # memory location.
  def broadcast_to(arr : BaseArray, newshape : Array(Int32))
    dim = newshape.size
    defstrides = [0] * dim
    sz = 1
    dim.times do |i|
      defstrides[dim - i - 1] = sz
      sz *= newshape[dim - i - 1]
    end

    newstrides = broadcast_strides(newshape, arr.shape, defstrides, arr.strides)
    newflags = Internal::ArrayFlags::None
    newbase = arr.base ? arr.base : arr

    arr.class.new(arr.buffer, newshape, newstrides, newflags, newbase, false)
  end

  # as_strided creates a view into the array given the exact strides and
  # shape. This means it manipulates the internal data structure of
  # a Tensor and, if done incorrectly, the array elements can point
  # to invalid memory and can corrupt results or crash your program.
  # It is advisable to always use the original x.strides when
  # calculating new strides to avoid reliance on a contiguous
  # memory layout.
  #
  # Furthermore, arrays created with this function often contain self
  # overlapping memory, so that two elements are identical.
  # Vectorized write operations on such arrays will typically be
  # unpredictable. They may even give different results for
  # small, large, or transposed arrays. Since writing to these
  # arrays has to be tested and done with great care, you may want
  # to use writeable=false to avoid accidental write operations.
  def as_strided(arr : BaseArray, shape : Array(Int32), strides : Array(Int32), writeable = false)
    newflags = arr.flags.dup
    if !writeable
      newflags = Internal::ArrayFlags::None
    end
    newbase = arr.base.nil? ? arr : arr.base
    arr.class.new(arr.buffer, shape, strides, newflags, newbase)
  end

  # Finds the strides that must be present in order to broadcast an existing
  # array to a new array, or raises that the array cannot be broadcasted into
  # the provided shape.
  #
  # This method is primarily used by unsafe methods, such as `broadcast_to`.
  # When using this method, the resulting array's flags should ideally be
  # set to readonly, since many locations can share memory.  This method is
  # a safer alternative to `as_strided`
  private def broadcast_strides(dest_shape, src_shape, dest_strides, src_strides)
    # Find where the strides need to begin to match the input
    # shape/strides to the new shape
    dims = dest_shape.size
    start = dims - src_shape.size

    ret = [0] * dims
    (dims - 1).step(to: start, by: -1) do |i|
      s = src_shape[i - start]
      case s
      # Zero strides in a dimension is the easiest way to "trick"
      # the nditerator to traverse that dimension multiple times
      # and produce the same value.  This does however mean that
      # the iterator will produce many instances of the same pointer
      # when iterating through a broadcasted array.
      #
      # This is the reason for the read only flag on an array.
      when 1
        ret[i] = 0
      when dest_shape[i]
        # Otherwise the broadcasted strides will be computed from
        # the source strides, this path will always be chosen when
        # trying to broadcast to an identically shaped array for example.
        ret[i] = src_strides[i - start]
      else
        # Since the zero shaped dimensions will appear with invalid broadcasts,
        # raise here to indicate that the two shapes are incompatible.
        raise Exceptions::ShapeError.new("Cannot broadcast from #{src_shape} to #{dest_shape}")
      end
    end
    ret
  end

  # This method checks if two shapes are broadcastable with each other
  # in their current form.  There are several manipulations that can
  # be done on shapes to make them broadcastable eventually, so this
  # may be checked several times when broadcasting.
  private def broadcast_equal(a, b)
    bc = true
    a.zip(b) do |i, j|
      # Shapes can be broadcast against each other if for every dimension
      # the following is true: the dimensions are equal among the two shapes,
      # or either of the dimensions is equal to 1
      if !(i == j || i == 1 || j == 1)
        bc = false
      end
    end
    bc
  end

  # Once an array is determined to be broadcastable, the resulting
  # shape is simply the maximum value found at each dimension of the
  # two shapes.
  private def broadcastable_shape(a, b)
    a.zip(b).map do |i|
      Math.max(i[0], i[1])
    end
  end
end
