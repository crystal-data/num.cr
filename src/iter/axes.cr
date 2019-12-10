require "../base/base"

struct Num::Iter::AxisIter(T)
  include Iterator(T)
  @shape : Array(Int32)
  @strides : Array(Int32)
  @ptr : Pointer(T)
  @tmp : BaseArray(T)
  @total : Int32
  @yielded : Int32 = 0
  @axis : Int32

  def initialize(arr : BaseArray(T), @axis : Int32 = -1, keepdims = false)
    if @axis < 0
      @axis += arr.ndims
    end
    unless @axis < arr.ndims
      raise Exceptions::AxisError.new("Axis out of range for array")
    end

    @shape = arr.shape.dup
    @strides = arr.strides.dup
    @ptr = arr.buffer

    if keepdims
      @shape[axis] = 1
      @strides[axis] = 0
    else
      @shape.delete_at(axis)
      @strides.delete_at(axis)
    end

    @tmp = arr.class.new(@ptr, @shape, @strides, Bottle::Internal::ArrayFlags::None, nil).dup

    @total = @shape[axis]
  end

  def next
    if @yielded > @total
      stop
    else
      ret = @tmp
      @yielded += 1
      @ptr += @strides[@axis] * @shape[@axis]
      @tmp = Tensor.new(@ptr, @shape, @strides, Bottle::Internal::ArrayFlags::None, nil)
      ret
    end
  end
end
