require "../base/base"
require "../tensor/tensor"

struct Num::Iter::ContigFlatIter(T)
  include Iterator(T)
  @ptr : Pointer(T)
  @size : Int32
  @step : Int32
  @offset : Int32

  def initialize(arr : BaseArray(T))
    @ptr = arr.buffer
    @size = arr.size
    @step = arr.strides.size == 0 ? 0 : arr.strides[-1]
    @offset = 0
  end

  def next
    ret = @ptr
    if @offset < @size
      @offset += 1
      @ptr += @step
      ret
    else
      stop
    end
  end
end

struct Num::Iter::UnsafeContigFlatIter(T)
  include Iterator(T)
  @ptr : Pointer(T)
  @size : Int32
  @step : Int32
  @offset : Int32

  def initialize(arr : BaseArray(T))
    @ptr = arr.buffer
    @size = arr.size
    @step = arr.strides[-1]
    @offset = 0
  end

  def next
    ret = @ptr
    @ptr += @step
    ret
  end
end
