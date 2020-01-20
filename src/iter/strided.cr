require "../base/base"

struct NumInternal::StridedIter(T)
  include Iterator(T)
  @ptr : Pointer(T)
  @coord : Pointer(Int32)
  @backstrides : Pointer(Int32)
  @shape : Pointer(Int32)
  @strides : Pointer(Int32)
  @dim : Int32
  @done : Bool = false

  def initialize(arr : Num::BaseArray(T))
    @ptr = arr.buffer
    @coord = Pointer(Int32).malloc(arr.ndims, 0)
    @backstrides = Pointer(Int32).malloc(arr.ndims) { |i| arr.strides[i] * (arr.shape[i] - 1) }
    @shape = arr.shape.to_unsafe
    @strides = arr.strides.to_unsafe
    @dim = arr.ndims - 1

    arr.ndims.times do |i|
      if @strides[i] < 0
        @ptr += (@shape[i] - 1) * @strides[i].abs
      end
    end
  end

  def next
    if @done
      return stop
    end
    ret = @ptr
    @dim.step(to: 0, by: -1) do |k|
      if @coord[k] < @shape[k] - 1
        @coord[k] += 1
        @ptr += @strides[k]
        break
      else
        if k == 0
          @done = true
        end
        @coord[k] = 0
        @ptr -= @backstrides[k]
      end
    end
    ret
  end
end
