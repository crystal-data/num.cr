require "../base/base"

struct Bottle::Iter::NDFlatIter(T)
  include Iterator(T)
  @ptr : Pointer(T)
  @shape : Pointer(Int32)
  @strides : Pointer(Int32)
  @track : Pointer(Int32)
  @dim : Int32

  def initialize(arr : BaseArray(T))
    @ptr = arr.buffer
    @shape = arr.shape.to_unsafe
    @strides = arr.strides.to_unsafe
    @track = Pointer(Int32).malloc(arr.ndims, 0)
    @dim = arr.ndims - 1
    @done = false
  end

  def next
    if @done
      return stop
    end

    ret = @ptr
    @dim.step(to: 0, by: -1) do |i|
      @track[i] += 1
      shape_i = @shape[i]
      stride_i = @strides[i]

      if @track[i] == shape_i
        if i == 0
          @done = true
        end
        @track[i] = 0
        @ptr -= (shape_i - 1) * stride_i
        next
      end
      @ptr += stride_i
      break
    end
    ret
  end
end

struct Bottle::Iter::UnsafeNDFlatIter(T)
  include Iterator(T)
  @ptr : Pointer(T)
  @shape : Pointer(Int32)
  @strides : Pointer(Int32)
  @track : Pointer(Int32)
  @dim : Int32

  def initialize(arr : BaseArray(T))
    @ptr = arr.buffer
    @shape = arr.shape.to_unsafe
    @strides = arr.strides.to_unsafe
    @track = Pointer(Int32).malloc(arr.ndims, 0)
    @dim = arr.ndims - 1
  end

  def next
    ret = @ptr
    @dim.step(to: 0, by: -1) do |i|
      @track[i] += 1
      shape_i = @shape[i]
      stride_i = @strides[i]
      if @track[i] == shape_i
        @track[i] = 0
        @ptr -= (shape_i - 1) * stride_i
        next
      end
      @ptr += stride_i
      break
    end
    ret
  end
end
