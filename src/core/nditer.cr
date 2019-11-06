require "./ndtensor"

struct Bottle::Internal::FlatIter(T)
  include Iterator(T)

  @buffer : Pointer(T)

  # Tracks the current dimensional status of the NDIter
  # To determine if a stride adjustment needs to be made
  # as the iterator jumps to the next dimension.
  @track : Pointer(Int32)

  # Stores the shape of a passed Tensor in order to compute
  # iteration.
  @shape : Array(Int32)

  # Stores the strides of a passed Tensor to compute
  # iteration offsets
  @strides : Array(Int32)

  # The next index that an NDIter will return
  @next_index : Pointer(T)

  # The last index that the NDIter returned
  @last_index : Pointer(T)

  # Flag checks if an NDIter has been exhausted
  @done : Bool

  @ndims : Int32

  def initialize(t : Tensor(T))
    @buffer = t.@buffer
    @ndims = t.ndims
    @track = Pointer(Int32).malloc(t.ndims, 0)
    @shape = t.shape
    @strides = t.strides
    @next_index = @buffer
    @last_index = @buffer
    @done = false
  end

  def next
    if @done
      return stop
    end
    v = @ndims - 1
    next_index = @next_index
    @last_index = next_index

    v.step(to: 0, by: -1) do |i|
      @track[i] += 1
      shape_i = @shape[i]
      stride_i = @strides[i]

      if @track[i] == shape_i
        if i == 0
          @done = true
        end
        @track[i] = 0
        next_index -= (shape_i - 1) * stride_i
        next
      end
      next_index += stride_i
      break
    end

    @next_index = next_index
    @last_index
  end

  def rewind(buffer : Pointer(T))
    @done = false
    @track = Pointer(Int32).malloc(@ndims, 0)
    @last_index = @buffer
    @next_index = @buffer
  end
end

struct Bottle::Internal::UnsafeIter(T)
  include Iterator(T)

  @buffer : Pointer(T)

  # Tracks the current dimensional status of the NDIter
  # To determine if a stride adjustment needs to be made
  # as the iterator jumps to the next dimension.
  @track : Pointer(Int32)

  # Stores the shape of a passed Tensor in order to compute
  # iteration.
  @shape : Array(Int32)

  # Stores the strides of a passed Tensor to compute
  # iteration offsets
  @strides : Array(Int32)

  # The next index that an NDIter will return
  @next_index : Pointer(T)

  # The last index that the NDIter returned
  @last_index : Pointer(T)

  # Flag checks if an NDIter has been exhausted
  @done : Bool

  @ndims : Int32

  @contiguous : Bool

  def initialize(t : Bottle::Tensor(T))
    @buffer = t.@buffer
    @ndims = t.ndims
    @track = Pointer(Int32).malloc(t.ndims, 0)
    @shape = t.shape
    @strides = t.strides
    @next_index = @buffer
    @last_index = @buffer
    @done = false
    @contiguous = (t.flags & ArrayFlags::Contiguous).value > 0
  end

  def next
    @contiguous ? onednext : ndnext
  end

  def onednext
    @last_index = @next_index
    @next_index += @strides[-1]
    @last_index
  end

  def ndnext
    v = @ndims - 1
    next_index = @next_index
    @last_index = next_index

    v.step(to: 0, by: -1) do |i|
      @track[i] += 1
      shape_i = @shape[i]
      stride_i = @strides[i]

      if @track[i] == shape_i
        if i == 0
          @done = true
        end
        @track[i] = 0
        next_index -= (shape_i - 1) * stride_i
        next
      end
      next_index += stride_i
      break
    end

    @next_index = next_index
    @last_index
  end

  def rewind(buffer : Pointer(T))
    @done = false
    @track = Pointer(Int32).malloc(@ndims, 0)
    @last_index = @buffer
    @next_index = @buffer
  end
end

struct Bottle::Internal::MatrixIter(T)
  include Iterator(T)
  getter tns : Tensor(T)
  getter axis : Int32
  getter dim : Int32
  getter ranges : Array(Int32 | Range(Int32, Int32))
  @idx : Int32 = 0

  def initialize(@tns : Tensor(T))
    if tns.ndims < 3
      raise "Dimensionality of the Tensor is not high enough to reduce"
    end

    @axis = tns.ndims - 3
    @dim = tns.shape[axis]
    @ranges = tns.shape.map_with_index do |a, i|
      axis == i ? 0 : 0...a
    end
  end

  def next
    if @idx == @dim
      return stop
    end
    @ranges[axis] = @idx
    @idx += 1
    tns.slice(@ranges)
  end
end
