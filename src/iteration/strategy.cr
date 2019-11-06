require "../core/ndtensor"
require "benchmark"

module Bottle::Internal::Iteration
  macro contig_iteration_macro(safe)
    @last_index = @next_index
    @next_index += @strides[@ndims - 1]
    {% if safe %}
      @size -= 1
      if @size < 0
        return stop
      end
    {% end %}
    @last_index
  end

  macro nd_iteration_macro(safe)
    {% if safe %}
      if @done
        return stop
      end
    {% end %}
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
        @next_index -= (shape_i - 1) * stride_i
        next
      end
      @next_index += stride_i
      break
    end

    @next_index = next_index
    @last_index
  end
end

module Bottle::Internal::NDIter(T)
  include Iterator(T)

  @buffer : Pointer(T)

  # Tracks the current dimensional status of the NDIter
  # To determine if a stride adjustment needs to be made
  # as the iterator jumps to the next dimension.
  @track : Pointer(Int32)

  # Stores the shape of a passed Tensor in order to compute
  # iteration.
  @shape : Pointer(Int32)

  # Stores the strides of a passed Tensor to compute
  # iteration offsets
  @strides : Pointer(Int32)

  # The next index that an NDIter will return
  @next_index : Pointer(T)

  # The last index that the NDIter returned
  @last_index : Pointer(T)

  # Flag checks if an NDIter has been exhausted
  @done : Bool

  # The number of dimensions in the `Tensor`
  @ndims : Int32

  # Total number of elements in the `Tensor`
  @size : Int32

  def initialize(t : Bottle::Tensor(T))
    @buffer = t.@buffer
    @ndims = t.ndims
    @track = Pointer(Int32).malloc(t.ndims, 0)
    @shape = t.shape.to_unsafe
    @strides = t.strides.to_unsafe
    @next_index = @buffer
    @last_index = @buffer
    @done = false
    @size = t.size
  end

  abstract def next

  def rewind(buffer : Pointer(T))
    @done = false
    @track = Pointer(Int32).malloc(@ndims, 0)
    @last_index = @buffer
    @next_index = @buffer
  end
end

struct Bottle::Internal::SafeNDIter(T)
  struct OneD(T)
    include NDIter(T)

    def next
      Iteration.contig_iteration_macro(true)
    end
  end

  struct ND(T)
    include NDIter(T)

    def next
      Iteration.nd_iteration_macro(true)
    end
  end

  getter strategy : OneD(T) | ND(T)

  def initialize(t : Bottle::Tensor(T))
    contiguous = (t.flags & ArrayFlags::Contiguous | ArrayFlags::Fortran).value > 0
    @strategy = (contiguous) ? OneD.new(t) : ND.new(t)
  end

  forward_missing_to @strategy
end

struct Bottle::Internal::UnsafeNDIter(T)
  struct OneD(T)
    include NDIter(T)

    def next
      Iteration.contig_iteration_macro(false)
    end
  end

  struct ND(T)
    include NDIter(T)

    def next
      Iteration.nd_iteration_macro(false)
    end
  end

  getter strategy : OneD(T) | ND(T)

  def initialize(t : Bottle::Tensor(T))
    contiguous = (t.flags & ArrayFlags::Contiguous | ArrayFlags::Fortran).value > 0
    @strategy = (contiguous) ? OneD.new(t) : ND.new(t)
  end

  forward_missing_to @strategy
end
