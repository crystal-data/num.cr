require "opencl"
require "./global"
require "../tensor/tensor"

class ClTensor(T)
  # Dimensions of the tensor
  getter shape : Array(Int32)

  # Step in each dimension of a tensor
  getter strides : Array(Int32)

  # Storage of a tensor
  getter buffer : LibCL::ClMem

  def rank
    @shape.size
  end

  def size
    @shape.product
  end

  def initialize(@shape : Array(Int32))
    @buffer = Cl.buffer(NumInternal::ClContext.instance.context, UInt64.new(@shape.product), dtype: T)
    @strides = NumInternal.shape_to_strides(@shape)
  end

  def initialize(@shape : Array(Int32), @strides : Array(Int32), @buffer : LibCL::ClMem)
  end

  def free
    Cl.release_buffer(@buffer)
  end

  def cpu
    ptr = Pointer(T).malloc(size, 0)
    LibCL.cl_enqueue_read_buffer(
      NumInternal::ClContext.instance.queue,
      @buffer,
      LibCL::CL_TRUE,
      0_u64,
      UInt64.new(size * sizeof(T)),
      ptr,
      0_u32, nil, nil
    )
    Tensor(T).new(ptr, @shape, @strides)
  end

  def layout_on_device
    ClTensorLayout(T).new(self)
  end
end

class ClTensorLayout(T)
  getter rank : Int32
  getter shape : LibCL::ClMem
  getter strides : LibCL::ClMem
  getter data : LibCL::ClMem
  getter size : Int32

  def initialize(t : ClTensor(T))
    @rank = t.rank
    @data = t.buffer
    @size = t.size

    @shape = Cl.buffer(NumInternal::ClContext.instance.context, UInt64.new(@rank), dtype: Int32)
    @strides = Cl.buffer(NumInternal::ClContext.instance.context, UInt64.new(@rank), dtype: Int32)

    bytes = t.rank * sizeof(Int32)

    LibCL.cl_enqueue_write_buffer(
      ClContext.instance.queue,
      @shape,
      LibCL::CL_FALSE,
      0,
      bytes,
      t.shape,
      0, nil, nil,
    )

    LibCL.cl_enqueue_write_buffer(
      ClContext.instance.queue,
      @strides,
      LibCL::CL_TRUE,
      0,
      bytes,
      t.shape,
      0, nil, nil,
    )
  end

  def free
    @shape.free
    @strides.free
  end
end

module NumInternal
  extend self

  def shape_to_strides(shape : Array(Int32), order : Char = 'C')
    ndims = shape.size
    strides = [0] * ndims
    sz = 1
    case order
    # For Fortran ordered arrays strides are calculated from
    # the beginning of the shape to the end, with strides
    # monotonically increasing.
    when 'F'
      ndims.times do |i|
        strides[i] = sz
        sz *= shape[i]
      end
      # Otherwise, row major order is chosen and strides are
      # calculated from the reversed shape, monotonically
      # decreasing.
    else
      ndims.times do |i|
        strides[ndims - i - 1] = sz
        sz *= shape[ndims - i - 1]
      end
    end
    strides
  end
end
