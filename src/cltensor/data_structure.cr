require "opencl"
require "./global"

module NumInternal

  # Wrapper to data stored on an OpenCL Device
  struct ClStorage(T)
    # Size of the data stored
    getter size : Int32

    # Pointer to the data stored
    getter data : LibCL::ClMem

    # Initializes an empty block of data on an opencl device
    def initialize(@size : Int)
      @data = Cl.buffer(ClContext.instance.context, UInt64.new(@size), dtype: T)
    end

    def to_unsafe
      @data
    end

    # Frees data on an opencl device
    def free
      Cl.release_buffer(@data)
      @size = 0
    end
  end

  struct ClTensorLayout(T)
    getter rank : Int32
    getter shape : ClStorage(Int32)
    getter strides : ClStorage(Int32)
    getter data : ClStorage(T)
    getter size : Int32

    def initialize(t : ClTensor(T))
      @rank = t.rank
      @data = t.storage
      @size = t.size

      @shape = ClStorage(Int32).new(t.rank)
      @strides = ClStorage(Int32).new(t.rank)

      bytes = t.rank * sizeof(Int32)

      LibCL.cl_enqueue_write_buffer(
        ClContext.instance.queue,
        @shape.data,
        LibCL::CL_FALSE,
        0,
        bytes,
        t.shape,
        0, nil, nil,
      )

      LibCL.cl_enqueue_write_buffer(
        ClContext.instance.queue,
        @strides.data,
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

  # Tensor object stored on an Opencl device
  class ClTensor(T)

    # Dimensions of the tensor
    getter shape : Array(Int32)

    # Step in each dimension of a tensor
    getter strides : Array(Int32)

    # Storage of a tensor
    getter storage : ClStorage(T)

    # Number of dimensions of a tensor
    def rank
      @shape.size
    end

    # Total elements in the tensor
    def size
      @shape.product
    end

    def initialize(@shape : Array(Int32))
      @strides = shape_to_strides(@shape)
      @storage = ClStorage(T).new(@shape.product)
    end

    def free
      @storage.free
      @shape = [] of Int32
      @strides = [] of Int32
    end

    def layout_on_device : ClTensorLayout
      ClTensorLayout.new(self)
    end
  end

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
