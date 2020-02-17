@[Link("OpenCL")]
lib LibCL
  alias ClPlatformId = Void*
  alias ClDeviceId = Void*
  alias ClMem = Void*
  alias ClContextProperties = Int32
  alias ClCommandQueue = Void*
  alias ClContext = Void*
  alias ClUint = UInt32
  alias ClInt = Int32

  CONTEXT_PLATFORM = 1084

  enum ClDeviceType
    DEFAULT = 1
    CPU = 2
    GPU = 4
    ACCELERATOR = 8
    CUSTOM = 16
  end

  enum ClDeviceInfo
    DEVICE_TYPE = 0x00001000
    DEVICE_VENDOR_ID = 0x00001001
    DEVICE_MAX_COMPUTE_UNITS = 0x00001002
    DEVICE_MAX_WORK_ITEM_DIMENSIONS = 0x00001003
    DEVICE_MAX_WORK_GROUP_SIZE = 0x00001004
    DEVICE_MAX_WORK_ITEM_SIZES = 0x00001005
    DEVICE_PREFERRED_VECTOR_WIDTH_CHAR = 0x00001006
    DEVICE_PREFERRED_VECTOR_WIDTH_SHORT = 0x00001007
    DEVICE_PREFERRED_VECTOR_WIDTH_INT = 0x00001008
    DEVICE_PREFERRED_VECTOR_WIDTH_LONG = 0x00001009
    DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT = 0x0000100A
    DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE = 0x0000100B
    DEVICE_MAX_CLOCK_FREQUENCY = 0x0000100C
    DEVICE_ADDRESS_BITS = 0x0000100D
    DEVICE_MAX_READ_IMAGE_ARGS = 0x0000100E
    DEVICE_MAX_WRITE_IMAGE_ARGS = 0x0000100F
    DEVICE_MAX_MEM_ALLOC_SIZE = 0x00001010
    DEVICE_IMAGE2D_MAX_WIDTH = 0x00001011
    DEVICE_IMAGE2D_MAX_HEIGHT = 0x00001012
    DEVICE_IMAGE3D_MAX_WIDTH = 0x00001013
    DEVICE_IMAGE3D_MAX_HEIGHT = 0x00001014
    DEVICE_IMAGE3D_MAX_DEPTH = 0x00001015
    DEVICE_IMAGE_SUPPORT = 0x00001016
    DEVICE_MAX_PARAMETER_SIZE = 0x00001017
    DEVICE_MAX_SAMPLERS = 0x00001018
    DEVICE_MEM_BASE_ADDR_ALIGN = 0x00001019
    DEVICE_MIN_DATA_TYPE_ALIGN_SIZE = 0x0000101A
    DEVICE_SINGLE_FP_CONFIG = 0x0000101B
    DEVICE_GLOBAL_MEM_CACHE_TYPE = 0x0000101C
    DEVICE_GLOBAL_MEM_CACHELINE_SIZE = 0x0000101D
    DEVICE_GLOBAL_MEM_CACHE_SIZE = 0x0000101E
    DEVICE_GLOBAL_MEM_SIZE = 0x0000101F
    DEVICE_MAX_CONSTANT_BUFFER_SIZE = 0x00001020
    DEVICE_MAX_CONSTANT_ARGS = 0x00001021
    DEVICE_LOCAL_MEM_TYPE = 0x00001022
    DEVICE_LOCAL_MEM_SIZE = 0x00001023
    DEVICE_ERROR_CORRECTION_SUPPORT = 0x00001024
    DEVICE_PROFILING_TIMER_RESOLUTION = 0x00001025
    DEVICE_ENDIAN_LITTLE = 0x00001026
    DEVICE_AVAILABLE = 0x00001027
    DEVICE_COMPILER_AVAILABLE = 0x00001028
    DEVICE_EXECUTION_CAPABILITIES = 0x00001029
    DEVICE_QUEUE_PROPERTIES = 0x0000102A
    DEVICE_NAME = 0x0000102B
    DEVICE_VENDOR = 0x0000102C
    DRIVER_VERSION = 0x0000102D
    DEVICE_PROFILE = 0x0000102E
    DEVICE_VERSION = 0x0000102F
    DEVICE_EXTENSIONS = 0x00001030
    DEVICE_PLATFORM = 0x00001031
    DEVICE_DOUBLE_FP_CONFIG = 0x00001032
  end

  enum ClPlatformInfo
    PLATFORM_PROFILE = 0x00000900
    PLATFORM_VERSION = 0x00000901
    PLATFORM_NAME = 0x00000902
    PLATFORM_VENDOR = 0x00000903
    PLATFORM_EXTENSIONS = 0x00000904
  end

  QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE = 1 << 0
  QUEUE_PROFILING_ENABLE = 1 << 1

  enum ClMemFlags
    READ_WRITE = 1 << 0
    WRITE_ONLY = 1 << 1
    READ_ONLY = 1 << 2
    USE_HOST_PTR = 1 << 3
    ALLOC_HOST_PTR = 1 << 4
    COPY_HOST_PTR = 1 << 5
  end

  fun cl_get_platform_ids = clGetPlatformIDs(num_entries : ClUint, platforms : ClPlatformId*, num_platforms : ClUint*) : ClInt
  fun cl_get_device_ids = clGetDeviceIDs(platform : ClPlatformId, cl_device_type : ClDeviceType, num_entries : ClUint, devices : ClDeviceId*, num_devices : ClUint*) : ClInt

  fun cl_create_context = clCreateContext(
    properties : ClContextProperties*,
    num_devices : ClUint,
    devices : ClDeviceId*,
    pfn_notify : (Char*, Void*, LibC::SizeT, Void* ->),
    user_data : Void*,
    errcode_ret : ClInt*) : ClContext

  fun cl_create_command_queue = clCreateCommandQueue(
    context : ClContext,
    device : ClDeviceId,
    properties : Int32,
    err : ClInt*
  ) : ClCommandQueue

  fun cl_get_device_info = clGetDeviceInfo(device : ClDeviceId, param_name : ClDeviceInfo, param_value_size : LibC::SizeT, param_value : Void*, param_value_size_ret : LibC::SizeT*) : ClInt
  fun cl_get_platform_info = clGetPlatformInfo(platform : ClPlatformId, param_name : ClPlatformInfo, param_value_size : LibC::SizeT, param_value : Void*, param_value_size_ret : LibC::SizeT*) : ClInt

  fun cl_create_buffer = clCreateBuffer(context : ClContext, flags : ClMemFlags, size : LibC::SizeT, host_ptr : Void*, errcode_ret : ClInt*) : ClMem
  fun cl_release_mem_object = clReleaseMemObject(memobj : ClMem) : ClInt
end

info = LibCL.cl_get_platform_ids(1, out platforms, out num_platforms)
info = LibCL.cl_get_device_ids(platforms, LibCL::ClDeviceType::GPU, 1, out device, out num_devices)

ctx = LibCL.cl_create_context(nil, 1, pointerof(device), nil, nil, out errcode)
queue = LibCL.cl_create_command_queue(ctx, device, 0, out err)

a = LibCL.cl_create_buffer(ctx, LibCL::ClMemFlags::READ_ONLY, 100 * sizeof(Float64), nil, out a_err)
LibCL.cl_release_mem_object(a)
