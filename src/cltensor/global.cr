require "opencl"

module NumInternal
  private struct ClInfo
    getter device : LibCL::ClDeviceId
    getter context : LibCL::ClContext
    getter queue : LibCL::ClCommandQueue

    def initialize(@device : LibCL::ClDeviceId, @context : LibCL::ClContext, @queue : LibCL::ClCommandQueue)
    end
  end

  class ClContext
    class_getter instance : ClInfo { ClInfo.new(*Cl.single_device_defaults) }
  end
end
