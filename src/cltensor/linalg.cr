require "../libs/clblast"
require "./cltensor"

class ClTensor(T) < NumInternal::AnyTensor(T)
  macro blast(name, *args, prefix = "")
    {%
      if T == Float32
        typ = :S.id
      elsif T == Float64
        typ = :D.id
      end
    %}
    event = Pointer(Void).malloc(1).unsafe_as(LibCL::ClEvent)
    queue = Num::ClContext.instance.queue

    LibBlast.clblast_{{prefix.id}}{{typ}}{{name}}({{*args}}, pointerof(queue), pointerof(event))
    Cl.check LibCL.cl_wait_for_events(1, pointerof(event))
    Cl.check LibCL.cl_release_event(event)
  end

  def scal!(a : T)
    blast(scal, size, a, self.to_unsafe, 0, 1)
  end

  def axpy!(a : T, x : ClTensor(T))
    blast(axpy, size, a, x.to_unsafe, 0, 1, to_unsafe, 0, 1)
  end

  def dot(other : ClTensor(T), ret : ClTensor(T)? = nil)
    if ret.nil?
      back = ClTensor(T).new([1])
    else
      back = ret
    end
    blast(dot, size, back.to_unsafe, 0, to_unsafe, 0, 1, other.to_unsafe, 0, 1)
    back
  end

  def nrm2(ret : ClTensor(T)? = nil)
    if ret.nil?
      back = ClTensor(T).new([1])
    else
      back = ret
    end
    blast(nrm2, size, back.to_unsafe, 0, to_unsafe, 0, 1)
    back
  end

  def asum(ret : ClTensor(T)? = nil)
    if ret.nil?
      back = ClTensor(T).new([1])
    else
      back = ret
    end
    blast(asum, size, back.to_unsafe, 0, to_unsafe, 0, 1)
    back
  end

  def sum(ret : ClTensor(T)? = nil)
    if ret.nil?
      back = ClTensor(T).new([1])
    else
      back = ret
    end
    blast(sum, size, back.to_unsafe, 0, to_unsafe, 0, 1)
    back
  end

  def amax(ret : ClTensor(Int32)? = nil)
    if ret.nil?
      back = ClTensor(Int32).new([1])
    else
      back = ret
    end
    blast(amax, size, back.to_unsafe, 0, to_unsafe, 0, 1, prefix: i)
    back
  end

  def amin(ret : ClTensor(Int32)? = nil)
    if ret.nil?
      back = ClTensor(Int32).new([1])
    else
      back = ret
    end
    blast(amin, size, back.to_unsafe, 0, to_unsafe, 0, 1, prefix: i)
    back
  end

  def argmax(ret : ClTensor(Int32)? = nil)
    if ret.nil?
      back = ClTensor(Int32).new([1])
    else
      back = ret
    end
    blast(max, size, back.to_unsafe, 0, to_unsafe, 0, 1, prefix: i)
    back
  end

  def argmin(ret : ClTensor(Int32)? = nil)
    if ret.nil?
      back = ClTensor(Int32).new([1])
    else
      back = ret
    end
    blast(min, size, back.to_unsafe, 0, to_unsafe, 0, 1, prefix: i)
    back
  end

  def matmul(other : ClTensor(T), ret : ClTensor(T)? = nil)
    if ret.nil?
      back = ClTensor(T).new([shape[0], other.shape[1]])
    else
      back = ret
    end

    blast(gemm, LibBlast::CLBlastLayout::CLBlastLayoutRowMajor,
      LibBlast::CLBlastTranspose::CLBlastTransposeNo,
      LibBlast::CLBlastTranspose::CLBlastTransposeNo,
      shape[0], other.shape[1], shape[1], 1.0, to_unsafe,
      0, shape[1], other.to_unsafe, 0, other.shape[1], 1.0,
      back.to_unsafe, 0, back.shape[1])

    back
  end
end
