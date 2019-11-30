require "../base/base"

module Bottle::Internal::Core
  def astensor(a : Tensor)
    a
  end

  def astensor(a : BaseArray(U)) forall U
    Tensor(U).new(a.buffer, a.shape, a.strides, a.flags, nil)
  end

  def astensor(a : Array)
    Tensor.from_array a
  end

  def astensor(a : Number)
    Tensor.new(a)
  end
end
