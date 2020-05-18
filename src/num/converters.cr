require "../tensor/tensor"

module Num
  # Converts input data, in any form that can be converted to a tensor,
  # into a tensor.
  #
  # This includes arrays, nested arrays, scalars, and tensors.
  # Data will not be copied unless necessary.  Base classes will not
  # be maintained, all inputs will be coerced to Tensors or raise.
  def astensor(a : Tensor)
    a
  end

  # Converts input data, in any form that can be converted to a tensor,
  # into a tensor.
  #
  # This includes arrays, nested arrays, scalars, and tensors.
  # Data will not be copied unless necessary.  Base classes will not
  # be maintained, all inputs will be coerced to Tensors or raise.
  def astensor(a : AnyArray(U)) forall U
    Tensor(U).new(a.to_unsafe, a.shape, a.strides, a.flags)
  end

  # Converts input data, in any form that can be converted to a tensor,
  # into a tensor.
  #
  # This includes arrays, nested arrays, scalars, and tensors.
  # Data will not be copied unless necessary.  Base classes will not
  # be maintained, all inputs will be coerced to Tensors or raise.
  def astensor(a : Array)
    Tensor.from_array a
  end

  # Converts input data, in any form that can be converted to a tensor,
  # into a tensor.
  #
  # This includes arrays, nested arrays, scalars, and tensors.
  # Data will not be copied unless necessary.  Base classes will not
  # be maintained, all inputs will be coerced to Tensors or raise.
  def astensor(a : Number)
    Tensor.new(a)
  end
end
