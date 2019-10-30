require "./ndtensor"

def add(a : NDTensor, x : Number)
  step = a.strides[-1]
  ptr = a.@ptr
  NDTensor.new(a.shape.dims) do |i|
    ptr[i * step] + x
  end
end
