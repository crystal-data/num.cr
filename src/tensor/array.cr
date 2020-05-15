require "./tensor"

class Array(T)
  def to_tensor
    Tensor.from_array(self)
  end
end
