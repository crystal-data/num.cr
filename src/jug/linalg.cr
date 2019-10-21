require "./*"
require "../libs/dtype"
require "../flask/*"
require "../strides/offsets"
require "../indexing/base"
require "../blas/*"
require "../linalg/*"

class Jug(T)
  def matmul(other : Jug(T))
    LL.matmul(self.clone, other)
  end

  def inv
    LA.inv(self.clone)
  end
end
