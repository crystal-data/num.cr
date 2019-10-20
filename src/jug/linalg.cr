require "./*"
require "../libs/dtype"
require "../flask/*"
require "../strides/offsets"
require "../indexing/base"
require "../blas/*"

class Jug(T)
  def matmul(other : Jug(T))
    LL.matmul(self.clone, other)
  end
end
