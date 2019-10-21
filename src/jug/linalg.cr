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

  def tril(k = 0)
    Jug(T).new(nrows, ncols) do |i, j|
      i < j - k ? self[i, j] : T.new(0)
    end
  end

  def triu(k = 0)
    Jug(T).new(nrows, ncols) do |i, j|
      i > j - k ? self[i, j] : T.new(0)
    end
  end

  def self.identity(n : Int32)
    one = LL.astype(1, T)
    zero = LL.astype(0, T)
    Jug(T).new(n, n) do |i, j|
      i == j ? T.new(1) : T.new(0)
    end
  end

  def diagonal
    n = Math.min(nrows, ncols)
    Flask(T).new(n) { |i| self[i, i] }
  end

  def trace
    diagonal.sum
  end
end
