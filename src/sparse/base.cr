require "../tensor/tensor"

abstract struct SparseBase(T)
  @shape : Array(Int32)
  @ndim : Int32 = 2
  @nnz : Int32 = 0
  @data : Tensor(T)
end
