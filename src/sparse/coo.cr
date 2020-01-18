require "../tensor/tensor"

# struct Num::Sparse::CooMatrix(T)
#   getter shape : Array(Int32)
#   getter ndims : Int32 = 2
#   getter nnz : Int32 = 0
#   getter data : Tensor(T)
#   getter row : Tensor(Int32)
#   getter col : Tensor(Int32)
#
#   def initialize(@shape)
#     @data = Tensor(T).new([] of Int32)
#     @row = Tensor(Int32).new([] of Int32)
#     @col = Tensor(Int32).new([] of Int32)
#   end
#
#   def initialize(@data : Tensor(T), idx : Tuple(Tensor(Int32), Tensor(Int32)), @shape)
#     @row, @col = idx
#     if @data.ndims != 1 || @row.ndims != 1 || @col.ndims != 1
#       raise Exceptions::ShapeError.new("All inputs must be one dimensional")
#     end
#     if @data.shape != @row.shape || @data.shape != @col.shape
#       raise Exceptions::ShapeError.new("All inputs must be the same size")
#     end
#     @nnz = @data.size
#   end
#
#   def totensor
#     ret = Tensor(T).new(@shape)
#     row.flat_iter.zip(col.flat_iter, data.flat_iter) do |i, j, el|
#       ret[i.value, j.value] = el.value
#     end
#     ret
#   end
# end
