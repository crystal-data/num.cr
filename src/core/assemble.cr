require "./tensor"
require "./matrix"

# Module for handling data joining operations
# along Tensor's and Matrices.
#
# All of these operations produce copies of data,
# so large memory costs may be incured on large
# containers
module Bottle::Internal::Assemble
  extend self

  # Join an Array of tensors along an existing axis.
  #
  # ```
  # t1 = Tensor.new [1, 2]
  # t2 = Tensor.new [1, 2, 3]
  #
  # concatenate(t1, t2) # => Tensor[  1  2  1  2  3]
  # ```
  def concatenate(tensors : Array(Tensor(U))) forall U
    sizes = tensors.map { |e| e.size }
    total = sizes.reduce { |i, j| i + j }

    buffer = Pointer(U).malloc(total)
    offset = 0

    tensors.each do |t|
      t = t.clone
      (buffer + offset).copy_from(t.@buffer, t.size)
      offset += t.size
    end

    Tensor.new(buffer, total, 1, true)
  end

  # Join an array of matrices as a flattened tensor.
  #
  # ```
  # m1 = Matrix.new [[1, 2], [3, 4]]
  # m2 = Matrix.new [[1, 2]]
  #
  # concatenate(m1, m2) # => Tensor[  1  2  3  4  1  2]
  # ```
  def concatenate(matrices : Array(Matrix(U)), axis : Nil = nil) forall U
    sizes = matrices.map { |e| e.nrows * e.ncols }

    total = sizes.reduce { |i, j| i + j }

    buffer = Pointer(U).malloc(total)
    offset = 0

    matrices.each do |m|
      m = m.clone
      (buffer + offset).copy_from(m.@buffer, m.nrows * m.ncols)
      offset += m.nrows * m.ncols
    end

    Tensor.new(buffer, total, 1, true)
  end

  # Join an array of matrices along an existing axis
  #
  # ```
  # m1 = Matrix.new [[1, 2], [3, 4]]
  # m2 = Matrix.new [[1, 2]]
  #
  # concatenate(m1, m2, axis: 0) # =>
  #
  # # Matrix[[  1  2]
  # #        [  3  4]
  # #        [  1  2]]
  # ```
  def concatenate(matrices : Array(Matrix(U)), axis : Int32) forall U
    if axis == 1
      matrices = matrices.map { |e| e.transpose }
    end

    sizes = matrices.map { |e| e.nrows * e.ncols }
    total = sizes.reduce { |i, j| i + j }

    lim = [] of Int32

    lim = matrices.map { |m| m.ncols }

    if lim.all? { |l| l != lim[0] }
      raise "Shapes cannot be aligned"
    end

    buffer = Pointer(U).malloc(total)
    offset = 0

    matrices.each do |m|
      m = m.clone
      (buffer + offset).copy_from(m.@buffer, m.nrows * m.ncols)
      offset += m.nrows * m.ncols
    end

    cols = lim[0]
    rows = total // cols

    ret = Matrix.new(buffer, rows, cols, cols, true)

    if axis == 0
      ret
    else
      ret.transpose
    end
  end
end

include Bottle::Internal::Assemble
include Bottle

m1 = Matrix.new [[1, 2], [3, 4]]
m2 = Matrix.new [[1, 2]]
puts concatenate([m1, m2], axis: 1)
