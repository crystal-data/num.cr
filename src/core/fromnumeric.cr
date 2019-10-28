require "./tensor"
require "./matrix"

# A module primarily responsible for `Tensor`
# and `Matrix` manipulation routines.
#
# This module should be namespaced as part of the
# external API to provide user facing methods
# for creation.
module Bottle::Internal::FromNumeric
  extend self

  # Gives a new shape to an `Matrix`
  # without changing its data.
  #
  # Currently, this method always copies data.
  # Eventually, it would be ideal to only modify attributes
  # of an array if the array is contiguous
  #
  # ```
  # m = Matrix.new [[1, 2, 3], [4, 5, 6]]
  #
  # reshape(m, 3, 2) # => [[1, 2], [3, 4], [5, 6]]
  # ```
  def reshape(a : Matrix, rows : Int32, cols : Int32)
    if (a.nrows * a.ncols) != (rows * cols)
      raise "Matrix of size #{a.nrows},#{a.ncols} cannot go into shape #{rows}, #{cols}"
    end

    Matrix.new(rows, cols) do |i, j|
      idx = i * cols + j
      i = idx // a.ncols
      j = idx % a.ncols
      a[i, j]
    end
  end

  # Gives a new shape to an `Tensor`
  # without changing its data.  This must always return
  # a `Matrix`
  #
  # Currently, this method always copies data.
  # Eventually, it would be ideal to only modify attributes
  # of an array if the array is contiguous
  #
  # ```
  # t = Tensor.new [1, 2, 3, 4, 5, 6]
  #
  # reshape(t, 3, 2) # => [[1, 2], [3, 4], [5, 6]]
  # ```
  def reshape(a : Tensor, rows : Int32, cols : Int32)
    if a.size != (rows * cols)
      raise "Tensor of size #{a.size} cannot go into shape #{rows}, #{cols}"
    end

    Matrix.new(rows, cols) do |i, j|
      idx = i * cols + j
      a[idx]
    end
  end

  # Transposes a matrix.  Currently always copies data
  # since F ordered arrays are not supported by Bottle.
  #
  # ```
  # m = Matrix.new [[1, 2, 3], [3, 4, 5]]
  # transpose(m) # => [[1, 3], [2, 4], [3, 5]]
  # ```
  def transpose(a : Matrix)
    Matrix.new(a.ncols, a.nrows) do |i, j|
      a[j, i]
    end
  end

  # Convert inputs to arrays with at least one dimension.
  #
  # Scalar inputs are converted to 1-dimensional `Tensor`s, whilst
  # higher-dimensional inputs are preserved.
  #
  # ```
  # atleast_1d(5) # => Tensor [  5]
  #
  # atleast_1d(Tensor.new [1, 2, 3]) # => Tensor[  1  2  3
  #
  # atleast_1d(Matrix.new [[1, 2], [3, 4]]) # => Matrix[[  1  2] [3  4]]
  # ```
  def atleast_1d(x : Number)
    Tensor.new [x]
  end

  # Convert inputs to arrays with at least one dimension.
  #
  # Scalar inputs are converted to 1-dimensional `Tensor`s, whilst
  # higher-dimensional inputs are preserved.
  #
  # ```
  # atleast_1d(5) # => Tensor [  5]
  #
  # atleast_1d(Tensor.new [1, 2, 3]) # => Tensor[  1  2  3
  #
  # atleast_1d(Matrix.new [[1, 2], [3, 4]]) # => Matrix[[  1  2] [3  4]]
  # ```
  def atleast_1d(x : Tensor)
    x
  end

  # Convert inputs to arrays with at least one dimension.
  #
  # Scalar inputs are converted to 1-dimensional `Tensor`s, whilst
  # higher-dimensional inputs are preserved.
  #
  # ```
  # atleast_1d(5) # => Tensor [  5]
  #
  # atleast_1d(Tensor.new [1, 2, 3]) # => Tensor[  1  2  3
  #
  # atleast_1d(Matrix.new [[1, 2], [3, 4]]) # => Matrix[[  1  2] [3  4]]
  # ```
  def atleast_1d(x : Matrix)
    x
  end

  # Convert inputs to arrays with at least two dimensions.
  #
  # Scalar inputs are converted to 2-dimensional `Matrix`s, whilst
  # higher-dimensional inputs are preserved.
  #
  # ```
  # atleast_1d(5) # => Matrix [[  5]]
  #
  # atleast_1d(Tensor.new [1, 2, 3]) # => Matrix[[  1  2  3]]
  #
  # atleast_1d(Matrix.new [[1, 2], [3, 4]]) # => Matrix[[  1  2] [3  4]]
  # ```
  def atleast_2d(x : Number)
    Matrix.new [[x]]
  end

  # Convert inputs to arrays with at least two dimensions.
  #
  # Scalar inputs are converted to 2-dimensional `Matrix`s, whilst
  # higher-dimensional inputs are preserved.
  #
  # ```
  # atleast_1d(5) # => Matrix [[  5]]
  #
  # atleast_1d(Tensor.new [1, 2, 3]) # => Matrix[[  1  2  3]]
  #
  # atleast_1d(Matrix.new [[1, 2], [3, 4]]) # => Matrix[[  1  2] [3  4]]
  # ```
  def atleast_2d(x : Tensor)
    reshape(x, 1, x.size)
  end

  # Convert inputs to arrays with at least two dimensions.
  #
  # Scalar inputs are converted to 2-dimensional `Matrix`s, whilst
  # higher-dimensional inputs are preserved.
  #
  # ```
  # atleast_1d(5) # => Matrix [[  5]]
  #
  # atleast_1d(Tensor.new [1, 2, 3]) # => Matrix[[  1  2  3]]
  #
  # atleast_1d(Matrix.new [[1, 2], [3, 4]]) # => Matrix[[  1  2] [3  4]]
  # ```
  def atleast_2d(x : Matrix)
    x
  end
end
