require "./base"
require "./flags"
require "../core/exceptions"

module Num::Transform
  extend self

  # Duplicates a BaseArray, respecting the passed order of memory
  # provided.  Useful for throwing `Tensor`s down to LAPACK
  # since they must be in Fortran style order
  #
  # ```
  # t = B.arange(5)
  # t.dup # => Tensor([0, 1, 2, 3, 4])
  # ```
  def dup(arr : BaseArray, order : Char? = nil)
    contig = uninitialized Internal::ArrayFlags
    case order
    when 'C'
      contig = Internal::ArrayFlags::Contiguous
    when 'F'
      contig = Internal::ArrayFlags::Fortran
    when nil
      contig = arr.flags & (Internal::ArrayFlags::Contiguous | Internal::ArrayFlags::Fortran)
    else
      raise Exceptions::ValueError.new(
        "Invalid argument for order.  Valid options or 'C', or 'F'")
    end
    ret = arr.class.new(arr.shape, contig)
    if (contig & arr.flags != Internal::ArrayFlags::None)
      ret = arr.class.new(arr.shape, contig)
      arr.buffer.copy_to(ret.buffer, arr.size)
    else
      ret.flat_iter.zip(arr.flat_iter).each do |i, j|
        i.value = j.value
      end
    end
    ret.update_flags(Internal::ArrayFlags::Fortran | Internal::ArrayFlags::Contiguous)
    ret
  end

  # Shallow copies the `Tensor`.  Shape and strides are copied, but
  # the underlying data is not.  The returned `Tensor` does
  # not own its own data, and its base reflects that.
  def dup_view(arr : BaseArray)
    newshape = arr.shape.dup
    newstrides = arr.strides.dup
    newflags = arr.flags.dup
    newflags &= ~Internal::ArrayFlags::OwnData
    newbase = arr.base ? arr.base : arr
    arr.class.new(arr.buffer, newshape, newstrides, newflags, newbase)
  end

  # Returns a view of the diagonal of a `Tensor`  only valid if
  # the `Tensor` has two dimensions.  Offsets are not supported.
  #
  # ```
  # t = Tensor.new([3, 3]) { |i| i }
  # t.diag_view # => Tensor([0, 4, 8])
  def diag_view(arr : BaseArray)
    raise Exceptions::ShapeError.new("Array must be two-dimensional") unless arr.ndims == 2
    nel = arr.shape.min
    newshape = [nel]
    newstrides = [arr.strides.sum]
    newflags = arr.flags.dup
    newflags &= ~Internal::ArrayFlags::OwnData
    newbase = arr.base ? arr.base : arr
    ret = arr.class.new(arr.buffer, newshape, newstrides, newflags, newbase)
    ret.update_flags(Internal::ArrayFlags::Fortran | Internal::ArrayFlags::Contiguous)
    ret
  end

  # Fits a `Tensor` into a new shape, not
  # altering memory if possible.  However, the `Tensor` is
  # usually copied.
  #
  # ```
  # t = Tensor.new([2, 4, 3]])
  #
  # t.reshape([2, 2, 2, 3]) # =>
  # Tensor([[[[ 0,  1,  2],
  #           [ 6,  7,  8]],
  #
  #          [[ 3,  4,  5],
  #           [ 9, 10, 11]]],
  #
  #
  #         [[[12, 13, 14],
  #           [18, 19, 20]],
  #
  #          [[15, 16, 17],
  #           [21, 22, 23]]]])
  # ```
  def reshape(arr : BaseArray, newshape : Array(Int32))
    if newshape == arr.shape
      return arr
    end
    newsize = 1
    cur_size = arr.size
    autosize = -1
    newshape.each_with_index do |val, i|
      if val < 0
        if autosize >= 0
          raise Exceptions::ValueError.new("Only shape dimension can be automatic")
        end
        autosize = i
      else
        newsize *= val
      end
    end

    if autosize >= 0
      newshape = newshape.dup
      newshape[autosize] = cur_size // newsize
      newsize *= newshape[autosize]
    end

    if newsize != cur_size
      raise Exceptions::ShapeError.new "Shapes #{arr.shape} cannot be reshaped to #{newshape}"
    end

    stride = uninitialized Int32
    newstrides = [0] * newshape.size
    newbase = arr.base ? arr.base : arr
    newdims = newshape.size

    if arr.flags & Internal::ArrayFlags::Contiguous
      stride = 1
      newdims.times do |i|
        newstrides[newdims - i - 1] = stride
        stride *= newshape[newdims - i - 1]
      end
    else
      stride = 1
      newshape.each_with_index do |d, i|
        newstrides[i] = stride
        stride *= d
      end
    end

    if arr.flags.fortran? || arr.flags.contiguous?
      newflags = arr.flags.dup
      newflags &= ~Internal::ArrayFlags::OwnData
      ret = arr.class.new(arr.buffer, newshape, newstrides, newflags, newbase)
      ret.update_flags(Internal::ArrayFlags::Fortran | Internal::ArrayFlags::Contiguous)
      ret
    else
      tmp = arr.dup
      ret = arr.class.new(tmp.buffer, newshape, newstrides, tmp.flags.dup, nil)
      ret.update_flags(Internal::ArrayFlags::Fortran | Internal::ArrayFlags::Contiguous)
      ret
    end
  end

  def reshape(arr : BaseArray, *args)
    reshape(arr, args.to_a)
  end

  # Flatten an array, returning a view if possible.
  # If the array is either fortran or c-contiguous, a view will be returned,
  #
  # otherwise, the array will be reshaped and copied.
  def ravel(arr : BaseArray)
    newshape = [arr.size]
    newflags = arr.flags.dup
    if arr.flags.contiguous?
      newstrides = [arr.strides[-1]]
      newbase = arr.base ? arr.base : arr
      newflags &= ~Internal::ArrayFlags::OwnData
      arr.class.new(arr.buffer, newshape, newstrides, newflags, newbase)
    elsif arr.flags.fortran?
      newstrides = [arr.strides[0]]
      newbase = arr.base ? arr.base : arr
      newflags &= ~Internal::ArrayFlags::OwnData
      arr.class.new(arr.buffer, newshape, newstrides, newflags, newbase)
    else
      reshape(arr, [-1])
    end
  end

  def astype(arr : BaseArray, dtype : U.class) forall U
    ret = arr.basetype(U).new(arr.shape)
    {% if U == Bool %}
      ret.flat_iter.zip(arr.flat_iter) do |i, j|
        i.value = (j.value != 0) && (!!j.value)
      end
    {% else %}
      ret.flat_iter.zip(arr.flat_iter) do |i, j|
        i.value = U.new(j.value)
      end
    {% end %}
    ret
  end

  # Permute the dimensions of a `Tensor`.  If no order is provided,
  # the dimensions will be reversed, a "true transpose".  Otherwise,
  # the dimensions will be permutated in the order provided.
  #
  # ```
  # t = Tensor.new([2, 4, 3]) { |i| i }
  # t.transpose([2, 0, 1])
  # Tensor([[[ 0,  3,  6,  9],
  #          [12, 15, 18, 21]],
  #
  #         [[ 1,  4,  7, 10],
  #          [13, 16, 19, 22]],
  #
  #         [[ 2,  5,  8, 11],
  #          [14, 17, 20, 23]]])
  # ```
  def transpose(arr : BaseArray, order : Array(Int32) = [] of Int32)
    newshape = arr.shape.dup
    newstrides = arr.strides.dup
    newbase = arr.base ? arr.base : arr
    if order.size == 0
      order = (0...arr.ndims).to_a.reverse
    end
    n = order.size
    if n != arr.ndims
      raise "Axes don't match array"
    end

    permutation = [0] * 32
    reverse_permutation = [0] * 32
    n.times do |i|
      reverse_permutation[i] = -1
    end

    n.times do |i|
      axis = order[i]
      if axis < 0
        axis = arr.ndims + axis
      end
      if axis < 0 || axis >= arr.ndims
        raise "Invalid axis for this array"
      end
      if reverse_permutation[axis] != -1
        raise "Repeated axis in transpose"
      end
      reverse_permutation[axis] = i
      permutation[i] = axis
    end

    n.times do |i|
      newshape[i] = arr.shape[permutation[i]]
      newstrides[i] = arr.strides[permutation[i]]
    end
    ret = arr.class.new(arr.buffer, newshape, newstrides, arr.flags.dup, newbase)
    ret.update_flags(Internal::ArrayFlags::Contiguous | Internal::ArrayFlags::Fortran)
    ret
  end

  def transpose(arr : BaseArray, *args)
    transpose(arr, args.to_a)
  end

  def swapaxes(arr : BaseArray, axis1 : Int32, axis2 : Int32)
    order = (0...arr.ndims).to_a
    order[axis1] = axis2
    order[axis2] = axis1
    transpose(arr, order)
  end
end
