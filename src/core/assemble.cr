require "./macros"
require "./exceptions"
require "./common"
require "./converters"
require "../tensor/creation"
require "../base/base"

module Bottle::Assemble
  include Internal
  include Creation
  include Convert
  extend self

  # Concatenates an array of `Tensor's` along a provided axis.
  #
  # Parameters
  # ----------
  # alist : Array(Tensor)
  #   - Array containing Tensors to be concatenated
  # axis : Int32
  #   - Axis for concatentation, must be an existing
  #     axis present in all Tensors, and all Tensors
  #     must have the same shape off-axis
  #
  # Returns
  # -------
  # ret : Tensor
  #   - Result of the concatenation
  #
  # Examples
  # --------
  # ```
  # t = Tensor.new([2, 2, 3]) { |i| i }
  #
  # concatenate([t, t, t], axis=-1)
  #
  # Tensor([[[ 0,  1,  2,  0,  1,  2,  0,  1,  2],
  #          [ 3,  4,  5,  3,  4,  5,  3,  4,  5]],
  #
  #         [[ 6,  7,  8,  6,  7,  8,  6,  7,  8],
  #          [ 9, 10, 11,  9, 10, 11,  9, 10, 11]]])
  # ```
  def concatenate(alist : Array(BaseArray(U)), axis : Int32) forall U
    raise_zerod alist
    newshape = alist[0].shape.dup
    clipaxis axis, newshape.size
    newshape[axis] = 0
    shape = assert_shape_off_axis(alist, axis, newshape)
    ret = alist[0].class.new(newshape)
    lo = [0] * newshape.size
    hi = shape.dup
    hi[axis] = 0
    alist.each do |a|
      if a.shape[axis] != 0
        hi[axis] += a.shape[axis]
        ranges = lo.zip(hi).map { |i, j| i...j }
        ret[ranges] = a
        lo[axis] = hi[axis]
      end
    end
    ret
  end

  def concatenate(alist : Array(BaseArray(U))) forall U
    totalsize = alist.reduce(0) { |i, j| i + j.size }
    ret = alist[0].class.new([totalsize])
    offset = 0
    alist.each do |a|
      ret[offset...(offset + a.size)] = a
      offset += a.size
    end
    ret
  end

  # Concatenates a list of `Tensor`s along axis 0
  #
  # Parameters
  # ----------
  # alist : Array(Tensor)
  #   - Array containing Tensors to be stacked
  #
  # Returns
  # -------
  # ret : Tensor
  #   - Result of the concatenation
  #
  # Examples
  # --------
  # ```
  # t = Tensor.new([2, 2, 3])
  # vstack([t, t, t])
  #
  # Tensor([[[ 0,  1,  2],
  #          [ 3,  4,  5]],
  #
  #         [[ 6,  7,  8],
  #          [ 9, 10, 11]],
  #
  #         [[ 0,  1,  2],
  #          [ 3,  4,  5]],
  #
  #         [[ 6,  7,  8],
  #          [ 9, 10, 11]],
  #
  #         [[ 0,  1,  2],
  #          [ 3,  4,  5]],
  #
  #         [[ 6,  7,  8],
  #          [ 9, 10, 11]]])
  # ```
  def vstack(alist : Array(BaseArray(U))) forall U
    alist = alist.map { |t| atleast_2d(t) }
    concatenate(alist, 0)
  end

  # Concatenates a list of `Tensor`s along axis 1
  #
  # Parameters
  # ----------
  # alist : Array(Tensor)
  #   - Array containing Tensors to be stacked
  #
  # Returns
  # -------
  # ret : Tensor
  #   - Result of the concatenation
  #
  # Examples
  # --------
  # ```
  # t = Tensor.new([2, 2, 3])
  # hstack([t, t, t])
  #
  # Tensor([[[ 0,  1,  2],
  #          [ 3,  4,  5],
  #          [ 0,  1,  2],
  #          [ 3,  4,  5],
  #          [ 0,  1,  2],
  #          [ 3,  4,  5]],
  #
  #         [[ 6,  7,  8],
  #          [ 9, 10, 11],
  #          [ 6,  7,  8],
  #          [ 9, 10, 11],
  #          [ 6,  7,  8],
  #          [ 9, 10, 11]]])
  # ```
  def hstack(alist : Array(BaseArray(U))) forall U
    assert_all_1d alist
    if alist.all? { |t| t.ndims == 1 }
      concatenate(alist)
    else
      concatenate(alist, 1)
    end
  end

  def dstack(alist : Array(BaseArray(U))) forall U
    assert_all_1d alist
    first = alist[0]
    shape = first.shape
    assert_shape(shape, alist)

    case first.ndims
    when 1
      alist = alist.map do |a|
        a.reshape([1, a.size, 1])
      end
      concatenate(alist, 2)
    when 2
      alist = alist.map do |a|
        a.reshape(a.shape + [1])
      end
      concatenate(alist, 2)
    else
      raise ShapeError.new("dstack was given arrays with more than two dimensions")
    end
  end

  def column_stack(alist : Array(BaseArray(U))) forall U
    assert_all_1d alist
    first = alist[0]
    shape = first.shape
    assert_shape(shape, alist)

    case first.ndims
    when 1
      alist = alist.map do |a|
        a.reshape([a.size, 1])
      end
      concatenate(alist, 1)
    when 2
      concatenate(alist, 1)
    else
      raise ShapeError.new("dstack was given arrays with more than two dimensions")
    end
  end

  def atleast_1d(inp : Number)
    Tensor.new([1]) { |_| inp }
  end

  def atleast_1d(inp : Tensor)
    if inp.ndims == 0
      Tensor.new([1]) { |_| inp.value }
    else
      inp
    end
  end

  def atleast_2d(inp : Number)
    Tensor.new([1, 1]) { |_| inp }
  end

  def atleast_2d(inp : Array)
    t = Tensor.from_array inp
    atleast_2d(t)
  end

  def atleast_2d(inp : Tensor)
    if inp.ndims > 1
      inp
    elsif inp.ndims == 0
      Tensor.new([1, 1]) { |_| inp.value }
    else
      inp.reshape([1, inp.size])
    end
  end

  def atleast_3d(inp : Number)
    Tensor.new([1, 1, 1]) { |_| inp }
  end

  def atleast_3d(inp : Tensor)
    if inp.ndims > 2
      inp
    else
      dim = 3 - inp.ndims
      newshape = [1] * dim + inp.shape
      inp.reshape(newshape)
    end
  end

  def kron(a : Tensor, b : Tensor)
    o = multiply_outer(a, b)
    l1 = (0...o.shape[0]).map { |i| o[i] }
    o2 = concatenate(l1, 1)
    l2 = (0...o2.shape[0]).map { |i| o2[i] }
    return concatenate(l2, 1)
  end

  def block_diag(*arrs)
    arrs = arrs.to_a.map { |a| B.atleast_2d(a) }
    shapes = arrs.to_a.map { |a| a.shape }
    m, n = Tensor.from_array(shapes).sum(0)
    r, c = 0, 0
    ret = B.zeros([m.value, n.value])
    shapes.each_with_index do |(rr, cc), i|
      ret[r...(r + rr), c...(c + cc)] = arrs[i]
      r += rr
      c += cc
    end
    ret
  end

  def helmert(n, full = false)
    dg = arange(n)
    z = zeros([n, n])
    z.diag_view[...] = dg
    h = tril(ones([n, n]), -1) - z
    d = arange(n) * arange(1, n + 1)
    h[0] = 1
    d[0] = n
    h_full = h / sqrt(d).bc?(1)
    if full
      h_full
    else
      h_full[1...]
    end
  end

  def hankel(c, r = nil)
    c = astensor(c).ravel
    if r.nil?
      ur = zeros_like(c, dtype: c.dtype)
    else
      ur = astensor(r).ravel.astype(c.dtype)
    end

    vals = concatenate([c, ur[1...]])
    out_shp = [c.size, ur.size]
    n = vals.strides[0]
    vals.as_strided(out_shp, [n, n]).dup
  end

  def hadamard(n, dtype : U.class = Int32) forall U
    if n < 1
      lg2 = 0
    else
      lg2 = Int32.new(Math.log(n, 2))
    end

    h = atleast_2d(1).astype(dtype)

    lg2.times do |_|
      h = vstack([hstack([h, h]), hstack([h, -1 * h])])
    end
    h
  end

  def leslie(f, s)
    f = atleast_1d(f)
    s = atleast_1d(s).astype(f.dtype)

    tmp = f[0] + s[0]
    n = f.size
    a = zeros([n, n], dtype: tmp.dtype)
    a[0] = f
    (1...n).zip(0...(n - 1)) do |i, j|
      puts i, j
    end
  end
end
