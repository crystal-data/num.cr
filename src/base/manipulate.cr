require "./base"
require "./flags"
require "./transform"
require "../core/exceptions"

module Num::Manipulate
  include Transform
  extend self

  def array_split(ary : BaseArray, ind : Int32, axis : Int32 = 0)
    ntotal = ary.shape[axis]
    neach, extras = ntotal.divmod(ind)
    sizes = [0] + [neach + 1] * extras + [neach] * (ind - extras)
    rt = 0
    sizes.map! do |el|
      tmp = rt
      rt += el
      el + tmp
    end
    splitter(ary, axis, ind, sizes)
  end

  def array_split(ary : BaseArray, ind : Array(Int32), axis : Int32 = 0)
    nsections = ind.size + 1
    div_points = [0] + ind + [ary.shape[axis]]
    splitter(ary, axis, nsections, div_points)
  end

  def split(ary : BaseArray, ind : Int32, axis : Int32 = 0)
    n = ary.shape[axis]
    if n % ind != 0
      raise Exceptions::ValueError.new("Array split does not result in an equal division")
    end
    array_split(ary, ind, axis)
  end

  def split(ary : BaseArray, ind : Array(Int32), axis : Int32 = 0)
    array_split(ary, ind, axis)
  end

  def hsplit(ary : BaseArray, ind)
    case ary.ndims
    when 1
      split(ary, ind, 0)
    else
      split(ary, ind, 1)
    end
  end

  def vsplit(ary : BaseArray, ind)
    unless ary.ndims >= 2
      raise Exceptions::ValueError.new("vsplit only works on arrays of 2 or more dimensions")
    end
    split(ary, ind, 0)
  end

  def repeat(ary : BaseArray, n : Int32)
    ret = ary.class.new([ary.size * n])
    uiter = ret.unsafe_iter
    repeat_inner(ary, n) do |i|
      uiter.next.value = i
    end
    ret
  end

  def repeat(ary : BaseArray, n, axis)
    newshape = ary.shape.dup
    newshape[axis] *= n
    ret = ary.class.new(newshape)
    uiter = ret.unsafe_axis_iter(axis)
    ary.yield_along_axis(axis) do |suba|
      n.times do |_|
        uiter.next[...] = suba
      end
    end
    ret
  end

  def tile(ary : BaseArray, n : Int32)
    if ary.ndims > 1
      d = [1] * (ary.ndims - 1) + [n]
    else
      d = [1]
    end
    tile_inner(ary, d)
  end

  def tile(ary : BaseArray, n : Array(Int32))
    d = n.size
    if d < ary.ndims
      n = [1] * (ary.ndims - d) + n
    end
    tile_inner(ary, n)
  end

  def flip(ary : BaseArray)
    indexer = [{..., -1}] * ary.ndims
    ary.slice(indexer)
  end

  def flip(ary : BaseArray, axis : Int32)
    indexer = (0...ary.ndims).map_with_index { |e, i| i == axis ? {..., -1} : (...) }
    ary.slice(indexer)
  end

  private def tile_inner(ary, reps)
    shape_out = ary.shape.zip(reps).map { |i, j| i * j }
    n = ary.size
    if n > 0
      ary.shape.zip(reps) do |dim, nrep|
        if nrep != 1
          ary = repeat(ary.reshape(-1, n), nrep, 0)
        end
        n //= dim
      end
    end
    ary.reshape(shape_out)
  end

  private def splitter(ary : BaseArray(U), axis : Int32, n : Int32, div_points : Array(Int32)) forall U
    sub_arys = [] of BaseArray(U)
    sary = swapaxes(ary, axis, 0)
    n.times do |i|
      st = div_points[i]
      en = div_points[i + 1]
      sub_arys << swapaxes(sary[st...en], axis, 0)
    end
    sub_arys
  end

  private def repeat_inner(a, n)
    a.flat_iter.each do |el|
      n.times do |_|
        yield el.value
      end
    end
  end
end
