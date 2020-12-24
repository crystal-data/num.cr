# Copyright (c) 2020 Crystal Data Contributors
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

module Num::Internal
  extend self

  # :nodoc:
  def broadcastable(a : Num::Backend::Storage, b : Num::Backend::Storage)
    return [] of Int32 unless a.shape != b.shape
    a_size = a.rank
    b_size = b.rank

    if a_size == b_size
      if broadcast_equal(a.shape, b.shape)
        return broadcastable_shape(arr.shape, other.shape)
      end
    else
      if a_size > b_size
        shape = [1] * (a_size - b_size) + b.shape
        if broadcast_equal(a.shape, shape)
          return broadcastable_shape(a.shape, shape)
        end
      else
        shape = [1] * (osz - sz) + a.shape
        if broadcast_equal(shape, b.shape)
          return broadcastable_shape(shape, b.shape)
        end
      end
    end
    raise "Shapes #{arr.shape} and #{other.shape} are not broadcastable"
  end

  # :nodoc:
  def broadcast_to(a : Num::Backend::Storage, shape : Array(Int32))
    dim = shape.size
    strides = [0] * dim
    size = 1
    dim.times do |i|
      strides[dim - i - 1] = size
      size *= shape[dim - i - 1]
    end

    new_strides = broadcast_strides(
      shape,
      a.shape,
      strides,
      a.strides
    )

    a.class.new(a.data, shape, new_strides, shape.product)
  end

  # :nodoc:
  def broadcast(a : Num::Backend::Storage(U), b : Num::Backend::Storage(V)) forall U, V
    t = {a, b}
    if a.shape == b.shape
      return t
    end

    nd = t.max_of do |i|
      i.rank
    end
    shape = [0] * nd

    2.times do |i|
      d = nd - t[i].rank
      t_shape = [1] * d + t[i].shape
      shape = shape.map_with_index do |e, j|
        e > t_shape[j] ? e : t_shape[j]
      end
    end

    return {bcast_if(t[0], shape), bcast_if(t[1], shape)}
  end

  # :nodoc:
  def broadcast(a : Num::Backend::Storage(U), b : Num::Backend::Storage(V), c : Num::Backend::Storage(W)) forall U, V, W
    t = {a, b, c}
    if a.shape == b.shape && b.shape == c.shape
      return t
    end

    nd = t.max_of do |i|
      i.rank
    end
    shape = [0] * nd

    3.times do |i|
      d = nd - t[i].rank
      t_shape = [1] * d + t[i].shape
      shape = shape.map_with_index do |e, j|
        e > t_shape[j] ? e : t_shape[j]
      end
    end

    {bcast_if(t[0], shape), bcast_if(t[1], shape), bcast_if(t[2], shape)}
  end

  private def bcast_if(item : Num::Backend::Storage, shape : Array(Int32))
    shape == item.shape ? item : broadcast_to(item, shape)
  end

  private def broadcast_strides(dest_shape, src_shape, dest_strides, src_strides)
    dims = dest_shape.size
    start = dims - src_shape.size

    ret = [0] * dims
    (dims - 1).step(to: start, by: -1) do |i|
      s = src_shape[i - start]
      case s
      when 1
        ret[i] = 0
      when dest_shape[i]
        ret[i] = src_strides[i - start]
      else
        raise "Cannot broadcast from #{src_shape} to #{dest_shape}"
      end
    end
    ret
  end

  private def broadcast_equal(a, b)
    bc = true
    a.zip(b) do |i, j|
      if !(i == j || i == 1 || j == 1)
        bc = false
      end
    end
    bc
  end

  private def broadcastable_shape(a, b)
    a.zip(b).map do |i|
      Math.max(i[0], i[1])
    end
  end
end
