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
  # :nodoc:
  protected def strides_for_broadcast(shape : Array(Int), strides : Array(Int), output_shape : Array(Int))
    output_strides = Num::Internal.shape_to_strides(shape, Num::RowMajor)
    Num::Internal.broadcast_strides(output_shape, shape, output_strides, strides)
  end

  # :nodoc:
  protected def strides_for_reshape(s0 : Array(Int), s1 : Array(Int))
    size0 = s0.product
    size1 = 1
    auto = -1

    s1.each_with_index do |axis, index|
      if axis < 0
        raise Num::Exceptions::ValueError.new(
          "Only a single dimension can be automatic") if auto >= 0
        auto = index
      else
        size1 *= axis
      end
    end

    if auto >= 0
      s1 = s1.dup
      s1[auto] = size0 // size1
      size1 *= s1[auto]
    end

    if size0 != size1
      raise Num::Exceptions::ValueError.new "Shapes #{s0} cannot be reshaped to #{s1}"
    end

    {s1, Num::Internal.shape_to_strides(s1, Num::RowMajor)}
  end

  # :nodoc:
  protected def shape_and_strides_for_transpose(shape : Array(Int), strides : Array(Int), axes : Array(Int))
    order = axes.map &.to_i
    new_shape = shape.dup
    new_strides = strides.dup
    rank = shape.size

    if order.size == 0
      order = (0...rank).to_a.reverse
    end

    n = order.size
    if n != rank
      raise "Axes do not match Tensor"
    end

    perm = [0] * rank
    r_perm = [-1] * rank

    n.times do |i|
      axis = order[i]
      if axis < 0
        axis = rank + axis
      end
      if axis < 0 || axis >= rank
        raise Num::Exceptions::ValueError.new "Invalid axis for Tensor"
      end
      if r_perm[axis] != -1
        raise Num::Exceptions::ValueError.new "Repeated axis in transpose"
      end
      r_perm[axis] = i
      perm[i] = axis
    end

    n.times do |i|
      new_shape[i] = shape[perm[i]]
      new_strides[i] = strides[perm[i]]
    end

    {new_shape, new_strides}
  end

  # :nodoc:
  protected def swap_axes_for_transpose(rank : Int, a : Int, b : Int)
    axes = (0...rank).to_a
    tmp = axes[a]
    axes[a] = axes[b]
    axes[b] = tmp
    axes
  end

  # :nodoc:
  protected def move_axes_for_transpose(rank : Int, source : Array(Int), destination : Array(Int))
    axes = (0...rank).to_a
    source.zip(destination) do |i, j|
      axes[i] = j
    end
    axes
  end

  # :nodoc:
  protected def assert_min_dimension(ts : Array(Tensor), min : Int)
    unbounded = ts.any? do |t|
      t.rank < min
    end
    if unbounded
      raise Num::Exceptions::ValueError.new "Wrong number of dimensions"
    end
  end

  # :nodoc:
  protected def all_shapes_equal(shapes : Array(Array(Int)))
    s0 = shapes[0]
    shapes[1...].each do |s|
      unless s0 == s
        raise Num::Exceptions::ValueError.new "All inputs must share a shape"
      end
    end
  end

  # :nodoc:
  protected def clip_axis(axis, size)
    if axis < 0
      axis += size
    end
    if axis < 0 || axis > size
      raise Num::Exceptions::ValueError.new "Axis out of range"
    end
    axis
  end

  # :nodoc:
  def concat_shape(ts : Array(Tensor), axis : Int, shape : Array(Int))
    rank = shape.size
    ts.each do |t|
      if t.rank != rank
        raise Num::Exceptions::ValueError.new "All inputs must share the same dimensions"
      end

      rank.times do |i|
        if i != axis && t.shape[i] != shape[i]
          raise Num::Exceptions::ValueError.new "All inputs must share a shape off-axis"
        end
      end
      shape[axis] += t.shape[axis]
    end
    shape
  end

  # :nodoc:
  protected def shape_for_broadcast(*args : Tensor)
    nd = (args.map &.rank).max
    shape = [0] * nd

    args.each_with_index do |arg, i|
      d = nd - arg.rank
      t_shape = [1] * d + arg.shape
      shape = shape.map_with_index do |e, j|
        e > t_shape[j] ? e : t_shape[j]
      end
    end
    shape
  end

  # :nodoc:
  protected def repeat_inner(a : Tensor, n : Int)
    a.each do |el|
      n.times do
        yield el
      end
    end
  end

  # :nodoc:
  protected def tile_inner(a : Tensor, r : Array(Int))
    shape_out = a.shape.zip(r).map do |i, j|
      (i * j).to_i
    end
    n = a.size
    a.shape.zip(r) do |d, rep|
      if rep != 1
        a = a.reshape(-1, n).repeat(rep, 0)
      end
      n //= d
    end
    a.reshape(shape_out)
  end
end
