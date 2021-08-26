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

module Num::Iteration
  extend self

  macro init_strided_iteration(coord, backstrides, t_shape, t_strides, t_rank, t_data)
    {{ coord.id }} = Pointer(Int32).malloc({{ t_rank }}, 0)
    {{ backstrides.id }} = Pointer(Int32).malloc({{ t_rank }})
    {{ t_rank }}.times do |i|
      {{ backstrides.id }}[i] = {{ t_strides }}[i] * ({{ t_shape }}[i] - 1)
      if {{ t_strides }}[i] < 0
        {{ t_data }} += ({{ t_shape }}[i] - 1) * {{ t_strides }}[i].abs
      end
    end
  end

  macro advance_strided_iteration(coord, backstrides, t_shape, t_strides, t_rank, iter_pos)
    ({{ t_rank }} - 1).step(to: 0, by: -1) do |k|
      if {{ coord.id }}[k] < {{ t_shape }}[k] - 1
        {{ coord.id }}[k] += 1
        {{ iter_pos }} += {{ t_strides }}[k]
        break
      else
        {{ coord.id }}[k] = 0
        {{ iter_pos }} -= {{ backstrides.id }}[k]
      end
    end
  end

  @[AlwaysInline]
  def strided_iteration(t : Tensor)
    data = t.raw + t.offset
    if t.flags.contiguous?
      t.size.times do |i|
        yield i, data
        data += 1
      end
    else
      t_shape, t_strides, t_rank = t.shape, t.strides, t.rank
      init_strided_iteration(:coord, :backstrides, t_shape, t_strides, t_rank, data)
      t.size.times do |i|
        yield i, data
        advance_strided_iteration(:coord, :backstrides, t_shape, t_strides, t_rank, data)
      end
    end
  end

  @[AlwaysInline]
  def dual_strided_iteration(t1 : Tensor, t2 : Tensor)
    n = t1.size

    t1_contiguous = t1.flags.contiguous?
    t2_contiguous = t2.flags.contiguous?

    t1data = t1.raw + t1.offset
    t2data = t2.raw + t2.offset

    t1_shape, t1_strides, t1_rank = t1.shape, t1.strides, t1.rank
    t2_shape, t2_strides, t2_rank = t2.shape, t2.strides, t2.rank

    if t1_contiguous && t2_contiguous
      n.times do |i|
        yield i, t1data, t2data
        t1data += 1
        t2data += 1
      end
    elsif t1_contiguous
      init_strided_iteration(:t2_coord, :t2_backstrides, t2_shape, t2_strides, t2_rank, t2data)
      n.times do |i|
        yield i, t1data, t2data
        t1data += 1
        advance_strided_iteration(:t2_coord, :t2_backstrides, t2_shape, t2_strides, t2_rank, t2data)
      end
    elsif t2_contiguous
      init_strided_iteration(:t1_coord, :t1_backstrides, t1_shape, t1_strides, t1_rank, t1data)
      n.times do |i|
        yield i, t1data, t2data
        advance_strided_iteration(:t1_coord, :t1_backstrides, t1_shape, t1_strides, t1_rank, t1data)
        t2data += 1
      end
    else
      init_strided_iteration(:t1_coord, :t1_backstrides, t1_shape, t1_strides, t1_rank, t1data)
      init_strided_iteration(:t2_coord, :t2_backstrides, t2_shape, t2_strides, t2_rank, t2data)
      n.times do |i|
        yield i, t1data, t2data
        advance_strided_iteration(:t1_coord, :t1_backstrides, t1_shape, t1_strides, t1_rank, t1data)
        advance_strided_iteration(:t2_coord, :t2_backstrides, t2_shape, t2_strides, t2_rank, t2data)
      end
    end
  end

  @[AlwaysInline]
  def tri_strided_iteration(t1 : Tensor, t2 : Tensor, t3 : Tensor)
    n = t1.size

    t1_contiguous = t1.flags.contiguous?
    t2_contiguous = t2.flags.contiguous?
    t3_contiguous = t3.flags.contiguous?

    t1data = t1.raw + t1.offset
    t2data = t2.raw + t2.offset
    t3data = t3.raw + t3.offset

    t1_shape, t1_strides, t1_rank = t1.shape, t1.strides, t1.rank
    t2_shape, t2_strides, t2_rank = t2.shape, t2.strides, t2.rank
    t3_shape, t3_strides, t3_rank = t3.shape, t3.strides, t3.rank

    if t1_contiguous && t2_contiguous && t3_contiguous
      n.times do |i|
        yield i, t1data, t2data, t3data
        t1data += 1
        t2data += 1
        t3data += 1
      end
    elsif t1_contiguous && t2_contiguous
      init_strided_iteration(:t3_coord, :t3_backstrides, t3_shape, t3_strides, t3_rank, t3data)
      n.times do |i|
        yield i, t1data, t2data, t3data
        t1data += 1
        t2data += 1
        advance_strided_iteration(:t3_coord, :t3_backstrides, t3_shape, t3_strides, t3_rank, t3data)
      end
    elsif t1_contiguous
      init_strided_iteration(:t2_coord, :t2_backstrides, t2_shape, t2_strides, t2_rank, t2data)
      init_strided_iteration(:t3_coord, :t3_backstrides, t3_shape, t3_strides, t3_rank, t3data)
      n.times do |i|
        yield i, t1data, t2data, t3data
        t1data += 1
        advance_strided_iteration(:t2_coord, :t2_backstrides, t2_shape, t2_strides, t2_rank, t2data)
        advance_strided_iteration(:t3_coord, :t3_backstrides, t3_shape, t3_strides, t3_rank, t3data)
      end
    else
      init_strided_iteration(:t1_coord, :t1_backstrides, t1_shape, t1_strides, t1_rank, t1data)
      init_strided_iteration(:t2_coord, :t2_backstrides, t2_shape, t2_strides, t2_rank, t2data)
      init_strided_iteration(:t3_coord, :t3_backstrides, t3_shape, t3_strides, t3_rank, t3data)
      n.times do |i|
        yield i, t1data, t2data, t3data
        advance_strided_iteration(:t1_coord, :t1_backstrides, t1_shape, t1_strides, t1_rank, t1data)
        advance_strided_iteration(:t2_coord, :t2_backstrides, t2_shape, t2_strides, t2_rank, t2data)
        advance_strided_iteration(:t3_coord, :t3_backstrides, t3_shape, t3_strides, t3_rank, t3data)
      end
    end
  end

  @[AlwaysInline]
  def outer_strided_iteration(t1 : Tensor, t2 : Tensor)
    n = t1.size
    m = t2.size

    index = 0

    t1_contiguous = t1.flags.contiguous?
    t2_contiguous = t2.flags.contiguous?

    t1data = t1.raw + t1.offset
    t2data = t2.raw + t2.offset

    t1_shape, t1_strides, t1_rank = t1.shape, t1.strides, t1.rank
    t2_shape, t2_strides, t2_rank = t2.shape, t2.strides, t2.rank

    if t1_contiguous && t2_contiguous
      n.times do
        m.times do
          yield index, t1data, t2data
          t2data += 1
          index += 1
        end
        t1data += 1
      end
    elsif t1_contiguous
      init_strided_iteration(:t2_coord, :t2_backstrides, t2_shape, t2_strides, t2_rank, t2data)
      n.times do
        m.times do
          yield index, t1data, t2data
          advance_strided_iteration(:t2_coord, :t2_backstrides, t2_shape, t2_strides, t2_rank, t2data)
          index += 1
        end
        t1data += 1
      end
    elsif t2_contiguous
      init_strided_iteration(:t1_coord, :t1_backstrides, t1_shape, t1_strides, t1_rank, t1data)
      n.times do
        m.times do
          yield index, t1data, t2data
          index += 1
          t2data += 1
        end
        advance_strided_iteration(:t1_coord, :t1_backstrides, t1_shape, t1_strides, t1_rank, t1data)
      end
    else
      init_strided_iteration(:t1_coord, :t1_backstrides, t1_shape, t1_strides, t1_rank, t1data)
      init_strided_iteration(:t2_coord, :t2_backstrides, t2_shape, t2_strides, t2_rank, t2data)
      n.times do
        m.times do
          yield index, t1data, t2data
          index += 1
          advance_strided_iteration(:t2_coord, :t2_backstrides, t2_shape, t2_strides, t2_rank, t2data)
        end
        advance_strided_iteration(:t1_coord, :t1_backstrides, t1_shape, t1_strides, t1_rank, t1data)
      end
    end
  end

  @[AlwaysInline]
  def at_axis_index(t : Tensor, axis : Int32, idx : Int32, length : Int32 = 1, dims : Bool = false)
    Num::Internal.check_axis_index(t, axis, idx, length)
    shape = t.shape.dup
    strides = t.strides.dup

    shape[axis] = 1
    strides[axis] = 0

    unless dims
      shape.delete_at(axis)
      strides.delete_at(axis)
    end

    offset = t.offset + t.strides[axis] * idx
    flags = t.flags.dup
    flags &= ~Num::ArrayFlags::OwnData

    t.class.new(t.raw, shape, strides, offset, flags)
  end

  @[AlwaysInline]
  def axis_iteration(
    t : Tensor,
    axis : Int32,
    iter_offset : Int32,
    iter_size : Int32,
    dims : Bool = false
  )
    result = at_axis_index(t, axis, iter_offset, 1, dims)

    iter_size.times do
      yield result
      result.offset += t.strides[axis]
    end
  end

  @[AlwaysInline]
  def dual_axis_iteration(
    a : Tensor,
    b : Tensor,
    axis : Int32,
    iter_offset : Int32,
    iter_size : Int32,
    dims : Bool = false
  )
    raise Num::Exceptions::ValueError.new("Shapes along axis must match") unless a.shape[axis] == b.shape[axis]

    r0 = at_axis_index(a, axis, iter_offset, 1, dims)
    r1 = at_axis_index(b, axis, iter_offset, 1, dims)

    iter_size.times do
      yield r0, r1
      r0.offset += a.strides[axis]
      r1.offset += b.strides[axis]
    end
  end
end
