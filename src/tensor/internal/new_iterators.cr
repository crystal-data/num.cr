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

require "../tensor"

macro init_strided_iteration(coord, backstrides, t_shape, t_strides, t_rank)
  {{ coord.id }} = Pointer(Int32).malloc({{ t_rank }}, 0)
  {{ backstrides.id }} = Pointer(Int32).malloc({{ t_rank }})
  {{ t_rank }}.times do |i|
    {{ backstrides.id }}[i] = {{ t_strides }}[i] * ({{ t_shape }}[i] - 1)
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
  data = t.to_unsafe
  if t.flags.contiguous?
    t.size.times do |i|
      yield i, data
      data += 1
    end
  else
    t_shape, t_strides, t_rank = t.iter_attrs
    init_strided_iteration(:coord, :backstrides, t_shape, t_strides, t_rank)
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

  t1data = t1.to_unsafe
  t2data = t2.to_unsafe

  t1_shape, t1_strides, t1_rank = t1.iter_attrs
  t2_shape, t2_strides, t2_rank = t2.iter_attrs

  if t1_contiguous && t2_contiguous
    n.times do |i|
      yield i, t1data, t2data
      t1data += 1
      t2data += 1
    end
  elsif t1_contiguous
    init_strided_iteration(:t2_coord, :t2_backstrides, t2_shape, t2_strides, t2_rank)
    n.times do |i|
      yield i, t1data, t2data
      t1data += 1
      advance_strided_iteration(:t2_coord, :t2_backstrides, t2_shape, t2_strides, t2_rank, t2data)
    end
  elsif t2_contiguous
    init_strided_iteration(:t1_coord, :t1_backstrides, t1_shape, t1_strides, t1_rank)
    n.times do |i|
      yield i, t1data, t2data
      advance_strided_iteration(:t1_coord, :t1_backstrides, t1_shape, t1_strides, t1_rank, t1data)
      t2data += 1
    end
  else
    init_strided_iteration(:t1_coord, :t1_backstrides, t1_shape, t1_strides, t1_rank)
    init_strided_iteration(:t2_coord, :t2_backstrides, t2_shape, t2_strides, t2_rank)
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

  t1data = t1.to_unsafe
  t2data = t2.to_unsafe
  t3data = t3.to_unsafe

  t1_shape, t1_strides, t1_rank = t1.iter_attrs
  t2_shape, t2_strides, t2_rank = t2.iter_attrs
  t3_shape, t3_strides, t3_rank = t3.iter_attrs

  if t1_contiguous && t2_contiguous && t3_contiguous
    n.times do |i|
      yield i, t1data, t2data, t3data
      t1data += 1
      t2data += 1
      t3data += 1
    end
  elsif t1_contiguous && t2_contiguous
    init_strided_iteration(:t3_coord, :t3_backstrides, t3_shape, t3_strides, t3_rank)
    n.times do |i|
      yield i, t1data, t2data, t3data
      t1data += 1
      t2data += 1
      advance_strided_iteration(:t3_coord, :t3_backstrides, t3_shape, t3_strides, t3_rank, t3data)
    end
  elsif t1_contiguous
    init_strided_iteration(:t2_coord, :t2_backstrides, t2_shape, t2_strides, t2_rank)
    init_strided_iteration(:t3_coord, :t3_backstrides, t3_shape, t3_strides, t3_rank)
    n.times do |i|
      yield i, t1data, t2data, t3data
      t1data += 1
      advance_strided_iteration(:t2_coord, :t2_backstrides, t2_shape, t2_strides, t2_rank, t2data)
      advance_strided_iteration(:t3_coord, :t3_backstrides, t3_shape, t3_strides, t3_rank, t3data)
    end
  else
    init_strided_iteration(:t1_coord, :t1_backstrides, t1_shape, t1_strides, t1_rank)
    init_strided_iteration(:t2_coord, :t2_backstrides, t2_shape, t2_strides, t2_rank)
    init_strided_iteration(:t3_coord, :t3_backstrides, t3_shape, t3_strides, t3_rank)
    n.times do |i|
      yield i, t1data, t2data, t3data
      advance_strided_iteration(:t1_coord, :t1_backstrides, t1_shape, t1_strides, t1_rank, t1data)
      advance_strided_iteration(:t2_coord, :t2_backstrides, t2_shape, t2_strides, t2_rank, t2data)
      advance_strided_iteration(:t3_coord, :t3_backstrides, t3_shape, t3_strides, t3_rank, t3data)
    end
  end
end
