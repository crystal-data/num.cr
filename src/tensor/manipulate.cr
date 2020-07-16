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

require "./tensor"

module Num
  extend self

  # Repeat elements of a `Tensor`, treating the `Tensor`
  # as flat
  #
  # Arguments
  # ---------
  # - `a` : Tensor | Enumerable
  #   Object to repeat
  # - `n` : Int
  #   Number of times to repeat
  #
  # Examples
  # ```
  # a = [1, 2, 3]
  # Num.repeat(a, 2) # => [1, 1, 2, 2, 3, 3]
  # ```
  def repeat(a : Tensor | Enumerable, n : Int)
    a_t = a.to_tensor
    t = a_t.class.new([a_t.size * n])
    it = t.unsafe_iter

    repeat_inner(a_t, n) do |i|
      it.next.value = i
    end
    t
  end

  # Repeat elements of a `Tensor` along an axis
  #
  # Arguments
  # ---------
  # - `a` : Tensor | Enumerable
  #   Object to repeat
  # - `n` : Int
  #   Number of times to repeat
  # - `axis` : Int
  #   Axis along which to repeat
  #
  # Examples
  # --------
  # ```
  # a = [[1, 2, 3], [4, 5, 6]]
  # Num.repeat(a, 2, 1)
  #
  # # [[1, 1, 2, 2, 3, 3],
  # #  [4, 4, 5, 5, 6, 6]]
  # ```
  def repeat(a : Tensor | Enumerable, n : Int, axis : Int)
    a_t = a.to_tensor

    shape = a_t.shape.dup
    shape[axis] *= n
    t = a_t.class.new(shape)

    it = t.unsafe_axis_iter(axis.to_i)
    a_t.each_axis(axis) do |ax|
      n.times do
        it.next[...] = ax
      end
    end
    t
  end

  # Tile elements of a `Tensor`
  #
  # Arguments
  # ---------
  # - `a` : Tensor | Enumerable
  #   Argument to tile
  # - `n` : Int
  #   Number of times to tile
  #
  # Examples
  # --------
  # ```
  # a = [[1, 2, 3], [4, 5, 6]]
  # puts Num.tile(a, 2)
  #
  # # [[1, 2, 3, 1, 2, 3],
  # #  [4, 5, 6, 4, 5, 6]]
  # ```
  def tile(a : Tensor | Enumerable, n : Int)
    a_t = a.to_tensor
    d = a_t.rank > 1 ? [1] * (a_t.rank - 1) + [n] : [1]
    tile_inner(a_t, d)
  end

  # Tile elements of a `Tensor`
  #
  # Arguments
  # ---------
  # - `a` : Tensor | Enumerable
  #   Argument to tile
  # - `n` : Array(Int)
  #   Number of times to tile in each dimension
  #
  # Examples
  # --------
  # ```
  # a = [[1, 2, 3], [4, 5, 6]]
  # puts Num.tile(a, [2, 2])
  #
  # # [[1, 2, 3, 1, 2, 3],
  # #  [4, 5, 6, 4, 5, 6],
  # #  [1, 2, 3, 1, 2, 3],
  # #  [4, 5, 6, 4, 5, 6]]
  # ```
  def tile(a : Tensor | Enumerable, n : Array(Int))
    a_t = a.to_tensor
    n = n.size < a_t.rank ? [1] * (a_t.rank - n.size) + n : n
    tile_inner(a_t, n)
  end

  # Flips a `Tensor` along all axes, returning a view
  #
  # Arguments
  # ---------
  # - `a` : Tensor | Enumerable
  #   Argument to flip
  #
  # Examples
  # --------
  # ```
  # a = [[1, 2, 3], [4, 5, 6]]
  # puts Num.flip(a)
  #
  # # [[6, 5, 4],
  # #  [3, 2, 1]]
  # ```
  def flip(a : Tensor | Enumerable)
    a_t = a.to_tensor
    i = [{..., -1}] * a_t.rank
    a_t[i]
  end

  # Flips a `Tensor` along an axis, returning a view
  #
  # Arguments
  # ---------
  # - `a` : Tensor | Enumerable
  #   Argument to flip
  # - `axis` : Int
  #   Axis to flip
  #
  # Examples
  # --------
  # ```
  # a = [[1, 2, 3], [4, 5, 6]]
  # puts Num.flip(a, 1)
  #
  # # [[3, 2, 1],
  # #  [6, 5, 4]]
  # ```
  def flip(a : Tensor | Enumerable, axis : Int)
    a_t = a.to_tensor
    s = (0...a_t.rank).map do |i|
      i == axis ? {..., -1} : (...)
    end
    a_t[s]
  end

  # :nodoc:
  private def repeat_inner(a : Tensor, n : Int)
    a.each do |el|
      n.times do
        yield el
      end
    end
  end

  # :nodoc:
  private def tile_inner(a : Tensor, r : Array(Int))
    shape_out = a.shape.zip(r).map do |i, j|
      (i * j).to_i
    end
    n = a.size
    a.shape.zip(r) do |d, rep|
      if rep != 1
        a = repeat(a.reshape(-1, n), rep, 0)
      end
      n //= d
    end
    a.reshape(shape_out)
  end
end
