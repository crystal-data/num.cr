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

module Num
  extend self

  # Broadcasts two `Tensor`'s' to a new shape.  This allows
  # for elementwise operations between the two Tensors with the
  # new shape.
  #
  # Broadcasting rules apply, and imcompatible shapes will raise
  # an error.
  #
  # ## Arguments
  #
  # * a : `Tensor(U, CPU(U))` - First `Tensor` to broadcast
  # * b : `Tensor(V, CPU(V))` - Second `Tensor` to broadcast
  #
  # ## Examples
  #
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # b = Tensor.new([3, 3]) { |i| i }
  #
  # x, y = a.broadcast(b)
  # x.shape # => [3, 3]
  # ```
  
  def broadcast(a : Tensor(U, CPU(U)), b : Tensor(V, CPU(V))) forall U, V
    if a.shape == b.shape
      return {a, b}
    end
    shape = Num::Internal.shape_for_broadcast(a, b)
    return {a.broadcast_to(shape), b.broadcast_to(shape)}
  end

  # Broadcasts two `Tensor`'s' to a new shape.  This allows
  # for elementwise operations between the two Tensors with the
  # new shape.
  #
  # Broadcasting rules apply, and imcompatible shapes will raise
  # an error.
  #
  # ## Arguments
  #
  # * a : `Tensor(U, CPU(U))` - First `Tensor` to broadcast
  # * b : `Tensor(V, CPU(V))` - Second `Tensor` to broadcast
  # * c : `Tensor(W, CPU(W))` - Third `Tensor` to broadcast
  #
  # ## Examples
  #
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # b = Tensor.new([3, 3]) { |i| i }
  # c = Tensor.new([3, 3, 3, 3]) { |i| i }
  #
  # x, y, z = a.broadcast(b, c)
  # x.shape # => [3, 3, 3, 3]
  # ```
  
  def broadcast(a : Tensor(U, CPU(U)), b : Tensor(V, CPU(V)), c : Tensor(W, CPU(W))) forall U, V, W
    if a.shape == b.shape && a.shape == c.shape
      return {a, b, c}
    end
    shape = Num::Internal.shape_for_broadcast(a, b, c)
    return {a.broadcast_to(shape), b.broadcast_to(shape), c.broadcast_to(shape)}
  end

  # Transform's a `Tensor`'s shape.  If a view can be created,
  # the reshape will not copy data.  The number of elements
  # in the `Tensor` must remain the same.
  #
  # ## Arguments
  #
  # * arr : `Tensor(U, CPU(U))` - `Tensor` to reshape
  # * shape : `Array(Int)` - New shape for the `Tensor`
  #
  # ## Examples
  #
  # ```
  # a = Tensor.from_array [1, 2, 3, 4]
  # a.reshape([2, 2])
  #
  # # [[1, 2],
  # #  [3, 4]]
  # ```
  
  def reshape(arr : Tensor(U, CPU(U)), shape : Array(Int)) forall U
    shape, strides = Num::Internal.strides_for_reshape(arr.shape, shape)
    flags = arr.flags.dup
    if arr.is_c_contiguous
      flags &= ~Num::ArrayFlags::OwnData
    else
      arr = arr.dup(Num::RowMajor)
    end
    Tensor(U, CPU(U)).new(arr.data, shape, strides, arr.offset, flags, U)
  end

  # :ditto:
  
  def reshape(arr : Tensor(U, CPU(U)), *shape : Int) forall U
    reshape(arr, shape.to_a)
  end

  # Flattens a `Tensor` to a single dimension.  If a view can be created,
  # the reshape operation will not copy data.
  #
  # ## Arguments
  #
  # * arr : `Tensor(U, CPU(U))` - `Tensor` to flatten
  #
  # ## Examples
  #
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # a.flat # => [0, 1, 2, 3]
  # ```
  
  def flat(arr : Tensor(U, CPU(U))) forall U
    reshape(arr, -1)
  end

  # Move axes of a Tensor to new positions, other axes remain
  # in their original order
  #
  # ## Arguments
  #
  # * arr : `Tensor(U, CPU(U))` - `Tensor` to permute
  # * source : `Array(Int)` - Original positions of axes
  # * destination : `Array(Int)` - Destination positions of axes
  #
  # ## Examples
  #
  # ```
  # a = Tensor(Int8, CPU(Int8)).new([3, 4, 5])
  # Num.moveaxis(a, [0], [-1]).shape # => 4, 5, 3
  # ```
  
  def move_axis(arr : Tensor(U, CPU(U)), source : Array(Int), destination : Array(Int)) forall U
    axes = Num::Internal.move_axes_for_transpose(arr.rank, source, destination)
    transpose(arr, axes)
  end

  # Move axes of a Tensor to new positions, other axes remain
  # in their original order
  #
  # ## Arguments
  #
  # * arr : `Tensor(U, CPU(U))` - `Tensor` to permute
  # * source : `Int` - Original position of axis
  # * destination : `Int` - Destination position of axis
  #
  # ## Examples
  #
  # ```
  # a = Tensor(Int8, CPU(Int8)).new([3, 4, 5])
  # Num.moveaxis(a, 0, 1).shape # => 4, 5, 3
  # ```
  
  def move_axis(arr : Tensor(U, CPU(U)), source : Int, destination : Int) forall U
    moveaxis(arr, [source], [destination])
  end

  # Permutes two axes of a `Tensor`.  This will always create a view
  # of the permuted `Tensor`
  #
  # ## Arguments
  #
  # * arr : `Tensor(U, CPU(U))` - `Tensor` to permute
  # * source : `Int` - First axis to swap
  # * destination : `Int` - Second axis to swap
  #
  # ## Examples
  #
  # ```
  # a = Tensor.new([4, 3, 2]) { |i| i }
  # a.swap_axes(2, 0)
  #
  # # [[[ 0,  6, 12, 18]
  # #   [ 2,  8, 14, 20]
  # #   [ 4, 10, 16, 22]]
  # #
  # #  [[ 1,  7, 13, 19]
  # #   [ 3,  9, 15, 21]
  # #   [ 5, 11, 17, 23]]]
  # ```
  
  def swap_axes(arr : Tensor(U, CPU(U)), a : Int, b : Int) forall U
    axes = Num::Internal.swap_axes_for_transpose(arr.rank, a, b)
    transpose(arr, axes)
  end

  # Permutes a `Tensor`'s axes to a different order.  This will
  # always create a view of the permuted `Tensor`.
  #
  # ## Arguments
  #
  # ## Arguments
  #
  # * arr : `Tensor(U, CPU(U))` - `Tensor` to permute
  # * axes : `Array(Int)` - Order of axes to permute
  #
  # ## Examples
  #
  # ```
  # a = Tensor.new([4, 3, 2]) { |i| i }
  # a.transpose([2, 0, 1])
  #
  # # [[[ 0,  2,  4],
  # #   [ 6,  8, 10],
  # #   [12, 14, 16],
  # #   [18, 20, 22]],
  # #
  # #  [[ 1,  3,  5],
  # #   [ 7,  9, 11],
  # #   [13, 15, 17],
  # #   [19, 21, 23]]]
  # ```
  
  def transpose(arr : Tensor(U, CPU(U)), axes : Array(Int) = [] of Int32) forall U
    shape, strides = Num::Internal.shape_and_strides_for_transpose(arr.shape, arr.strides, axes)
    flags = arr.flags.dup
    flags &= ~Num::ArrayFlags::OwnData
    Tensor(U, CPU(U)).new(arr.data, shape, strides, arr.offset, U)
  end

  # :ditto:
  
  def transpose(arr : Tensor(U, CPU(U)), *args : Int) forall U
    transpose(arr, args.to_a)
  end

  # Repeat elements of a `Tensor`, treating the `Tensor`
  # as flat
  #
  # ## Arguments
  #
  # * a : `Tensor(U, CPU(U))` - `Tensor` to repeat
  # * n : `Int` - Number of times to repeat
  #
  # ## Examples
  #
  # ```
  # a = [1, 2, 3]
  # Num.repeat(a, 2) # => [1, 1, 2, 2, 3, 3]
  # ```
  
  def repeat(a : Tensor(U, CPU(U)), n : Int) forall U
    result = Tensor(U, CPU(U)).new([a.size * n])
    iter = result.each
    Num::Internal.repeat_inner(a, n) do |value|
      iter.next.value = value
    end
    result
  end

  # Repeat elements of a `Tensor` along an axis
  #
  # ## Arguments
  #
  # * a : `Tensor(U, CPU(U))` - `Tensor` to repeat
  # * n : `Int` - Number of times to repeat
  # * axis : `Int` - Axis along which to repeat
  #
  # ## Examples
  #
  # ```
  # a = [[1, 2, 3], [4, 5, 6]]
  # Num.repeat(a, 2, 1)
  #
  # # [[1, 1, 2, 2, 3, 3],
  # #  [4, 4, 5, 5, 6, 6]]
  # ```
  
  def repeat(a : Tensor(U, CPU(U)), n : Int, axis : Int) forall U
    shape = a.shape.dup
    shape[axis] *= n
    result = Tensor(U, CPU(U)).new(shape)
    iter = each_axis(result, axis.to_i)
    each_axis(a, axis) do |ax|
      n.times do
        iter.next[...] = ax
      end
    end
    result
  end

  # Tile elements of a `Tensor`
  #
  # ## Arguments
  #
  # * a : `Tensor(U, CPU(U))` - `Tensor` to repeat
  # * n : `Int` - Number of times to tile
  #
  # ## Examples
  #
  # ```
  # a = [[1, 2, 3], [4, 5, 6]]
  # puts Num.tile(a, 2)
  #
  # # [[1, 2, 3, 1, 2, 3],
  # #  [4, 5, 6, 4, 5, 6]]
  # ```
  
  def tile(a : Tensor(U, CPU(U)), n : Int) forall U
    d = a.rank > 1 ? [1] * (a.rank - 1) + [n] : [1]
    Num::Internal.tile_inner(a, d)
  end

  # Tile elements of a `Tensor`
  #
  # ## Arguments
  #
  # * a : `Tensor(U, CPU(U))` - `Tensor` to repeat
  # * n : `Array(Int)` - Number of times to repeat
  #
  # ## Examples
  #
  # ```
  # a = [[1, 2, 3], [4, 5, 6]]
  # puts Num.tile(a, 2)
  #
  # # [[1, 2, 3, 1, 2, 3],
  # #  [4, 5, 6, 4, 5, 6]]
  # ```
  
  def tile(a : Tensor(U, CPU(U)), n : Array(Int)) forall U
    n = n.size < a.rank ? [1] * (a.rank - n.size) + n : n
    Num::Internal.tile_inner(a, n)
  end

  # Flips a `Tensor` along all axes, returning a view
  #
  # ## Arguments
  #
  # * a : `Tensor(U, CPU(U))` - `Tensor` to flip
  #
  # ## Examples
  #
  # ```
  # a = [[1, 2, 3], [4, 5, 6]]
  # puts Num.flip(a)
  #
  # # [[6, 5, 4],
  # #  [3, 2, 1]]
  # ```
  
  def flip(a : Tensor(U, CPU(U))) forall U
    i = [{..., -1}] * a.rank
    a[i]
  end

  # Flips a `Tensor` along an axis, returning a view
  #
  # ## Arguments
  #
  # * a : `Tensor(U, CPU(U))` - `Tensor` to flip
  # * axis : `Int` - Axis along which to flip
  #
  # ## Examples
  #
  # ```
  # a = [[1, 2, 3], [4, 5, 6]]
  # puts Num.flip(a, 1)
  #
  # # [[3, 2, 1],
  # #  [6, 5, 4]]
  # ```
  
  def flip(a : Tensor(U, CPU(U)), axis : Int) forall U
    s = (0...a.rank).map do |i|
      i == axis ? {..., -1} : (...)
    end
    a[s]
  end

  # Split a `Tensor` into multiple sub-`Tensor`s
  #
  # ## Arguments
  #
  # * a : `Tensor(U, CPU(U))` - `Tensor` to split`
  # * ind : `Int` - Number of sections of resulting `Array`
  # * axis : `Int` - Axis along which to split
  #
  # ## Examples
  #
  # ```
  # a = Tensor.range(9)
  # puts Num.array_split(a, 2) # => [[0, 1, 2, 3, 4], [5, 6, 7, 8]]
  # ```
  def array_split(
    a : Tensor(U, CPU(U)),
    ind : Int,
    axis : Int = 0
  ) : Array(Tensor(U, CPU(U))) forall U
    n = a.shape[axis]
    e = n // ind
    extra = n % ind
    sizes = [0]
    sizes += Array.new(extra, e + 1)
    sizes += Array.new(ind - extra, e)
    r = 0
    sizes.each_with_index do |s, i|
      tmp = r
      r += s
      sizes[i] = s + tmp
    end
    split_internal(a, axis, ind, sizes)
  end

  # Split a `Tensor` into multiple sub-`Tensor`s, using an explicit mapping
  # of indices to split the `Tensor`
  #
  # ## Arguments
  #
  # * a : `Tensor(U, CPU(U))` - `Tensor` to split`
  # * ind : `Int` - Array of indices to use when splitting the `Tensor`
  # * axis : `Int` - Axis along which to split
  #
  # ## Examples
  #
  # ```
  # a = Tensor.range(9)
  # puts Num.array_split(a, [1, 3, 5]) # => [[0], [1, 2], [3, 4], [5, 6, 7, 8]]
  # ```
  def array_split(
    a : Tensor(U, CPU(U)),
    ind : Array(Int),
    axis : Int = 0
  ) : Array(Tensor(U, CPU(U))) forall U
    n = ind.size + 1
    div_points = [0]
    div_points += ind
    div_points << a.shape[axis]
    split_internal(a, axis, n, div_points)
  end

  # Split a `Tensor` into multiple sub-`Tensor`s.  The number of sections
  # must divide the `Tensor` equally.
  #
  # ## Arguments
  #
  # * a : `Tensor(U, CPU(U))` - `Tensor` to split`
  # * ind : `Int` - Number of sections of resulting `Array`
  # * axis : `Int` - Axis along which to split
  #
  # ## Examples
  #
  # ```
  # a = Tensor.range(1, 9)
  # puts Num.array_split(a, 2) # => [[1, 2, 3, 4], [5, 6, 7, 8]]
  # ```
  def split(
    a : Tensor(U, CPU(U)),
    ind : Int,
    axis : Int = 0
  ) : Array(Tensor(U, CPU(U))) forall U
    n = a.shape[axis]
    if n % ind != 0
      raise Num::Exceptions::ValueError.new(
        "Split does not result in equal size sub-Tensors"
      )
    end
    array_split(a, ind, axis)
  end

  # Split a `Tensor` into multiple sub-`Tensor`s, using an explicit mapping
  # of indices to split the `Tensor`
  #
  # ## Arguments
  #
  # * a : `Tensor(U, CPU(U))` - `Tensor` to split`
  # * ind : `Int` - Array of indices to use when splitting the `Tensor`
  # * axis : `Int` - Axis along which to split
  #
  # ## Examples
  #
  # ```
  # a = Tensor.range(9)
  # puts Num.array_split(a, [1, 3, 5]) # => [[0], [1, 2], [3, 4], [5, 6, 7, 8]]
  # ```
  def split(
    a : Tensor(U, CPU(U)),
    ind : Array(Int),
    axis : Int = 0
  ) : Array(Tensor(U, CPU(U))) forall U
    array_split(a, ind, axis)
  end

  private def split_internal(
    arr : Tensor(U, V),
    axis : Int,
    n : Int,
    div_points : Array(Int)
  ) : Array(Tensor(U, V)) forall U, V
    result = [] of Tensor(U, V)
    swapped = arr.swap_axes(axis, 0)
    n.times do |i|
      s0 = div_points[i]
      s1 = div_points[i + 1]
      result << arr[s0...s1].swap_axes(axis, 0)
    end
    result
  end
end
