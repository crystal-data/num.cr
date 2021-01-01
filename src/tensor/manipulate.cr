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

class Tensor(T, S)
  # Broadcasts a `Tensor` to a new shape.  Returns a read-only
  # view of the original `Tensor`.  Many elements in the `Tensor`
  # will refer to the same memory location, and the result is
  # rarely contiguous.
  #
  # Shapes must be broadcastable, and an error will be raised
  # if broadcasting fails.
  #
  # Arguments
  # ---------
  # *shape* : Array(Int)
  #   The shape of the desired output `Tensor`
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3]
  # a.broadcast_to([3, 3])
  #
  # # [[1, 2, 3],
  # #  [1, 2, 3],
  # #  [1, 2, 3]]
  # ```
  delegate_to_backend broadcast_to

  # Broadcasts two `Tensor`'s' to a new shape.  This allows
  # for elementwise operations between the two Tensors with the
  # new shape.
  #
  # Broadcasting rules apply, and imcompatible shapes will raise
  # an error.
  #
  # Examples
  # ````````
  # a = Tensor.from_array [1, 2, 3]
  # b = Tensor.new([3, 3]) { |i| i }
  #
  # x, y = a.broadcast(b)
  # x.shape # => [3, 3]
  # ````````
  delegate_to_backend broadcast

  # Transform's a `Tensor`'s shape.  If a view can be created,
  # the reshape will not copy data.  The number of elements
  # in the `Tensor` must remain the same.
  #
  # Arguments
  # ---------
  # *result_shape* : Array(Int)
  #   Result shape for the `Tensor`
  #
  # Examples
  # --------
  # ```
  # a = Tensor.from_array [1, 2, 3, 4]
  # a.reshape([2, 2])
  #
  # # [[1, 2],
  # #  [3, 4]]
  # ```
  delegate_to_backend reshape

  # Flattens a `Tensor` to a single dimension.  If a view can be created,
  # the reshape operation will not copy data.
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # a.flat # => [0, 1, 2, 3]
  # ```
  delegate_to_backend flat

  # Move axes of a Tensor to new positions, other axes remain
  # in their original order
  #
  # Arguments
  # ---------
  # *arr* : Tensor
  #   Tensor to permute
  # *source* : Array(Int)
  #   Original positions of axes to move
  # *destination*
  #   Destination positions to permute axes
  #
  # Examples
  # --------
  # ```
  # a = Tensor(Int8, CPU(Int8)).new([3, 4, 5])
  # moveaxis(a, [0], [-1]).shape # => 4, 5, 3
  # ```
  delegate_to_backend move_axis

  # Permutes two axes of a `Tensor`.  This will always create a view
  # of the permuted `Tensor`
  #
  # Arguments
  # ---------
  # *a* : Int
  #   First axis of permutation
  # *b* : Int
  #   Second axis of permutation
  #
  # Examples
  # --------
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
  delegate_to_backend swap_axes

  # Permutes a `Tensor`'s axes to a different order.  This will
  # always create a view of the permuted `Tensor`.
  #
  # Arguments
  # ---------
  # *axes* : Array(Int)
  #   New ordering of axes for the permuted `Tensor`.  If empty,
  #   a full transpose will occur
  #
  # Examples
  # --------
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
  delegate_to_backend transpose

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
  delegate_to_backend repeat

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
  delegate_to_backend tile

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
  delegate_to_backend flip
end
