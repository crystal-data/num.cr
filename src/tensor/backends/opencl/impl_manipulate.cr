# Copyright (c) 2021 Crystal Data Contributors
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
  def broadcast_to(arr : Tensor(U, OCL(U)), shape : Array(Int)) forall U
    strides = Num::Internal.strides_for_broadcast(arr.shape, arr.strides, shape)
    Tensor.new(arr.data, shape, strides, arr.offset, U)
  end

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
  def broadcast(a : Tensor(U, OCL(U)), b : Tensor(V, OCL(V))) forall U, V
    if a.shape == b.shape
      return {a, b}
    end
    shape = Num::Internal.shape_for_broadcast(a, b)
    return {a.broadcast_to(shape), b.broadcast_to(shape)}
  end
end
