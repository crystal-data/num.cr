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
  # Returns a view of a `Tensor` from any valid indexers. This view
  # must be able to be represented as valid strided/shaped view, slicing
  # as a copy is not supported.
  #
  #
  # When an Integer argument is passed, an axis will be removed from
  # the `Tensor`, and a view at that index will be returned.
  #
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # a[0] # => [0, 1]
  # ```
  #
  # When a Range argument is passed, an axis will be sliced based on
  # the endpoints of the range.
  #
  # ```
  # a = Tensor.new([2, 2, 2]) { |i| i }
  # a[1...]
  #
  # # [[[4, 5],
  # #   [6, 7]]]
  # ```
  #
  # When a Tuple containing a Range and an Integer step is passed, an axis is
  # sliced based on the endpoints of the range, and the strides of the
  # axis are updated to reflect the step.  Negative steps will reflect
  # the array along an axis.
  #
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # a[{..., -1}]
  #
  # # [[2, 3],
  # #  [0, 1]]
  # ```
  def slice(arr : Tensor(U, V), *args) forall U, V
    slice(arr, args.to_a)
  end

  # Returns a view of a `Tensor` from any valid indexers. This view
  # must be able to be represented as valid strided/shaped view, slicing
  # as a copy is not supported.
  #
  #
  # When an Integer argument is passed, an axis will be removed from
  # the `Tensor`, and a view at that index will be returned.
  #
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # a[0] # => [0, 1]
  # ```
  #
  # When a Range argument is passed, an axis will be sliced based on
  # the endpoints of the range.
  #
  # ```
  # a = Tensor.new([2, 2, 2]) { |i| i }
  # a[1...]
  #
  # # [[[4, 5],
  # #   [6, 7]]]
  # ```
  #
  # When a Tuple containing a Range and an Integer step is passed, an axis is
  # sliced based on the endpoints of the range, and the strides of the
  # axis are updated to reflect the step.  Negative steps will reflect
  # the array along an axis.
  #
  # ```
  # a = Tensor.new([2, 2]) { |i| i }
  # a[{..., -1}]
  #
  # # [[2, 3],
  # #  [0, 1]]
  # ```
  def slice(arr : Tensor(U, V), args : Array) forall U, V
    offset, shape, strides = Num::Internal.offset_for_index(arr, args)
    flags = arr.flags.dup
    flags &= ~Num::ArrayFlags::OwnData
    Tensor.new(arr.data, shape, strides, offset, flags, U)
  end
end
