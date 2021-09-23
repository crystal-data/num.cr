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
  extend self

  # :nodoc:
  def each_axis(a0 : Tensor(U, OCL(U)), axis : Int, dims : Bool = false, &block : Tensor(U, OCL(U)) -> _) forall U
    axis = normalize_axis_index(axis, a0.rank)
    0.step(to: a0.shape[axis] - 1) do |i|
      yield at_axis_index(a0, axis, i, dims)
    end
  end

  # :nodoc:
  def reduce_axis(a0 : Tensor(U, OCL(U)), axis : Int, dims : Bool = false, &block : Tensor(U, OCL(U)), Tensor(U, OCL(U)) -> _) forall U
    axis = normalize_axis_index(axis, a0.rank)
    memo = at_axis_index(a0, axis, 0, dims)
    result = Tensor(U, OCL(U)).zeros(memo.shape)
    yield result, memo
    1.step(to: a0.shape[axis] - 1) do |i|
      yield result, at_axis_index(a0, axis, i, dims)
    end
    result
  end
end
