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

class Tensor(T)
  # Return a shallow copy of a `Tensor`.  The underlying data buffer
  # is shared, but the `Tensor` owns its other attributes.  Changes
  # to a view of a `Tensor` will be reflected in the original `Tensor`
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # a = Tensor(Int32).new([3, 3])
  # b = a.view
  # b[...] = 99
  # a
  #
  # # [[99, 99, 99],
  # #  [99, 99, 99],
  # #  [99, 99, 99]]
  # ```
  def view : Tensor(T)
    shape_ = @shape.dup
    strides_ = @strides.dup
    flags_ = @flags.dup
    flags_ &= ~Num::ArrayFlags::OwnData
    self.class.new(@raw, shape_, strides_, @offset, flags_)
  end
end
