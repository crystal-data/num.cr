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

struct Number
  # :nodoc:
  macro op(name, operator)
    @[Inline]
    def {{operator.id}}(other : Tensor)
      Num.{{name}}(self, other)
    end
  end

  # Adds a `Tensor` to a number.  The number is broadcasted across
  # all elements of the `Tensor`.
  #
  # ## Arguments
  #
  # * other : `Tensor` - `Tensor` on which to perform the operation
  #
  # ## Examples
  #
  # ```
  # a = [1, 2, 3].to_tensor
  # 3 + a # => [4, 5, 6]
  # ```
  op add, :+

  # Divides a number by a `Tensor`.  The number is broadcasted across
  # all elements of the `Tensor`.
  #
  # ## Arguments
  #
  # * other : `Tensor` - `Tensor` on which to perform the operation
  #
  # ## Examples
  #
  # ```
  # a = [3, 3, 3].to_tensor
  # 3 / a # => [1, 1, 1]
  # ```
  op divide, :/

  # Multiplies a `Tensor` with a number.  The number is broadcasted across
  # all elements of the `Tensor`.
  #
  # ## Arguments
  #
  # * other : `Tensor` - `Tensor` on which to perform the operation
  #
  # ## Examples
  #
  # ```
  # a = [1, 2, 3].to_tensor
  # 3 * a # => [3, 6, 9]
  # ```
  op multiply, :*

  # Subtracts a `Tensor` from a number.  The number is broadcasted across
  # all elements of the `Tensor`.
  #
  # ## Arguments
  #
  # * other : `Tensor` - `Tensor` on which to perform the operation
  #
  # ## Examples
  #
  # ```
  # a = [3, 3, 3].to_tensor
  # 3 - a # => [0, 0, 0]
  # ```
  op subtract, :-

  # Raises a number to a `Tensor`.  The number is broadcasted across
  # all elements of the `Tensor`.
  #
  # ## Arguments
  #
  # * other : `Tensor` - `Tensor` on which to perform the operation
  #
  # ## Examples
  #
  # ```
  # a = [1, 2, 3].to_tensor
  # 2 ** a # => [2, 4, 8]
  # ```
  op power, :**
end
