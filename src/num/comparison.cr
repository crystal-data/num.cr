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

require "../tensor/tensor"

module Num
  extend self

  # Asserts that two equally shaped `Tensor`s are equal within a provided
  # tolerance.  Useful for floating point comparison where direct equality might
  # not work
  #
  # ```
  # t = Tensor.new([2, 2, 3]) { |i| i * 1.0 }
  # tf = t + 0.00000000001
  # allclose(t, tf) # => true
  # ```
  def allclose(a : AnyArray(U), b : AnyArray(U), rtol = 1e-5, atol = 1e-8) forall U
    if a.shape != b.shape
      raise NumInternal::ShapeError.new "Shape of arguments must match"
    end
    iter_a = a.iter
    iter_b = b.iter

    if (rtol > 0)
      iter_a.zip(iter_b).each do |i, j|
        c = !((i.value - j.value).abs > atol + rtol * j.value.abs)
        return false unless c
      end
    else
      iter_a.zip(iter_b).each do |i, j|
        c = (i.value - j.value).abs > atol
        return false unless c
      end
    end
    true
  end

  # Asserts that two boolean arrays are the same
  def allclose(a : AnyArray(Bool), b : AnyArray(Bool))
    if a.shape != b.shape
      raise "Shape of arguments must match"
    end
    a.iter.zip(b.iter).each do |i, j|
      return false unless i.value == j.value
    end
    true
  end

  # Makes an actual assertion about two arrays
  def assert_array_equal(a, b)
    allclose(a, b).should be_true
  end

  # Asserts that two arrays are compatible to be compared
  def assert_compatible_shape(a : Array(Int32), b : Int32)
    toraise = false
    if a.size == 0
      if b != 0
        toraise = true
      end
    else
      sz = a.reduce { |i, j| i * j }
      toraise = true unless b == sz
    end
    if toraise
      raise NumInternal::ShapeError.new("An array of size #{b} cannot go
        into a Tensor of shape #{a}")
    end
  end
end
