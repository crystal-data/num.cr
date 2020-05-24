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

require "../array/array"
require "../base/array"

class Tensor(T) < AnyArray(T)
  # A flexible method to create `Tensor`'s of arbitrary shapes
  # filled with random values of arbitrary types.  Since
  # Ranges can contain any dtype, the type of tensor is
  # inferred from the passed range, and a new `Tensor` is
  # sampled using the provided shape.
  def self.random(r : Range(U, U), _shape : Array(Int32)) forall U
    if _shape.size == 0
      Tensor(U).new(_shape)
    else
      new(_shape) { |_| Random.rand(r) }
    end
  end

  def self.zeros(shape : Array(Int32))
    Tensor(T).new(shape, T.new(0))
  end

  def self.zeros_like(other : NumInternal::AnyTensor)
    Tensor(T).new(other.shape, T.new(0))
  end

  def zeros_like
    Tensor(T).new(@shape, T.new(0))
  end

  def self.ones(shape : Array(Int32))
    Tensor(T).new(shape, T.new(1))
  end

  def self.ones_like(other : NumInternal::AnyTensor)
    Tensor(T).new(other.shape, T.new(1))
  end

  def ones_like
    Tensor(T).new(@shape, T.new(1))
  end

  def self.full(shape : Array(Int32), value : Number)
    Tensor(T).new(shape, T.new(value))
  end

  def self.full_like(other : NumInternal::AnyTensor, value : Number)
    Tensor(T).new(other.shape, T.new(value))
  end
end
