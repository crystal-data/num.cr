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

  def self.ones(shape : Array(Int32))
    Tensor(T).new(shape, T.new(1))
  end

  def self.ones_like(other : NumInternal::AnyTensor)
    Tensor(T).new(other.shape, T.new(1))
  end

  def self.full(shape : Array(Int32), value : Number)
    Tensor(T).new(shape, T.new(value))
  end

  def self.full_like(other : NumInternal::AnyTensor, value : Number)
    Tensor(T).new(other.shape, T.new(value))
  end

  def self.range(start : T, stop : T, step : T)
    if start > stop && step > 0
      raise NumInternal::ValueError.new("Range must return at at least one value")
    end
    r = (stop - start)
    num = (r / step).ceil.abs
    Tensor.new([Int32.new(num)]) { |i| T.new(start + (i * step)) }
  end

  def self.range(stop : T)
    Tensor.range(T.new(0), stop, T.new(1))
  end

  def self.range(start : T, stop : T)
    Tensor.range(start, stop, T.new(1))
  end

  def self.from_range(rng : Range(T, T))
    last = rng.excludes_end? ? rng.end : rng.end + T.new(1)
    self.range(rng.begin, last, T.new(1))
  end

  def self.eye(m : Int, n : Int? = nil, k : Int = 0)
    n = n.nil? ? m : n.as(Int32)
    Tensor.new(Int32.new(m), n) do |i, j|
      i == j - k ? T.new(1) : T.new(0)
    end
  end

  def self.identity(n : Int)
    n32 = Int32.new(n)
    Tensor.new(n32, n32) do |i, j|
      i == j ? T.new(1) : T.new(0)
    end
  end

  def self.diag(a : Tensor(T), k : Int32 = 0)
    if a.ndims > 1
      raise "Only 1 dimensional Tensors are supported"
    end
    iter = NumInternal::UnsafeNDFlatIter.new(a)
    Tensor(T).new(a.shape[0], a.shape[0]) do |i, j|
      i == j - k ? iter.next.value : T.new(0)
    end
  end
end
