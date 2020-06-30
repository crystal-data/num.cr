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

require "./cl_tensor"

class ClTensor(T)
  # Creates a `ClTensor` of a provided shape, filled with 0.  The generic type
  # must be specified.
  #
  # Arguments
  # ---------
  # *shape*
  #   Shape of returned `ClTensor`
  #
  # Examples
  # --------
  # ```
  # t = ClTensor(Float32).zeros([3])
  # ```
  def self.zeros(shape : Array(Int))
    ClTensor(T).new(shape, T.new(0))
  end

  # Creates a `ClTensor` filled with 0, sharing the shape of another
  # provided `Tensor` or `ClTensor`
  #
  # Arguments
  # ---------
  # *t*
  #   Item to use for output shape
  #
  # Examples
  # --------
  # ```
  # t = Tensor.new([3]) &.to_f
  # u = ClTensor(Float32).zeros_like(t)
  # ```
  def self.zeros_like(other : ClTensor | Tensor)
    ClTensor(T).new(other.shape, T.new(0))
  end

  # Creates a `ClTensor` of a provided shape, filled with 1.  The generic type
  # must be specified.
  #
  # Arguments
  # ---------
  # *shape*
  #   Shape of returned `ClTensor`
  #
  # Examples
  # --------
  # ```
  # t = ClTensor(Float32).ones([3])
  # ```
  def self.ones(shape : Array(Int))
    ClTensor(T).new(shape, T.new(1))
  end

  # Creates a `ClTensor` filled with 1, sharing the shape of another
  # provided `Tensor` or `ClTensor`
  #
  # Arguments
  # ---------
  # *t*
  #   Item to use for output shape
  #
  # Examples
  # --------
  # ```
  # t = Tensor.new([3]) &.to_f
  # u = ClTensor(Float32).ones_like(t)
  # ```
  def self.ones_like(other : ClTensor | Tensor)
    ClTensor(T).new(other.shape, T.new(1))
  end

  # Creates a `ClTensor` of a provided shape, filled with a value.
  # The generic type must be specified.
  #
  # Arguments
  # ---------
  # *shape*
  #   Shape of returned `ClTensor`
  #
  # Examples
  # --------
  # ```
  # t = ClTensor(Float32).full([3], 3.5)
  # ```
  def self.full(shape : Array(Int), value : Number)
    ClTensor(T).new(shape, T.new(value))
  end

  # Creates a `ClTensor` filled with a value, sharing the shape of another
  # provided `Tensor` or `ClTensor`
  #
  # Arguments
  # ---------
  # *t*
  #   Item to use for output shape
  #
  # Examples
  # --------
  # ```
  # t = Tensor.new([3]) &.to_f
  # u = ClTensor(Float32).full_like(t, 3.5)
  # ```
  def self.full_like(other : ClTensor | Tensor, value : Number)
    ClTensor(T).new(other.shape, T.new(value))
  end
end
