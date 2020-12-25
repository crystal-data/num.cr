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

class Tensor(T)
  # Creates a `Tensor` of a provided shape, filled with 0.  The generic type
  # must be specified.
  #
  # Arguments
  # ---------
  # *shape*
  #   Shape of returned `Tensor`
  #
  # Examples
  # --------
  # ```
  # t = Tensor(Int8).zeros([3]) # => [0, 0, 0]
  # ```
  def self.zeros(shape : Array(Int), device = CPU(T)) : Tensor(T)
    self.new(shape, T.new(0), Num::RowMajor, device)
  end

  # Creates a `Tensor` filled with 0, sharing the shape of another
  # provided `Tensor`
  #
  # Arguments
  # ---------
  # *t*
  #   `Tensor` to use for output shape
  #
  # Examples
  # --------
  # ```
  # t = Tensor.new([3]) &.to_f
  # u = Tensor(Int8).zeros_like(t) # => [0, 0, 0]
  # ```
  def self.zeros_like(t : Tensor, device = CPU(T)) : Tensor(T)
    self.new(t.shape, T.new(0), Num::RowMajor, device)
  end

  # Creates a `Tensor` of a provided shape, filled with 1.  The generic type
  # must be specified.
  #
  # Arguments
  # ---------
  # *shape*
  #   Shape of returned `Tensor`
  #
  # Examples
  # --------
  # ```
  # t = Tensor(Int8).ones([3]) # => [1, 1, 1]
  # ```
  def self.ones(shape : Array(Int), device = CPU(T)) : Tensor(T)
    self.new(shape, T.new(1), Num::RowMajor, device)
  end

  # Creates a `Tensor` filled with 1, sharing the shape of another
  # provided `Tensor`
  #
  # Arguments
  # ---------
  # *t*
  #   `Tensor` to use for output shape
  #
  # Examples
  # --------
  # ```
  # t = Tensor.new([3]) &.to_f
  # u = Tensor(Int8).ones_like(t) # => [0, 0, 0]
  # ```
  def self.ones_like(t : Tensor, device = CPU(T)) : Tensor(T)
    self.new(t.shape, T.new(1), Num::RowMajor, device)
  end

  # Creates a `Tensor` of a provided shape, filled with a value.  The generic type
  # is inferred from the value
  #
  # Arguments
  # ---------
  # *shape*
  #   Shape of returned `Tensor`
  #
  # Examples
  # --------
  # ```
  # t = Tensor(Int8).full([3], 1) # => [1, 1, 1]
  # ```
  def self.full(shape : Array(Int), value : T, device = CPU(T)) : Tensor(T)
    self.new(shape, value, Num::RowMajor, device)
  end

  # Creates a `Tensor` filled with a value, sharing the shape of another
  # provided `Tensor`
  #
  # Arguments
  # ---------
  # *t*
  #   `Tensor` to use for output shape
  #
  # Examples
  # --------
  # ```
  # t = Tensor.new([3]) &.to_f
  # u = Tensor.full_like(t, 3) # => [3, 3, 3]
  # ```
  def self.full_like(t : Tensor, value : T, device = CPU(T)) : Tensor(T)
    self.new(t.shape, value, Num::RowMajor, device)
  end

  # Creates a flat `Tensor` containing a monotonically increasing
  # or decreasing range.  The generic type is inferred from
  # the inputs to the method
  #
  # Arguments
  # ---------
  # *start*
  #   Beginning value for the range
  # *stop*
  #   End value for the range
  # *step*
  #   Offset between values of the range
  #
  # Examples
  # --------
  # ```
  # Tensor.range(0, 5, 2)       # => [0, 2, 4]
  # Tensor.range(5, 0, -1)      # => [5, 4, 3, 2, 1]
  # Tensor.range(0.0, 3.5, 0.7) # => [0  , 0.7, 1.4, 2.1, 2.8]
  # ```
  def self.range(start : T, stop : T, step : T, device = CPU(T)) : Tensor(T)
    if start > stop && step > 0
      raise Num::Internal::ValueError.new(
        "Range must return at least one value"
      )
    end

    r = stop - start
    n = (r / step).ceil.abs
    self.new([n.to_i], device: device) do |i|
      T.new(start + i * step)
    end
  end

  # :ditto:
  def self.range(stop : T, device = CPU(T)) : Tensor(T)
    self.range(T.new(0), stop, T.new(1), device)
  end

  # :ditto:
  def self.range(start : T, stop : T, device = CPU(T)) : Tensor(T)
    self.range(start, stop, T.new(1), device)
  end

  # Return a two-dimensional `Tensor` with ones along the diagonal,
  # and zeros elsewhere
  #
  # Arguments
  # ---------
  # *m* : Int
  #   Number of rows in the `Tensor`
  # *n* : Int?
  #   Number of columsn in the `Tensor`, defaults to `m` if nil
  # *offset* : Int
  #   Indicates which diagonal to fill with ones
  #
  # Examples
  # --------
  # ```
  # Tensor(Int32).eye(3, offset: -1)
  #
  # # [[0, 0, 0],
  # #  [1, 0, 0],
  # #  [0, 1, 0]]
  #
  # Tensor(Int8).eye(2)
  #
  # # [[1, 0],
  # #  [0, 1]]
  # ```
  def self.eye(m : Int, n : Int? = nil, offset : Int = 0, device = CPU(T))
    n = n.nil? ? m : n
    Tensor.new(m, n, device: device) do |i, j|
      i == j - offset ? T.new(1) : T.new(0)
    end
  end

  # Returns an identity `Tensor` with ones along the diagonal,
  # and zeros elsewhere
  #
  # Arguments
  # ---------
  # *n* : Number of rows and columns in output
  #
  # Examples
  # --------
  # ```
  # Tensor(Int8).identity(2)
  #
  # # [[1, 0],
  # #  [0, 1]]
  # ```
  def self.identity(n : Int, device = CPU(T))
    self.new(n, n, device: device) do |i, j|
      i == j ? T.new(1) : T.new(0)
    end
  end
end
