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

# ND contains the basic building blocks for all other implementations
# of an ND container, whether it be a Tensor(T) or a CLTensor(T) or
# any other future storage backends.  It provides a simple way to
# calculate contiguity + flags
module Num::ND(T)
  # Returns the size of an ND container along each dimension
  #
  # ```
  # a.shape # => [2, 3, 4]
  # ```
  getter shape : Array(Int32)

  # Returns the step of an ND container along each dimension
  #
  # ```
  # a.strides # => [4, 2, 1]
  # ```
  getter strides : Array(Int32)

  # Returns the offset of an ND container's data
  #
  # ```
  # a.offset # => 0
  # ```
  getter offset : Int32

  # Returns the number of elements in an ND container
  #
  # ```
  # a.size # => 24
  # ```
  getter size : Int32

  # Returns the memory/write flags of an ND container
  #
  # ```
  # a.flags => Contiguous | Fortran | OwnData | Write
  # ```
  property flags : Num::ArrayFlags = Num::ArrayFlags::All

  # Returns the number of dimensions in an ND container
  #
  # ```
  # a.rank # => 4
  # ```
  def rank : Int32
    self.shape.size
  end

  # Checks if an ND container's data is stored in RowMajor
  # order.
  def c_contiguous? : Bool
    return true unless self.rank != 0
    if self.rank == 1
      return self.shape[0] == 1 || self.strides[0] == 1
    end
    step = 1
    rm1 = self.rank - 1
    rm1.step(to: 0, by: -1) do |i|
      dim = self.shape[i]
      return true unless dim != 0
      return false unless self.strides[i] == step
      step *= dim
    end
    true
  end

  # Checks if an ND container's data is stored in ColMajor
  # order, commonly used by linear algebra libraries
  def f_contiguous? : Bool
    return true unless self.rank != 0
    if self.rank == 1
      return self.shape[0] == 1 || self.strides[0] == 1
    end
    step = 1
    self.rank.times do |i|
      dim = self.shape[i]
      return true unless dim != 0
      return false unless self.strides[i] == step
      step *= dim
    end
    true
  end

  # Update flags to match detected contiguity of an ND
  # container.  This MUST be called in every method that
  # changes the shape or strides of an ND container
  private def update_flags(m = Num::ArrayFlags::All)
    if m.fortran?
      if self.f_contiguous?
        self.flags |= Num::ArrayFlags::Fortran
        if self.rank > 1
          self.flags &= ~Num::ArrayFlags::Contiguous
        end
      else
        self.flags &= ~Num::ArrayFlags::Fortran
      end
    end
    if m.contiguous?
      if self.c_contiguous?
        self.flags |= Num::ArrayFlags::Contiguous
        if self.rank > 1
          self.flags &= ~Num::ArrayFlags::Fortran
        end
      else
        self.flags &= ~Num::ArrayFlags::Contiguous
      end
    end
  end
end
