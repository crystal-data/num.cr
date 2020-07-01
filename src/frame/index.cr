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

# :nodoc:
class Hash(K, V)
  def entries
    @entries
  end
end

# An `Index` implements a `Hash` that allows rows to be quickly
# looked up in a `Frame` or `Series` in constant time.
#
# Many types of indexes will be supported, but currently only
# `Integer` indexes are implemented.
class Index(T)
  @data : Hash(T, Int32)

  # Initialize an `Index` from a `Hash`.  This should
  # only ever be used by internal methods
  private def initialize(@data : Hash(T, Int32))
  end

  # Create a range index from a size.  This index will always
  # monotonically increase and be unique.
  #
  # Arguments
  # ---------
  # `n` : Int
  #   Size of the index
  #
  # Examples
  # --------
  # ```
  # Index.range(5) # => Index[0, 1, 2, 3, 4]
  # ```
  def self.range(n : Int)
    d = Hash(Int32, Int32).new
    n.times do |i|
      d[i] = i
    end
    new(d)
  end

  # Find the corresponding row value of an index at a provided
  # index value.
  #
  # Arguments
  # ---------
  # `i` : T
  #   Key to lookup
  #
  # Examples
  # ```
  # i = Index.range(5)
  # i[0] # => 0
  # ```
  def [](i : T)
    @data[i]
  end

  # Returns the appropriate `Entry` of an index at a given
  # integer value.  This relies on the fact that Crystal
  # hashes are insertion sorted.
  def iat(index : Int)
    @data.entries[index]
  end

  # :nodoc:
  def each
    @data.each do |k, v|
      yield k, v
    end
  end

  # :nodoc:
  def max_repr_width
    w = 0
    each do |k, _|
      l = Num::Internal.format(k).size
      if l > w
        w = l
      end
    end
    w
  end
end
