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

require "./internal/iteration"
require "./internal/matrix"
require "./internal/utils"

class Num::Sparse::CSR(T) < Num::Sparse::Matrix(T)
  getter m : Int32
  getter n : Int32
  getter nnz : Int32
  getter rows : Array(Int32)
  getter cols : Array(Int32)
  getter vals : Array(T)

  # Brief description of initialize
  #
  # Arguments
  # ---------
  # rows : Array(Int)
  #   Brief description of rows : Array(Int)
  # cols : Array(Int)
  #   Brief description of cols : Array(Int)
  # vals : Array(T)
  #   Brief description of vals : Array(T)
  # num_cols : Int
  #   Brief description of num_cols : Int
  #
  # Returns
  # -------
  # nil
  #
  # Examples
  # --------
  def initialize(rows : Array(Int), cols : Array(Int), @vals : Array(T), num_cols : Int)
    @rows = rows.map &.to_i
    @cols = cols.map &.to_i
    @m = @rows.size - 1
    @n = num_cols
    @nnz = @vals.size
  end

  # Iterates through each elemnent of a CSR matrix in
  # Row Major order, yielding each element and its
  # index
  #
  # Arguments
  # ---------
  #
  # Returns
  # -------
  # nil
  #
  # Examples
  # --------
  def each_with_index
    a = self
    init_csr_iteration a, T
    a_val = T.new(0)
    @m.times do |i|
      a_max = a.rows[i + 1]
      @n.times do |j|
        advanced_csr_iteration a
        yield a_val, i, j
      end
    end
  end

  # Yield each element of a CSR matrix in row-major
  # order
  #
  # Arguments
  # ---------
  #
  # Returns
  # -------
  # nil
  #
  # Examples
  # --------
  def each
    each_with_index do |el, _, _|
      yield el
    end
  end

  # Map a function across a CSR matrix
  #
  # Arguments
  # ---------
  # &block : T -> U
  #   Function to map across the matrix
  #
  # Returns
  # -------
  # Num::Sparse::CSR(U)
  #
  # Examples
  # --------
  def map(&block : T -> U) : Num::Sparse::CSR(U) forall U
    new_vals = @vals.map do |el|
      yield el
    end

    Num::Sparse::CSR(U).new(@rows, @cols, new_vals, @n)
  end

  # Map a method along two CSR Sparse matrices
  #
  # Arguments
  # ---------
  # b : Num::Sparse::CSR(U)
  #   Right hand side of the operation
  # &block : T, U -> V
  #   Callback to apply to each pairwise element
  #
  # Returns
  # -------
  # Num::Sparse::CSR(V)
  #
  # Examples
  # --------
  def map(b : Num::Sparse::CSR(U), &block : T, U -> V) : Num::Sparse::CSR(V) forall U, V
    new_rows = [0]
    new_cols = [] of Int32
    new_vals = [] of V

    a = self

    init_csr_iteration a, T
    init_csr_iteration b, U

    a_val = T.new(0)
    b_val = U.new(0)

    n = 0

    @m.times do |i|
      a_max = a.rows[i + 1]
      b_max = b.rows[i + 1]

      @n.times do |j|
        advanced_csr_iteration a
        advanced_csr_iteration b
        result = yield a_val, b_val
        add_csr_vals
      end
      add_csr_rows
    end

    Num::Sparse::CSR(V).new(new_rows, new_cols, new_vals, @n)
  end

  def map(b : Num::Sparse::Matrix(U), &block : T, U -> V) : Num::Sparse::CSR(V) forall U, V
    map(self, b.to_csr) do |i, j|
      yield i, j
    end
  end

  # Converts a sparse CSR Matrix
  # to a dense Tensor.
  #
  # Arguments
  # ---------
  #
  # Returns
  # -------
  # Tensor(T)
  #
  # Examples
  # --------
  def to_tensor : Tensor(T)
    ret = Tensor(T).new([@m, @n])
    data = ret.to_unsafe
    i = 0
    each do |el|
      data[i] = el
      i += 1
    end
    ret
  end

  # Returns self, since self is already a
  # CSR matrix
  #
  # Arguments
  # ---------
  #
  # Returns
  # -------
  # Num::Sparse::CSR(T)
  #
  # Examples
  # --------
  def to_csr : Num::Sparse::CSR(T)
    self
  end

  # Converts a CSR matrix to CSC format
  #
  # Arguments
  # ---------
  #
  # Returns
  # -------
  # Num::Sparse::CSC(T)
  #
  # Examples
  # --------
  def to_csc : Num::Sparse::CSC(T)
    args = Num::Sparse.csr_to_csc(@m, @n, @nnz, @rows, @cols, @vals)
    Num::Sparse::CSC(T).new(*args)
  end
end
