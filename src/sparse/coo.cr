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

class Num::Sparse::COO(T) < Num::Sparse::Matrix(T)
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
  # t : Tensor(T)
  #   Brief description of t : Tensor(T)
  #
  # Returns
  # -------
  # nil
  #
  # Examples
  # --------
  def initialize(t : Tensor(T))
    unless t.rank == 2
      raise Num::Internal::ShapeError.new("Invalid matrix")
    end

    @rows = [] of Int32
    @cols = [] of Int32
    @vals = [] of T
    @nnz = 0

    @m, @n = t.shape
    @m.times do |i|
      @n.times do |j|
        val = t[i, j].value
        unless val == 0
          @rows << i
          @cols << j
          @vals << val
          @nnz += 1
        end
      end
    end
  end

  # Brief description of initialize
  #
  # Arguments
  # ---------
  # m : Int
  #   Brief description of m : Int
  # n : Int
  #   Brief description of n : Int
  #
  # Returns
  # -------
  # nil
  #
  # Examples
  # --------
  def initialize(m : Int, n : Int)
    @m = m.to_i
    @n = n.to_i
    @rows = [] of Int32
    @cols = [] of Int32
    @vals = [] of T
    @nnz = 0
  end

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
  def initialize(rows : Array(Int), cols : Array(Int), @vals : Array(T), shape : Array(Int))
    @rows = rows.map &.to_i
    @cols = cols.map &.to_i
    @m = shape[0].to_i
    @n = shape[1].to_i
    @nnz = @vals.size
  end

  # Iterates through each elemnent of a COO matrix in
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
    init_coo_iteration a, T
    a_val = T.new(0)
    @m.times do |i|
      @n.times do |j|
        advanced_coo_iteration a
        yield a_val, i, j
      end
    end
  end

  # Yield each element of a COO matrix in row-major
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

  # Map a function across all elements of a CSR matrix
  #
  # Arguments
  # ---------
  # &block : T -> U
  #   Brief description of &block : T -> U
  #
  # Returns
  # -------
  # Num::Sparse::CSR(U)
  #
  # Examples
  # --------
  def map(&block : T -> U) : Num::Sparse::CSR(U) forall U
    new_rows = [] of Int32
    new_cols = [] of Int32
    new_vals = [] of U

    a = self

    init_coo_iteration a, T
    a_val = T.new(0)

    @m.times do |i|
      @n.times do |j|
        advanced_coo_iteration a
        result = yield a_val
        add_coo_vals
      end
    end

    Num::Sparse::COO(U).new(new_rows, new_cols, new_vals, [@m, @n])
  end

  # Map a function across the nonzero elements
  # of a CSR matrix
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
  def nonzero_map(&block : T -> U) : Num::Sparse::CSR(U) forall U
    new_vals = @vals.map do |el|
      yield el
    end

    Num::Sparse::COO(U).new(@rows.dup, @cols.dup, new_vals, [@m, @n])
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
  def map(b : Num::Sparse::COO(U), &block : T, U -> V) : Num::Sparse::COO(V) forall U, V
    new_rows = [] of Int32
    new_cols = [] of Int32
    new_vals = [] of V

    a = self

    init_coo_iteration a, T
    init_coo_iteration b, U

    a_val = T.new(0)
    b_val = U.new(0)

    @m.times do |i|
      @n.times do |j|
        advanced_coo_iteration a
        advanced_coo_iteration b
        result = yield a_val, b_val
        add_coo_vals
      end
    end

    Num::Sparse::COO(V).new(new_rows, new_cols, new_vals, [@m, @n])
  end

  # :ditto:
  def map(b : Num::Sparse::Matrix(U), &block : T, U -> V) : Num::Sparse::COO(V) forall U, V
    map(self, b.to_coo) do |i, j|
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
    each_with_index do |el, i, j|
      ret[i, j] = el
    end
    ret
  end

  def to_coo : Num::Sparse::COO(T)
    self
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
    args = Num::Sparse.coo_to_csr(@m, @n, @nnz, @rows, @cols, @vals)
    Num::Sparse::CSR(T).new(*args)
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
    a, b, c, d = Num::Sparse.coo_to_csr(@n, @m, @nnz, @cols, @rows, @vals)
    d.reverse!
    Num::Sparse::CSC(T).new(b, a, c, d)
  end

  # Brief description of rows
  #
  # Arguments
  # ---------
  # dims : Bool = false
  #   Brief description of dims : Bool = false
  # &block : Array(T) -> U
  #   Brief description of &block : Array(T) -> U
  #
  # Returns
  # -------
  # nil
  #
  # Examples
  # --------
  def each_row(dims : Bool = false, &block : Array(T) -> U) forall U
    self.to_csr.each_row(dims) do |el|
      yield el
    end
  end

  # Brief description of cols
  #
  # Arguments
  # ---------
  # dims : Bool = false
  #   Brief description of dims : Bool = false
  # &blok : Array(T) -> U
  #   Brief description of &blok : Array(T) -> U
  #
  # Returns
  # -------
  # nil
  #
  # Examples
  # --------
  def each_column(dims : Bool = false, &block : Array(T) -> U) forall U
    self.to_csc.each_column(dims) do |el|
      yield el
    end
  end

  # Brief description of axis
  #
  # Arguments
  # ---------
  # i : Int
  #   Brief description of i : Int
  # dims : Bool = false
  #   Brief description of dims : Bool = false
  # &block : Array(T) -> U
  #   Brief description of &block : Array(T) -> U
  #
  # Returns
  # -------
  # Tensor(U)
  #
  # Examples
  # --------
  def axis(i : Int, dims : Bool = false, &block : Array(T) -> U) : Tensor(U) forall U
    if i == 0
      each_row(dims) { |el| yield el }
    else
      each_column(dims) { |el| yield el }
    end
  end
end
