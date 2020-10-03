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

module Num::Sparse
  extend self

  # Brief description of csrtocsc
  #
  # Arguments
  # ---------
  # m : Int32
  #   Brief description of m : Int32
  # n : Int32
  #   Brief description of n : Int32
  # nnz : Int32
  #   Brief description of nnz : Int32
  # rows : Array(Int32)
  #   Brief description of rows : Array(Int32)
  # cols : Array(Int32)
  #   Brief description of cols : Array(Int32)
  # vals : Array(U)
  #   Brief description of vals : Array(U)
  #
  # Returns
  # -------
  # Tuple(Array(Int32),
  #
  # Examples
  # --------
  def csr_to_csc(
    m : Int32,
    n : Int32,
    nnz : Int32,
    rows : Array(Int32),
    cols : Array(Int32),
    vals : Array(U)
  ) : Tuple(Array(Int32), Array(Int32), Array(U), Int32) forall U
    new_cols = Array(Int32).new(n + 1, 0)
    new_rows = Array(Int32).new(nnz, 0)
    new_vals = Array(U).new(nnz, U.new(0))

    nnz.times do |i|
      new_cols[cols[i]] += 1
    end

    cumsum = 0
    n.times do |col|
      temp = new_cols[col]
      new_cols[col] = cumsum
      cumsum += temp
    end
    new_cols[n] = nnz

    m.times do |row|
      rows[row].step(to: rows[row + 1] - 1) do |jj|
        col = cols[jj]
        dest = new_cols[col]
        new_rows[dest] = row
        new_vals[dest] = vals[jj]
        new_cols[col] += 1
      end
    end

    last = 0
    n.times do |col|
      temp = new_cols[col]
      new_cols[col] = last
      last = temp
    end

    {new_rows, new_cols, new_vals, m}
  end
end
