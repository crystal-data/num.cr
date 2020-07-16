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

module SciKit
  extend self

  # Construct a circulant matrix.
  #
  # Arguments
  # ---------
  # *c* : Tensor | Enumerable
  #   First column of the matrix
  #
  # Examples
  # --------
  # ```
  # SciKit.circulant([1, 2, 3])
  #
  # # [[1, 3, 2],
  # #  [2, 1, 3],
  # #  [3, 2, 1]]
  # ```
  def circulant(c : Tensor | Enumerable)
    a_c = c.to_tensor.flat
    c_ext = Num.concat(a_c[{..., -1}], a_c[{1..., -1}])
    l = a_c.size
    n = c_ext.strides[0]
    c_ext.as_strided([l, l], [-n, n]).dup
  end

  # Construct a Toeplitz matrix.

  # The Toeplitz matrix has constant diagonals, with c as its
  # first column and r as its first row. If r is not given,
  # r == conjugate(c) is assumed.
  #
  # Arguments
  # ---------
  # *c* : Tensor | Enumerable
  #   First column of the Matrix, assumed flat
  # *r* : Tensor | Enumerable
  #   First row of the Matrix, assumed flat
  #
  # Examples
  # --------
  # ```
  # SciKit.toeplitz([1, 2, 3], [1, 4, 5, 6])
  #
  # # [[1, 4, 5, 6],
  # #  [2, 1, 4, 5],
  # #  [3, 2, 1, 4]]
  # ```
  def toeplitz(c : Tensor | Enumerable, r : Tensor | Enumerable)
    c_t = c.to_tensor.flat
    r_t = r.to_tensor.flat
    vals = Num.concat(c_t[{..., -1}], r_t[1...])
    out_shp = [c_t.size, r_t.size]
    n = vals.strides[0]
    vals.as_strided(out_shp, [-n, n]).dup
  end

  # :ditto:
  def toeplitz(c : Tensor | Enumerable)
    c_t = c.to_tensor.flat
    r = Num.conj(c_t)
    toeplitz(c_t, r)
  end

  # Construct a Hankel matrix.
  #
  # The Hankel matrix has constant anti-diagonals, with c as
  # its first column and r as its last row. If r is not given,
  # then r = zeros_like(c) is assumed.
  #
  # Arguments
  # ---------
  # *c* : Tensor
  #   The first column of the matrix
  # *r* : Tensor
  #   Last row of the matrix
  #
  # Examples
  # --------
  # ```
  # puts SciKit.hankel([1, 2, 99])
  #
  # # [[ 1,  2, 99],
  # #  [ 2, 99,  0],
  # #  [99,  0,  0]]
  # ```
  def hankel(c : Tensor | Enumerable, r : Tensor | Enumerable)
    c_t = c.to_tensor.flat
    r_t = r.to_tensor.flat
    vals = Num.concat(c_t, r[1...])
    out_shp = [c.size, r.size]
    n = vals.strides[0]
    vals.as_strided(out_shp, [n, n]).dup
  end

  # :ditto:
  def hankel(c : Tensor | Enumerable)
    c_t = c.to_tensor.flat
    r = c_t.class.zeros_like(c_t)
    hankel(c_t, r)
  end

  # Construct an Hadamard matrix.
  #
  # Constructs an n-by-n Hadamard matrix, using Sylvester’s
  # construction. n must be a power of 2.
  #
  # Arguments
  # ---------
  # *n* : Int
  #   The order of the matrix, must be a power of 2
  #
  # Examples
  # --------
  # ```
  # SciKit.hadamard(2, dtype: Complex)
  #
  # # [[1+0j , 1+0j ],
  # #  [1+0j , -1+0j]]
  # ```
  def hadamard(n : Int, dtype : U.class = Int32) forall U
    if n < 1
      lg2 = 0
    else
      lg2 = Math.log2(n)
    end
    if 2 ** lg2 != n
      raise Num::Internal::ValueError.new(
        "n must be a positive power of 2"
      )
    end
    h = [[1]].to_tensor.as_type(U)
    lg2.to_i.times do
      h = Num.v_concat(Num.h_concat(h, h), Num.h_concat(h, h.map { |i| -i }))
    end
    h
  end

  # Create a block diagonal matrix from provided arrays.
  #
  # Given the inputs A, B and C, the output will have these
  # arrays arranged on the diagonal
  #
  # Arguments
  # ---------
  # *arrs* : Tensor | Enumerable
  #   Arguments to place along the diagonal, must be a common dtype
  #
  # Examples
  # --------
  # ```
  # puts SciKit.block_diag([1, 2], [[5, 6], [7, 8]])
  #
  # # [[1, 2, 0, 0],
  # #  [0, 0, 5, 6],
  # #  [0, 0, 7, 8]]
  # ```
  def block_diag(*arrs : Tensor | Enumerable)
    ts = arrs.map &.to_tensor.with_dims(2)
    shapes = (ts.map &.shape).to_a
    s0 = Num.sum(shapes, 0).to_a
    t = ts[0].class.zeros(s0)
    r, c = 0, 0
    shapes.each_with_index do |s, i|
      rr, cc = s
      t[r...r + rr, c...c + cc] = ts[i]
      r += rr
      c += cc
    end
    t
  end
end
