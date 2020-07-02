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

  def circulant(c : Tensor | Enumerable)
    a_c = c.to_tensor.flat
    c_ext = Num.concat(a_c[{..., -1}], a_c[{1..., -1}])
    l = a_c.size
    n = c_ext.strides[0]
    c_ext.as_strided([l, l], [-n, n]).dup
  end

  def toeplitz(c : Tensor | Enumerable, r : Tensor | Enumerable)
    c_t = c.to_tensor.flat
    r_t = r.to_tensor.flat
    vals = Num.concat(c_t[{..., -1}], r_t[1...])
    out_shp = [c_t.size, r_t.size]
    n = vals.strides[0]
    vals.as_strided(out_shp, [-n, n]).dup
  end

  def toeplitz(c : Tensor | Enumerable)
    c_t = c.to_tensor.flat
    r = Num.conj(c_t)
    toeplitz(c_t, r)
  end

  def hankel(c : Tensor | Enumerable, r : Tensor | Enumerable)
    c_t = c.to_tensor.flat
    r_t = r.to_tensor.flat
    vals = Num.concat(c_t, r[1...])
    out_shp = [c.size, r.size]
    n = vals.strides[0]
    vals.as_strided(out_shp, [n, n]).dup
  end

  def hankel(c : Tensor | Enumerable)
    c_t = c.to_tensor.flat
    r = c_t.class.zeros_like(c_t)
    hankel(c_t, r)
  end

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
