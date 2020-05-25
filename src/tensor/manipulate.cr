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

class Tensor(T) < AnyArray(T)
  private def triu2d(a : AnyArray(T), k)
    m, n = a.shape
    a.iter_flat_indexed do |el, idx|
      i = idx // n
      j = idx % n
      if i > j - k
        el.value = T.new(0)
      end
    end
  end

  private def tril2d(a : AnyArray(T), k)
    m, n = a.shape
    a.iter_flat_indexed do |el, idx|
      i = idx // n
      j = idx % n
      if i < j - k
        el.value = T.new(0)
      end
    end
  end

  def triu!(k = 0)
    if ndims == 2
      triu2d(self, k)
    else
      matrix_iter.each do |subm|
        triu2d(subm, k)
      end
    end
  end

  def triu(k = 0)
    ret = dup
    ret.triu!
    ret
  end

  def tril!(k = 0)
    if ndims == 2
      tril2d(self, k)
    else
      matrix_iter.each do |subm|
        tril2d(subm, k)
      end
    end
  end

  def tril(k = 0)
    ret = dup
    ret.tril!
    ret
  end

  def bincount(min_count = 0)
    if @ndims != 1
      raise NumInternal::ShapeError.new("Input must be 1-dimensional")
    end
    sz = Math.max(min_count, Num.max(self) + 1)
    ret = Pointer(Int32).malloc(sz)
    iter.each do |i|
      val = i.value
      if val < 0
        raise NumInternal::ValueError.new "All values must be positive"
      end
      ret[i.value] += 1
    end
    Tensor.new(ret, [sz], [1])
  end

  def bincount(weights : Tensor(U), min_count = 0) forall U
    if @ndims != 1
      raise NumInternal::ShapeError.new("Input must be 1-dimensional")
    end
    if @shape != weights.shape
      raise "Weights do not match input"
    end
    sz = Math.max(min_count, Num.max(self) + 1)
    ret = Pointer(U).malloc(sz)
    iter2(weights).each do |i, j|
      iv, jv = {i.value, j.value}
      if iv < 0
        raise NumInternal::ValueError.new("All values must be positive")
      end
      ret[iv] += jv
    end
    Tensor.new(ret, [sz], [1])
  end
end
