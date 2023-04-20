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

macro test_unary_operator(operator, backend = :cpu)
  it "Maps the {{ operator.id }} unary operator across one Tensor" do
    a = [1, 2, 3]

    a_tensor = a.to_tensor
    {% if backend != :cpu %}
      a_tensor = a_tensor.{{backend.id}}
    {% end %}
    result = {{ operator.id }} a_tensor
    expected = a.map { |i|  {{ operator.id }} i }

    result.to_a.should eq expected
  end

end

macro test_operator(operator)
  it "Maps the {{ operator.id }} operator across two Tensors" do
    a = [1, 2, 3]
    b = [4, 5, 6]

    result = a.to_tensor {{ operator.id }} b.to_tensor
    expected = a.zip(b).map { |i, j| i {{ operator.id }} j }

    result.to_a.should eq expected
  end

  it "Maps the {{ operator }} operator across non-contiguous Tensors" do
    a = [0, 1, 2, 3, 4, 5]
    b = [0, 1, 2, 3, 4, 5]

    a0 = a.to_tensor[{..., 2}]
    b0 = b.to_tensor[{1..., 2}]

    a1 = [0, 2, 4]
    b1 = [1, 3, 5]

    result = a0 {{ operator.id }} b0
    expected = a1.zip(b1).map { |i, j| i {{ operator.id }} j }

    result.to_a.should eq expected
  end

  it "Broadcasts the {{ operator.id }} operator across two Tensors" do
    a = [1, 2, 3].to_tensor
    b = [[1, 2, 3], [3, 4, 5]].to_tensor

    a0 = [[1, 2, 3], [1, 2, 3]]
    b0 = [[1, 2, 3], [3, 4, 5]]

    result = a {{ operator.id }} b

    expected = a0.zip(b0).map { |i, j| i.zip(j).map { |i0, j0| i0 {{ operator.id }} j0 } }

    result.to_a.should eq expected.flatten
  end
end

macro test_builtin(fn)
  it "Maps the {{ fn.id }} method across two Tensors" do
    a = [1, 1, 1]

    result = Num.{{ fn.id }}(a.to_tensor)
    expected = a.map { |i| Math.{{ fn.id }}(i) }

    result.to_a.should eq expected
  end

  it "Maps the {{ fn.id }} method across non-contiguous Tensors" do
    a = [0.5, 1.0, 0.5, 1.0, 0.5, 1.0]
    a0 = a.to_tensor[{1..., 2}]
    a1 = [1.0, 1.0, 1.0]

    result = Num.{{ fn.id }}(a0)
    expected = a1.map { |i| Math.{{fn.id}}(i) }

    result.to_a.should eq expected
  end
end

describe Tensor do
  test_unary_operator :-, :opencl
  test_unary_operator :-

  test_operator :+
  test_operator :-
  test_operator :*
  test_operator :/
  test_operator ://
  test_operator :**
  test_operator :%
  test_operator :<<
  test_operator :>>
  test_operator :&
  test_operator :|
  test_operator :^
  test_operator :>
  test_operator :>=
  test_operator :==
  test_operator :!=
  test_operator :<
  test_operator :<=

  test_builtin acos
  test_builtin acosh
  test_builtin asin
  test_builtin asinh
  test_builtin atan
  test_builtin atanh
  test_builtin besselj0
  test_builtin besselj1
  test_builtin bessely0
  test_builtin bessely1
  test_builtin cbrt
  test_builtin cos
  test_builtin cosh
  test_builtin erf
  test_builtin erfc
  test_builtin exp
  test_builtin exp2
  test_builtin expm1
  test_builtin gamma
  test_builtin ilogb
  test_builtin lgamma
  test_builtin log
  test_builtin log10
  test_builtin log1p
  test_builtin log2
  test_builtin logb
  test_builtin sin
  test_builtin sinh
  test_builtin sqrt
  test_builtin tan
  test_builtin tanh
end
