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

describe Tensor do
  it "calculates upper triangular of a matrix inplace" do
    a = Tensor(Int32, CPU(Int32)).ones([3, 3])
    a.triu!

    expected = [[1, 1, 1], [0, 1, 1], [0, 0, 1]].to_tensor
    Num::Testing.tensor_equal(a, expected).should be_true
  end

  it "calculates upper triangular of a matrix" do
    a = Tensor(Int32, CPU(Int32)).ones([3, 3])
    b = a.triu

    expected = [[1, 1, 1], [0, 1, 1], [0, 0, 1]].to_tensor
    Num::Testing.tensor_equal(b, expected).should be_true
  end

  it "calculates upper triangular of a matrix inplace with offset" do
    a = Tensor(Int32, CPU(Int32)).ones([3, 3])
    a.triu!(1)

    expected = [[0, 1, 1], [0, 0, 1], [0, 0, 0]].to_tensor
    Num::Testing.tensor_equal(a, expected).should be_true
  end

  it "calculates upper triangular of a matrix with offset" do
    a = Tensor(Int32, CPU(Int32)).ones([3, 3])
    b = a.triu(1)

    expected = [[0, 1, 1], [0, 0, 1], [0, 0, 0]].to_tensor
    Num::Testing.tensor_equal(b, expected).should be_true
  end

  it "calculates lower triangular of a matrix inplace" do
    a = Tensor(Int32, CPU(Int32)).ones([3, 3])
    a.tril!

    expected = [[1, 0, 0], [1, 1, 0], [1, 1, 1]].to_tensor
    Num::Testing.tensor_equal(a, expected).should be_true
  end

  it "calculates lower triangular of a matrix" do
    a = Tensor(Int32, CPU(Int32)).ones([3, 3])
    b = a.tril

    expected = [[1, 0, 0], [1, 1, 0], [1, 1, 1]].to_tensor
    Num::Testing.tensor_equal(b, expected).should be_true
  end

  it "calculates lower triangular of a matrix inplace with offset" do
    a = Tensor(Int32, CPU(Int32)).ones([3, 3])
    a.tril!(-1)

    expected = [[0, 0, 0], [1, 0, 0], [1, 1, 0]].to_tensor
    Num::Testing.tensor_equal(a, expected).should be_true
  end

  it "calculates lower triangular of a matrix with offset" do
    a = Tensor(Int32, CPU(Int32)).ones([3, 3])
    b = a.tril(-1)

    expected = [[0, 0, 0], [1, 0, 0], [1, 1, 0]].to_tensor
    Num::Testing.tensor_equal(b, expected).should be_true
  end

  it "calculates the cholesky decomposition", tags: "blas" do
    t = [[2, -1, 0], [-1, 2, -1], [0, -1, 2]].to_tensor.as_type(Float32)
    t = t.dup(Num::ColMajor)
    t.cholesky!

    expected = [[1.414, 0.0, 0.0], [-0.707, 1.225, 0.0], [0.0, -0.816, 1.155]].to_tensor
    Num::Testing.tensor_equal(t, expected, tolerance: 1e-3).should be_true
  end

  it "calculates the QR decomposition of a matrix", tags: "blas" do
    t = [[0, 1], [1, 1], [1, 1], [2, 1]].to_tensor.as_type(Float32)
    q, r = t.qr

    q_expected = [[0, 0.866025],
                  [-0.408248, 0.288675],
                  [-0.408248, 0.288675],
                  [-0.816497, -0.288675]].to_tensor
    r_expected = [[-2.44949, -1.63299],
                  [0, 1.1547],
                  [0, 0],
                  [0, 0]].to_tensor

    Num::Testing.tensor_equal(q, q_expected, tolerance: 1e-3).should be_true
    Num::Testing.tensor_equal(r, r_expected, tolerance: 1e-3).should be_true
  end

  it "calculates the SVD decomposition of a matrix", tags: "blas" do
    t = [[0, 1], [1, 1], [1, 1], [2, 1]].to_tensor.as_type(Float32)
    a, b, c = t.svd

    a_exp = [[-0.203749, 0.841716, -0.330613, 0.375094],
             [-0.464705, 0.184524, -0.19985, -0.842651],
             [-0.464705, 0.184524, 0.861075, 0.092463],
             [-0.725662, -0.472668, -0.330613, 0.375094]].to_tensor
    b_exp = [3.02045, 0.936426].to_tensor
    c_exp = [[-0.788205, -0.615412],
             [-0.615412, 0.788205]].to_tensor

    Num::Testing.tensor_equal(a, a_exp, tolerance: 1e-3).should be_true
    Num::Testing.tensor_equal(b, b_exp, tolerance: 1e-3).should be_true
    Num::Testing.tensor_equal(c, c_exp, tolerance: 1e-3).should be_true
  end

  it "calculates the eigenvalues + right eigenvectors of a symmetric matrix", tags: "blas" do
    t = [[0, 1], [1, 1]].to_tensor.as_type(Float32)
    w, v = t.eigh

    w_exp = [-0.618034, 1.61803].to_tensor
    v_exp = [[-0.850651, 0.525731],
             [0.525731, 0.850651]].to_tensor

    Num::Testing.tensor_equal(w, w_exp, tolerance: 1e-3).should be_true
    Num::Testing.tensor_equal(v, v_exp, tolerance: 1e-3).should be_true
  end

  it "calculates the eigenvalues + right eigenvectors of a general matrix", tags: "blas" do
    t = [[0, 1], [1, 1]].to_tensor.as_type(Float32)
    w, v = t.eig

    w_exp = [-0.618034, 1.61803].to_tensor
    v_exp = [[-0.850651, 0.525731],
             [-0.525731, -0.850651]].to_tensor

    Num::Testing.tensor_equal(w, w_exp, tolerance: 1e-3).should be_true
    Num::Testing.tensor_equal(v, v_exp, tolerance: 1e-3).should be_true
  end

  it "calculates the eigenvals of a symmetric matrix", tags: "blas" do
    t = [[0, 1], [1, 1]].to_tensor.as_type(Float32)
    result = t.eigvalsh

    t_exp = [-0.618034, 1.61803].to_tensor

    Num::Testing.tensor_equal(result, t_exp, tolerance: 1e-3).should be_true
  end

  it "calculates the eigenvals of a general matrix", tags: "blas" do
    t = [[0, 1], [1, 1]].to_tensor.as_type(Float32)
    result = t.eigvals

    t_exp = [-0.618034, 1.61803].to_tensor

    Num::Testing.tensor_equal(result, t_exp, tolerance: 1e-3).should be_true
  end

  {% unless flag?(:darwin) %}
    it "calculates the matrix norm" do
      t = [[0, 1], [1, 1], [1, 1], [2, 1]].to_tensor.as_type(Float32)
      expected = [3.1622777].to_tensor

      Num::Testing.tensor_equal(t.norm('F'), expected, tolerance: 1e-3).should be_true
    end
  {% end %}

  it "calculates the matrix determinant", tags: "blas" do
    t = [[1, 2], [3, 4]].to_tensor.as_type(Float32)
    Num::Testing.tensor_equal(t.det, [-2].to_tensor).should be_true
  end

  it "calculates the multiplicative inverse of a square matrix", tags: "blas" do
    t = [[1, 2], [3, 4]].to_tensor.as_type(Float32)
    i = t.inv

    i_exp = [[-2, 1],
             [1.5, -0.5]].to_tensor

    Num::Testing.tensor_equal(i, i_exp, tolerance: 1e-2).should be_true
  end

  it "solves a linear matrix equation", tags: "blas" do
    a = [[3, 1], [1, 2]].to_tensor.as_type(Float32)
    b = [9, 8].to_tensor.as_type(Float32)
    result = a.solve(b)
    expected = [2, 3].to_tensor

    Num::Testing.tensor_equal(result, expected).should be_true
  end

  it "calculates the dot product of two vectors", tags: "blas" do
    a = [1, 2, 3, 4, 5].to_tensor.as_type(Float32)
    b = a.dot(a)

    expected = [55.0].to_tensor

    Num::Testing.tensor_equal(b, expected).should be_true
  end

  it "calculates the dot product of two vectors opencl", tags: ["clblast", "opencl"] do
    a = [1_f32, 2_f32, 3_f32, 4_f32, 5_f32].to_tensor(OCL)
    b = a.dot(a)

    expected = [55.0].to_tensor

    Num::Testing.tensor_equal(b.cpu, expected).should be_true
  end

  it "calculates the dot product of two matrices", tags: "blas" do
    a = [[1, 1], [1, 1]].to_tensor.as_type(Float32)
    b = a.matmul(a)

    expected = [[2, 2], [2, 2]].to_tensor

    Num::Testing.tensor_equal(b, expected).should be_true
  end

  it "calculates the dot product of two matrices opencl", tags: ["clblast", "opencl"] do
    a = [[1, 1] of Float32, [1, 1] of Float32].to_tensor(OCL)
    b = a.matmul(a)

    expected = [[2, 2], [2, 2]].to_tensor

    Num::Testing.tensor_equal(b.cpu, expected).should be_true
  end
end
