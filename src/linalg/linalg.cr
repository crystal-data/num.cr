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

class Tensor(T, S)
  # Computes the upper triangle of a `Tensor`.  Zeros
  # out values below the `k`th diagonal
  #
  # Arguments
  # ---------
  # *k* : Int
  #   Diagonal
  #
  # Examples
  # --------
  # ```
  # a = Tensor(Int32).ones([3, 3])
  # a.triu!
  # a
  #
  # # [[1, 1, 1],
  # #  [0, 1, 1],
  # #  [0, 0, 1]]
  # ```
  def triu!(k : Int = 0)
    self.each_pointer_with_index do |e, i|
      m = i // @shape[1]
      n = i % @shape[1]
      e.value = m > n - k ? T.new(0) : e.value
    end
  end

  # :ditto:
  def triu(k : Int = 0)
    t = self.dup
    t.triu!(k)
    t
  end

  # Computes the lower triangle of a `Tensor`.  Zeros
  # out values above the `k`th diagonal
  #
  # Arguments
  # ---------
  # *k* : Int
  #   Diagonal
  #
  # Examples
  # --------
  # ```
  # a = Tensor(Int32).ones([3, 3])
  # a.tril!
  # a
  #
  # # [[1, 0, 0],
  # #  [1, 1, 0],
  # #  [1, 1, 1]]
  # ```
  def tril!(k : Int = 0)
    self.each_pointer_with_index do |e, i|
      m = i // @shape[1]
      n = i % @shape[1]
      e.value = m < n - k ? T.new(0) : e.value
    end
  end

  # :ditto:
  def tril(k : Int = 0)
    t = self.dup
    t.tril!(k)
    t
  end

  # Cholesky decomposition.
  #
  # Return the Cholesky decomposition, L * L.H, of the square matrix a, where
  # L is lower-triangular and .H is the conjugate transpose operator (which
  # is the ordinary transpose if a is real-valued). a must be Hermitian
  # (symmetric if real-valued) and positive-definite. Only L is actually
  # returned.
  #
  # Arguments
  # ---------
  # *lower*
  #   Triangular of decomposition to return
  #
  # Examples
  # --------
  # ```
  # t = [[2, -1, 0], [-1, 2, -1], [0, -1, 2]].to_tensor.astype(Float32)
  # t.cholesky
  #
  # # [[ 1.414,    0.0,    0.0],
  # #  [-0.707,  1.225,    0.0],
  # #  [   0.0, -0.816,  1.155]]
  # ```
  def cholesky!(*, lower = true)
    assert_square_matrix
    assert_fortran

    char = lower ? 'L' : 'U'
    lapack(potrf, char.ord.to_u8, shape[0], to_unsafe, shape[0])
    lower ? tril! : triu!
  end

  # DOT forms the dot product of two vectors.
  # Uses unrolled loops for increments equal to one.
  #
  # Arguments
  # ---------
  # *u* : Tensor
  #   Right hand side of the dot product
  #
  # Examples
  # --------
  # ```
  # a = [1, 2, 3, 4, 5].to_tensor
  # a.dot(a) # => 55.0
  # ```
  def dot(u : Tensor(T, S))
    self.assert_is_vector
    u.assert_is_vector
    result = Tensor(T, S).new([1])

    {% if S < OCL %}
      blast(dot, @size, result.to_unsafe, 0, self.to_unsafe, @offset, @strides[0], u.to_unsafe, u.offset, u.strides[0])
    {% else %}
      dotvalue = blas_call(
        dot,
        @size,
        self.to_unsafe,
        @strides[0],
        u.to_unsafe,
        u.strides[0]
      )
      result.to_unsafe.value = dotvalue
    {% end %}
    result
  end

  # Computes a matrix multiplication between two `Tensors`.  The `Tensor`s
  # must be two dimensional with compatible shapes.  Currently
  # only Float and Complex `Tensor`s are supported, as BLAS is used
  # for this operation
  #
  # Arguments
  # ---------
  # *other* : Tensor(T)
  #   The right hand side of the operation
  #
  # Examples
  # --------
  # ```
  # Num::Rand.set_seed(0)
  # a = Tensor.random(0.0...10.0, [3, 3])
  # a.matmul(a)
  #
  # # [[28.2001, 87.4285, 30.5423],
  # #  [12.4381, 30.9552, 26.2495],
  # #  [34.0873, 73.5366, 40.5504]]
  # ```
  def matmul(other : Tensor(T, S), output : Tensor(T, S)? = nil)
    self.assert_is_matrix
    other.assert_is_matrix

    # unless self.shape[1] == other.shape[0]
    #   raise "Invalid shapes for matrix multiplication: #{@shape}, #{other.shape}"
    # end

    if output.nil?
    else
      unless output.shape == [self.shape[0], other.shape[1]]
        raise "Invalid output size"
      end
    end

    {% if S < OCL %}
      unless (self.is_c_contiguous && other.is_c_contiguous) || (self.is_f_contiguous || other.is_f_contiguous)
        raise "Inputs must be contiguous"
      end
      a = self
      b = other

      m = a.shape[0]
      n = b.shape[1]
      k = a.shape[1]
      lda = self.is_c_contiguous ? a.shape[1] : a.shape[0]
      ldb = other.is_c_contiguous ? b.shape[1] : b.shape[0]

      if output.nil?
        result = Tensor(T, S).new([m, n])
      else
        result = output
      end

      blast(
        gemm,
        self.is_c_contiguous ? LibBlast::CLBlastLayout::CLBlastLayoutRowMajor : LibBlast::CLBlastLayout::CLBlastLayoutColMajor,
        LibBlast::CLBlastTranspose::CLBlastTransposeNo,
        LibBlast::CLBlastTranspose::CLBlastTransposeNo,
        m,
        n,
        k,
        T.new(1.0),
        self.to_unsafe,
        @offset,
        k,
        other.to_unsafe,
        other.offset,
        n,
        0.0,
        result.to_unsafe,
        0,
        n
      )
      result
    {% else %}
      a = self.is_c_contiguous || self.is_f_contiguous ? self : self.dup(Num::RowMajor)
      b = other.is_c_contiguous || other.is_c_contiguous ? other : other.dup(Num::RowMajor)
      m = a.shape[0]
      n = b.shape[1]
      k = a.shape[1]
      lda = self.is_c_contiguous ? a.shape[1] : a.shape[0]
      ldb = other.is_c_contiguous ? b.shape[1] : b.shape[0]

      if output.nil?
        dest = Tensor(T, S).new([m, n])
      else
        dest = output
      end

      a_trans = self.is_c_contiguous ? LibCblas::CblasTranspose::CblasNoTrans : LibCblas::CblasTranspose::CblasTrans
      b_trans = other.is_c_contiguous ? LibCblas::CblasTranspose::CblasNoTrans : LibCblas::CblasTranspose::CblasTrans
      alpha = T.new(1.0)
      c_alpha = T.new(0.0)
      blas(
        ge,
        mm,
        a_trans,
        b_trans,
        m,
        n,
        k,
        blas_const(alpha),
        a.to_unsafe,
        lda,
        b.to_unsafe,
        ldb,
        blas_const(c_alpha),
        dest.to_unsafe,
        dest.shape[1]
      )
      dest
    {% end %}
  end

  private def assert_square_matrix
    raise "Matrix must be square" unless rank == 2 && @shape[0] == @shape[1]
  end

  private def assert_fortran
    raise "Matrix must be fortran contiguous" unless self.is_f_contiguous
  end

  def assert_is_vector
    raise "Inputs must be vectors" unless self.rank == 1
  end

  def assert_is_matrix
    raise "Input must be a matrix" unless self.rank == 2
  end
end
