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

require "./cl_tensor"
require "../libs/clblast"
require "./creation"

class ClTensor(T)
  # :nodoc:
  macro blast(name, *args, prefix = "")
    {%
      if T == Float32
        typ = :S.id
      elsif T == Float64
        typ = :D.id
      end
    %}
    event = Pointer(Void).malloc(1).unsafe_as(LibCL::ClEvent)
    queue = Num::ClContext.instance.queue
    LibBlast.clblast_{{prefix.id}}{{typ}}{{name}}({{*args}}, pointerof(queue), pointerof(event))
    Cl.check LibCL.cl_wait_for_events(1, pointerof(event))
    Cl.check LibCL.cl_release_event(event)
  end

  # SCAL scales a vector by a constant.
  # Uses unrolled loops for increment equal to 1.
  #
  # Arguments
  # ---------
  # *a* : Number
  #   Scalar alpha
  #
  # Examples
  # --------
  # ```
  # a = ClTensor(Float32).ones([3])
  # a.scal!(3)
  # ```
  def scal!(a : Number)
    blast(scal, @size, T.new(a), self.to_unsafe, 0, 1)
  end

  # :ditto:
  def scal(a : Number)
    c = self.copy
    c.scal!(a)
    c
  end

  # AXPY constant times a vector plus a vector.
  # uses unrolled loops for increments equal to one.
  #
  # Arguments
  # ---------
  # *a* : Number
  #   Scalar alpha
  # *c* : ClTensor
  #   Argument to add
  #
  # Examples
  # --------
  # ```
  # a = ClTensor(Float32).ones([3])
  # b = ClTensor(Float32).ones([3])
  # c = 1.45
  # a.axpy!(c, b)
  # ```
  def axpy!(a : Number, c : ClTensor(T))
    blast(axpy, @size, T.new(a), c.to_unsafe, 0, 1, self.to_unsafe, 0, 1)
  end

  # :ditto:
  def axpy(a : Number, c : ClTensor(T))
    r = self.copy
    r.axpy!(a, c)
    r
  end

  # DOT forms the dot product of two vectors.
  # uses unrolled loops for increments equal to one.
  #
  # This method treats all inputs as though they were one-dimensional
  #
  # Arguments
  # ---------
  # *c* : ClTensor
  #   Argument to dot against self
  #
  # Examples
  # --------
  # ```
  # a = ClTensor(Float32).ones([3])
  # b = ClTensor(Float32).ones([3])
  # c = a.dot(b)
  # ```
  def dot(c : ClTensor(T))
    u = ClTensor(T).new([1])
    blast(dot, @size, u.to_unsafe, 0, self.to_unsafe, 0, 1, c.to_unsafe, 0, 1)
    u
  end

  # NRM2 returns the euclidean norm of a vector via the function
  # name, so that
  #
  # DNRM2 := sqrt( x'*x )
  #
  # Treats `ClTensor` as though they were one-dimensional
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # a = ClTensor(Float32).new([3])
  # a.nrm2
  # ```
  def nrm2
    u = ClTensor(T).new([1])
    blast(nrm2, @size, u.to_unsafe, 0, self.to_unsafe, 0, 1)
    u
  end

  # COPY copies a vector, x, to a vector, y.
  # uses unrolled loops for increments equal to 1.
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # a = ClTensor(Float32).ones([3, 3, 2])
  # b = a.copy
  # b.shape # => [3, 3, 2]
  # ```
  def copy
    c = ClTensor(T).new(@shape)
    blast(copy, @size, self.to_unsafe, 0, 1, c.to_unsafe, 0, 1)
    c
  end

  # ASUM sums the absolute values of the elements of a
  # vector
  #
  # Arguments
  # ---------
  #
  # Examples
  # --------
  # ```
  # a = ClTensor(Flaot32).new([3, 3, 2])
  # a.asum
  # ```
  def asum
    u = ClTensor(T).new([1])
    blast(asum, @size, u.to_unsafe, 0, self.to_unsafe, 0, 1)
    u
  end

  # GEMM  performs one of the matrix-matrix operations
  #
  #    C := alpha*op( A )*op( B ) + beta*C,
  #
  # where  op( X ) is one of
  #
  #    op( X ) = X   or   op( X ) = X**T,
  #
  # alpha and beta are scalars, and A, B and C are matrices, with op( A )
  # an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix.
  #
  # Arguments
  # ---------
  # ```
  # *b* : ClTensor
  #   Second operand for the matrix multiplication
  # ```
  #
  # Examples
  # --------
  # ```
  # a = ClTensor(Float32).ones([3, 3])
  # b = a.copy
  # puts a.gemm(b).cpu
  #
  # # [[3, 3, 3],
  # #  [3, 3, 3],
  # #  [3, 3, 3]]
  # ```
  def gemm(b : ClTensor(T))
    i, j = @shape
    x, y = b.shape
    c = ClTensor(T).new([i, y])
    blast(
      gemm,
      LibBlast::CLBlastLayout::CLBlastLayoutRowMajor,
      LibBlast::CLBlastTranspose::CLBlastTransposeNo,
      LibBlast::CLBlastTranspose::CLBlastTransposeNo,
      i,
      y,
      j,
      1.0,
      self.to_unsafe,
      0,
      j,
      b.to_unsafe,
      0,
      y,
      0.0,
      c.to_unsafe,
      0,
      y
    )
    c
  end
end
