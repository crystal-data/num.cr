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

module Num::Grad
  # :nodoc:
  def subtract_backward(gradient : Tensor(U, OCL(U))) : Array(Tensor(U, OCL(U))) forall U
    negative_one = [U.new(-1)].to_tensor(OCL)
    [gradient, gradient * negative_one]
  end

  # :nodoc:
  def divide_backward(
    gradient : Tensor(U, OCL(U)),
    av : Variable(Tensor(U, OCL(U))),
    bv : Variable(Tensor(U, OCL(U)))
  ) : Array(Tensor(U, OCL(U))) forall U
    a = av.value
    b = bv.value

    r0 = gradient / b
    r1 = {% if U == Float32 %}
           Num::Float32DivideBackwardsTwoKernel.instance.call(gradient, a, b)
         {% elsif U == Float64 %}
           Num::Float64DivideBackwardsTwoKernel.instance.call(gradient, a, b)
         {% else %}
      {% raise "Invalid Dtype" %}
    {% end %}

    [r0, r1]
  end

  # :nodoc:
  def power_backward(
    gradient : Tensor(U, OCL(U)),
    av : Variable(Tensor(U, OCL(U))),
    bv : Variable(Tensor(U, OCL(U)))
  ) forall U
    r0 = {% if U == Float32 %}
           Num::Float32PowerBackwardsOneKernel.instance.call(gradient, av.value, bv.value)
         {% elsif U == Float64 %}
           Num::Float64PowerBackwardsOneKernel.instance.call(gradient, av.value, bv.value)
         {% else %}
      {% raise "Invalid Dtype" %}
    {% end %}
    r1 = {% if U == Float32 %}
           Num::Float32PowerBackwardsTwoKernel.instance.call(gradient, av.value, bv.value)
         {% elsif U == Float64 %}
           Num::Float64PowerBackwardsTwoKernel.instance.call(gradient, av.value, bv.value)
         {% else %}
      {% raise "Invalid Dtype" %}
    {% end %}

    [r0, r1]
  end

  # :nodoc:
  def exp_backward(
    gradient : Tensor(U, OCL(U)),
    av : Variable(Tensor(U, OCL(U)))
  ) forall U
    result = {% if U == Float32 %}
               Num::Float32ExpBackwardsKernel.instance.call(gradient, av.value)
             {% elsif U == Float64 %}
               Num::Float64ExpBackwardsKernel.instance.call(gradient, av.value)
             {% else %}
      {% raise "Invalid Dtype" %}
    {% end %}
    [result]
  end

  # :nodoc:
  def sin_backward(
    gradient : Tensor(U, OCL(U)),
    a : Variable(Tensor(U, OCL(U)))
  ) forall U
    result = {% if U == Float32 %}
               Num::Float32SinBackwardKernel.instance.call(gradient, a.value)
             {% elsif U == Float64 %}
               Num::Float64SinBackwardKernel.instance.call(gradient, a.value)
             {% else %}
      {% raise "Invalid Dtype" %}
    {% end %}
    [result]
  end

  # :nodoc:
  def cos_backward(
    gradient : Tensor(U, OCL(U)),
    a : Variable(Tensor(U, OCL(U)))
  ) forall U
    result = {% if U == Float32 %}
               Num::Float32CosBackwardKernel.instance.call(gradient, a.value)
             {% elsif U == Float64 %}
               Num::Float64CosBackwardKernel.instance.call(gradient, a.value)
             {% else %}
      {% raise "Invalid Dtype" %}
    {% end %}
    [result]
  end

  # :nodoc:
  def tan_backward(
    gradient : Tensor(U, OCL(U)),
    a : Variable(Tensor(U, OCL(U)))
  ) forall U
    result = {% if U == Float32 %}
               Num::Float32TanBackwardKernel.instance.call(gradient, a.value)
             {% elsif U == Float64 %}
               Num::Float64TanBackwardKernel.instance.call(gradient, a.value)
             {% else %}
      {% raise "Invalid Dtype" %}
    {% end %}
    [result]
  end

  # :nodoc:
  def asin_backward(
    gradient : Tensor(U, OCL(U)),
    a : Variable(Tensor(U, OCL(U)))
  ) forall U
    result = {% if U == Float32 %}
               Num::Float32AsinBackwardKernel.instance.call(gradient, a.value)
             {% elsif U == Float64 %}
               Num::Float64AsinBackwardKernel.instance.call(gradient, a.value)
             {% else %}
      {% raise "Invalid Dtype" %}
    {% end %}
    [result]
  end

  # :nodoc:
  def acos_backward(
    gradient : Tensor(U, OCL(U)),
    a : Variable(Tensor(U, OCL(U)))
  ) forall U
    result = {% if U == Float32 %}
               Num::Float32AcosBackwardKernel.instance.call(gradient, a.value)
             {% elsif U == Float64 %}
               Num::Float64AcosBackwardKernel.instance.call(gradient, a.value)
             {% else %}
      {% raise "Invalid Dtype" %}
    {% end %}
    [result]
  end

  # :nodoc:
  def atan_backward(
    gradient : Tensor(U, OCL(U)),
    a : Variable(Tensor(U, OCL(U)))
  ) forall U
    result = {% if U == Float32 %}
               Num::Float32AtanBackwardKernel.instance.call(gradient, a.value)
             {% elsif U == Float64 %}
               Num::Float64AtanBackwardKernel.instance.call(gradient, a.value)
             {% else %}
      {% raise "Invalid Dtype" %}
    {% end %}
    [result]
  end
end
