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

module Num::NN
  extend self

  # Brief description of streamingmaxsumexp
  #
  # Arguments
  # ---------
  # t : Tensor(U)
  #   Brief description of t : Tensor(U)
  #
  # Returns
  # -------
  # nil
  #
  # Examples
  # --------
  def streaming_max_sumexp(t : Tensor(U)) forall U
    mx = U::MIN
    sumexp = U.new(0)

    t.each do |el|
      if el <= mx
        sumexp += Math.exp(el - mx)
      else
        sumexp = sumexp * Math.exp(mx - el) + 1
        mx = el
      end
    end
    {mx, sumexp}
  end

  # Brief description of logsumexp
  #
  # Arguments
  # ---------
  # t : Tensor(U)
  #   Brief description of t : Tensor(U)
  #
  # Returns
  # -------
  # nil
  #
  # Examples
  # --------
  def logsumexp(t : Tensor(U)) forall U
    mx, sumexp = streaming_max_sumexp(t)
    mx + Math.log(sumexp, Math::E)
  end

  # Brief description of stablesoftmax
  #
  # Arguments
  # ---------
  # x : U
  #   Brief description of x : U
  # mx : U
  #   Brief description of mx : U
  # sumexp : U
  #   Brief description of sumexp : U
  #
  # Returns
  # -------
  # nil
  #
  # Examples
  # --------
  def stable_softmax(x : U, mx : U, sumexp : U) forall U
    Math.exp(x - mx) / sumexp
  end

  # Brief description of softmaxcrossentropy
  #
  # Arguments
  # ---------
  # input : Tensor(U)
  #   Brief description of input : Tensor(U)
  # target : Tensor(U)
  #   Brief description of target : Tensor(U)
  #
  # Returns
  # -------
  # U
  #
  # Examples
  # --------
  def softmax_cross_entropy(input : Tensor(U), target : Tensor(U)) : U forall U
    n = input.shape[0]
    result = (input * target).sum

    sum_logsumexp = U.new(0)

    input.each_axis(0) do |a|
      sum_logsumexp += logsumexp(a)
    end

    U.new((sum_logsumexp - result) / n)
  end

  # Brief description of softmaxcrossentropybackward
  #
  # Arguments
  # ---------
  # gradient : U
  #   Brief description of gradient : U
  # cached : Tensor(U)
  #   Brief description of cached : Tensor(U)
  # target : Tensor(U)
  #   Brief description of target : Tensor(U)
  #
  # Returns
  # -------
  # nil
  #
  # Examples
  # --------
  def softmax_cross_entropy_backward(gradient : U, cached : Tensor(U), target : Tensor(U)) forall U
    n = cached.shape[0]

    result = Tensor(U).zeros_like(cached)

    n.times do |i|
      mx, sumexp = Num::NN.streaming_max_sumexp(cached[i])
      res_slice = result[i]

      res_slice.map!(cached[i], target[i]) do |_, y, z|
        gradient * (Num::NN.stable_softmax(y, mx, sumexp) - z) / n
      end
    end
    result
  end
end
