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

module Num::NN
  extend self

  def mse_backwards(
    gradient : Tensor(U, CPU(U)),
    cache : Tensor(U, CPU(U)),
    target : Tensor(U, CPU(U)),
  ) forall U
    norm = gradient.value * 2 / gradient.size
    result = cache.map(target) do |x, y|
      norm * (x - y)
    end
    [result]
  end

  def sigmoid_cross_entropy_backwards(
    gradient : Tensor(U, CPU(U)),
    cache : Tensor(U, CPU(U)),
    target : Tensor(U, CPU(U))
  ) forall U
    grad = gradient.value
    batch_size = cache.shape[0]
    output = cache.map(target) do |x, y|
      grad * ((1 / (1 + Math.exp(-x))) - y) / batch_size
    end
    [output]
  end
end
