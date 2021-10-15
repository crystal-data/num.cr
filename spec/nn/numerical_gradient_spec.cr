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

describe Num::NN do
  it "calculates the correct derivative of a float" do
    f = ->(x : Float64) { x * x + x + 1.0 }
    Num::NN.numerical_gradient(2.0, f).should be_close(5, 1e-8)
  end

  it "calculates the correct derivative of a Tensor" do
    f = ->(t : Tensor(Float64, CPU(Float64))) do
      x = t[0].value
      y = t[1].value
      x * x + y * y + x * y + x + y + 1.0
    end

    input = [2.0, 3.0].to_tensor
    grad = [8.0, 9.0].to_tensor

    Num::NN.mean_relative_error(Num::NN.numerical_gradient(input, f), grad).should be < 1e-8
  end
end
