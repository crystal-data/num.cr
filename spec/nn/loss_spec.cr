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
  it "calculates softmax cross entropy loss" do
    predicted = [-3.44, 1.16, -0.81, 3.91].to_tensor.reshape(1, 4)
    truth = [0, 0, 0, 1].to_tensor.as_type(Float64).reshape(1, 4)

    sce_loss = Num::NN.softmax_cross_entropy(predicted, truth)

    sce_loss.value.should be_close(0.0709, 1e-4)

    prok = ->(pred : Float64Tensor) { Num::NN.softmax_cross_entropy(pred, truth).value }
    expected_grad = sce_loss * Num::NN.numerical_gradient(predicted, prok)

    grad = Num::NN.softmax_cross_entropy_backward(sce_loss, predicted, truth)

    Num::NN.mean_relative_error(grad[0], expected_grad).should be < 1e-6
  end
end
