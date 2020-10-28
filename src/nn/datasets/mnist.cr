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

require "csv"

module Num::NN
  extend self

  MNIST_URL = "https://pjreddie.com/media/files/mnist_test.csv"

  def load_mnist_dataset
    csv = CSV.parse(load_dataset_http(MNIST_URL))

    features = csv[1...].map &.[1...]
    labels = csv[1...].map &.[0]

    l = labels.to_tensor.as_type(Int32)
    lf = Tensor(Int32).zeros([l.shape[0], 10])

    l.each_with_index do |el, i|
      lf[i, el] = 1
    end

    {features.to_tensor.as_type(Float32), lf.as_type(Float32)}
  end
end
