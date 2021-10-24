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

  # :nodoc:
  MNIST_TEST_URL = "https://pjreddie.com/media/files/mnist_test.csv"

  # :nodoc:
  MNIST_TRAIN_URL = "https://pjreddie.com/media/files/mnist_train.csv"

  # :nodoc:
  struct MNIST
    getter features : Tensor(Float32, CPU(Float32))
    getter labels : Tensor(Float32, CPU(Float32))
    getter test_features : Tensor(Float32, CPU(Float32))
    getter test_labels : Tensor(Float32, CPU(Float32))

    def initialize(
      @features : Tensor(Float32, CPU(Float32)),
      @labels : Tensor(Float32, CPU(Float32)),
      @test_features : Tensor(Float32, CPU(Float32)),
      @test_labels : Tensor(Float32, CPU(Float32))
    )
    end
  end

  # :nodoc:
  def load_mnist_helper(url)
    csv = CSV.parse(load_dataset_http(url))
    features = csv[1...].map &.[1...]
    labels = csv[1...].map &.[0]
    l = labels.to_tensor.as_type(Int32)
    lf = Tensor(Int32, CPU(Int32)).zeros([l.shape[0], 10])
    l.each_with_index do |el, i|
      lf[i, el] = 1
    end
    {features.to_tensor.as_type(Float32), lf.as_type(Float32)}
  end

  # Returns a struct containing features, labels, as well as
  # test_features and test_labels for the MNIST dataset
  def load_mnist_dataset
    f, l = load_mnist_helper(MNIST_TEST_URL)
    tf, tl = load_mnist_helper(MNIST_TRAIN_URL)
    return MNIST.new(f, l, tf, tl)
  end
end
