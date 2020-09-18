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

require "http"

module Num::NN
  extend self

  IRIS_URL = "https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/639388c2cbc2120a14dcf466e85730eb8be498bb/iris.csv"

  def load_iris_dataset
    response = HTTP::Client.get(IRIS_URL)
    csv = CSV.parse response.body

    features = csv[1...].map &.[...-1]
    labels = csv[1...].map &.[-1]

    rng = (0...labels.size).to_a
    rng.shuffle!

    features = features.map_with_index do |_, i|
      features[rng[i]]
    end

    labels = labels.map_with_index do |_, i|
      labels[rng[i]]
    end

    x_train = features.to_tensor.as_type(Float64)

    label_mapping = {
      "setosa"     => [0, 0, 1],
      "versicolor" => [0, 1, 0],
      "virginica"  => [1, 0, 0],
    }

    mapped = labels.map { |el| label_mapping[el] }
    y_train = mapped.to_tensor.as_type(Float64)

    {labels, x_train.transpose, y_train.transpose}
  end
end
