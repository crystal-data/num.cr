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
require "../../spec_helper"

describe Num::Internal do
  it "Creates RowMajor strides from shape" do
    shape = [2, 2, 2]
    Num::Internal.shape_to_strides(shape).should eq [4, 2, 1]
  end

  it "Creates ColMajor strides from shape" do
    shape = [2, 2, 2]
    Num::Internal.shape_to_strides(shape, Num::ColMajor).should eq [1, 2, 4]
  end

  it "Infers shape from nested stdlib array" do
    ary = [[1, 2, 3], [4, 5, 6]]
    Num::Internal.stdlib_array_to_nd_shape(ary).should eq [2, 3]
  end

  it "Infers shape from complex stdlib array" do
    ary = [[[[1, 2], [3, 4]]]]
    Num::Internal.stdlib_array_to_nd_shape(ary).should eq [1, 1, 2, 2]
  end

  it "Raises ValueError if subarrays are different lengths" do
    ary = [[1, 2, 3], [1, 2]]

    expect_raises(Num::Exceptions::ValueError) do
      Num::Internal.stdlib_array_to_nd_shape(ary)
    end
  end

  it "Detects axis out of range" do
    t = Tensor(Int32).new([3, 3])
    expect_raises(Num::Exceptions::AxisError) do
      Num::Internal.check_axis_index(t, 3, 0, 3)
    end
  end
end
