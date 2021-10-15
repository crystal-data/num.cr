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

describe Number do
  it "should add against a Tensor" do
    expected = [1, 2, 3].to_tensor
    result = 1 + [0, 1, 2].to_tensor
    Num::Testing.tensor_equal(expected, result).should be_true
  end

  it "should subtract against a Tensor" do
    expected = [1, 0, -1].to_tensor
    result = 1 - [0, 1, 2].to_tensor
    Num::Testing.tensor_equal(expected, result).should be_true
  end

  it "should multiply against a Tensor" do
    expected = [1, 2, 3].to_tensor
    result = 1 * [1, 2, 3].to_tensor
    Num::Testing.tensor_equal(expected, result).should be_true
  end

  it "should divide against a Tensor" do
    expected = [12, 3, 1].to_tensor
    result = 12 / [1, 4, 12].to_tensor
    Num::Testing.tensor_equal(expected, result).should be_true
  end

  it "should exponentiate against a Tensor" do
    expected = [1, 2, 4].to_tensor
    result = 2 ** [0, 1, 2].to_tensor
    Num::Testing.tensor_equal(expected, result).should be_true
  end
end
