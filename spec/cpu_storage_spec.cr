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

require "./spec_helper"

describe CPU do
  it "initializes with an initial capacity" do
    a = CPU(Int32).new([2, 2])
    expected = [0, 0, 0, 0]
    result = Num::Backend.tensor_to_crystal_array(a)
    result.should eq expected
  end

  it "initializes with an initial capacity and value" do
    a = CPU.new([3], 1.5)
    expected = [1.5, 1.5, 1.5]
    result = Num::Backend.tensor_to_crystal_array(a)
    expected.should eq result
  end

  it "creates storage fron an array" do
    a = [1, 2, 3, 4]
    s = Num::Backend.flat_array_to_storage(a, [4], CPU)
    result = Num::Backend.tensor_to_crystal_array(s)
    a.should eq result
  end

  it "creates storage from a hostptr" do
    a = [1, 2, 3, 4]
    s = Num::Backend.hostptr_to_storage(a.to_unsafe, [4], Num::RowMajor, CPU)
    result = Num::Backend.tensor_to_crystal_array(s)
    result.should eq a
  end
end
