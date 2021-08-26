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

require "../spec_helper"

describe Tensor do
  it "initializes a null Tensor" do
    t = Tensor(Int32).new

    t.shape.should eq [] of Int32
    t.strides.should eq [1]
    t.offset.should eq 0
    t.size.should eq 0
    t.raw.should eq Pointer(Int32).null
  end

  it "initializes a contiguous empty Tensor" do
    t = Tensor(Int32).new([2, 2])

    t.shape.should eq [2, 2]
    t.strides.should eq [2, 1]
    t.offset.should eq 0
    t.size.should eq 4
    t.to_a.should eq [0, 0, 0, 0]
  end

  it "initializes a fortran contiguous empty Tensor" do
    t = Tensor(Int32).new([2, 2], Num::ColMajor)

    t.shape.should eq [2, 2]
    t.strides.should eq [1, 2]
    t.offset.should eq 0
    t.size.should eq 4
    t.to_a.should eq [0, 0, 0, 0]
  end

  it "initializes a tensor with an initial value" do
    t = Tensor.new([2, 2], 4.5)
    t.to_a.should eq [4.5, 4.5, 4.5, 4.5]
  end

  it "updates flags correctly in a non-contiguous tensor" do
    raw = Pointer(Int32).malloc(4, 1)
    t = Tensor.new(raw, [2], [2], 0)
    t.flags.contiguous?.should be_false
    t.flags.fortran?.should be_false
  end

  it "leaves non-contiguity related flags alone when initializing" do
    raw = Pointer(Int32).malloc(4, 1)
    flags = Num::ArrayFlags::All
    flags &= ~Num::ArrayFlags::Write
    t = Tensor.new(raw, [4], [1], 0, flags)
    t.flags.write?.should be_false
  end

  it "populates tensor data from a block" do
    t = Tensor.new([2, 2]) { |i| i }
    t.to_a.should eq [0, 1, 2, 3]
  end

  it "populates tensor data from a block with fortran order" do
    t = Tensor.new([2, 2], Num::ColMajor) { |i| i }
    t.to_a.should eq [0, 2, 1, 3]
  end

  it "disallows empty matrix creation" do
    expect_raises(Num::Exceptions::ValueError) do
      Tensor.new(0, 1) { |i, j| 1 }
    end
  end

  it "creates a matrix from a block" do
    t = Tensor.new(2, 2) do |i, j|
      i + j
    end
    t.to_a.should eq [0, 1, 1, 2]
  end
end
