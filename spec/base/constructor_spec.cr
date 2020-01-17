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
require "../spec_helper"

describe Num::BaseArray do
  describe "BaseArray#constructor" do
    it "Creates a valid empty array" do
      m = Num::BaseArray(Int32).new([] of Int32)
      m.shape.should eq [0]
      m.strides.should eq [1]
    end

    it "Creates empty array from valid shape" do
      m = Num::BaseArray(Int32).new([2])
      m.shape.should eq [2]
      m.strides.should eq [1]
      m.ndims.should eq 1
    end

    it "Creates multidimensional array from valid shape" do
      m = Num::BaseArray(Int32).new([2, 2, 2])
      m.shape.should eq [2, 2, 2]
      m.strides.should eq [4, 2, 1]
      m.ndims.should eq 3
    end

    it "Creates multidimensional array from array" do
      m = Num::BaseArray.from_array [[2, 2], [3, 4]]
      m.shape.should eq [2, 2]
      m.strides.should eq [2, 1]
      m.ndims.should eq 2
    end

    it "Creates multidimensional array from block" do
      m = Num::BaseArray.new([2, 2]) { |i| i }
      m.shape.should eq [2, 2]
      m.strides.should eq [2, 1]
      m.ndims.should eq 2
    end

    it "creates valid scalar array" do
      m = Num::BaseArray.new(2)
      m.value.should eq 2
    end

    it "correctly identifies c-style array flags" do
      m = Num::BaseArray(Int32).new([2, 2], 'C')
      m.flags.contiguous?.should be_true
      m.flags.fortran?.should be_false
    end

    it "correctly identifies fortran style array flags" do
      m = Num::BaseArray(Int32).new([2, 2], 'F')
      m.flags.fortran?.should be_true
      m.flags.contiguous?.should be_false
    end

    it "picks up flags from block" do
      m = Num::BaseArray.new([2, 2], 'C') { |i| i }
      m.flags.contiguous?.should be_true
      m.flags.fortran?.should be_false
    end

    it "picks up fortran flags from block" do
      m = Num::BaseArray.new([2, 2], 'F') { |i| i }
      m.flags.contiguous?.should be_false
      m.flags.fortran?.should be_true
    end

    it "correctly creates array from proc" do
      prok = ->(x : Int32) { x * 2 }
      m = Num::BaseArray.from_proc([2, 2], prok)
      expected = Num::BaseArray.from_array [[0, 2], [4, 6]]
      assert_array_equal m, expected
    end

    it "correctly creates matrix from block" do
      m = Num::BaseArray.new(2, 2) { |i, j| i + j }
      expected = Num::BaseArray.from_array [[0, 1], [1, 2]]
      assert_array_equal expected, m
    end
  end
end
