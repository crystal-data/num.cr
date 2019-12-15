require "../spec_helper"

describe Num::BaseArray do
  describe "BaseArray#constructor" do
    it "Creates a valid empty array" do
      m = MockArray(Int32).new([] of Int32)
      m.shape.should eq [0]
      m.strides.should eq [1]
    end

    it "Creates empty array from valid shape" do
      m = MockArray(Int32).new([2])
      m.shape.should eq [2]
      m.strides.should eq [1]
      m.ndims.should eq 1
    end

    it "Creates multidimensional array from valid shape" do
      m = MockArray(Int32).new([2, 2, 2])
      m.shape.should eq [2, 2, 2]
      m.strides.should eq [4, 2, 1]
      m.ndims.should eq 3
    end

    it "Creates multidimensional array from array" do
      m = MockArray.from_array [[2, 2], [3, 4]]
      m.shape.should eq [2, 2]
      m.strides.should eq [2, 1]
      m.ndims.should eq 2
    end

    it "Creates multidimensional array from block" do
      m = MockArray.new([2, 2]) { |i| i }
      m.shape.should eq [2, 2]
      m.strides.should eq [2, 1]
      m.ndims.should eq 2
    end

    it "creates valid scalar array" do
      m = MockArray.new(2)
      m.value.should eq 2
    end

    it "correctly identifies c-style array flags" do
      m = MockArray(Int32).new([2, 2], ArrayFlags::Contiguous)
      m.flags.contiguous?.should be_true
      m.flags.fortran?.should be_false
    end

    it "correctly identifies fortran style array flags" do
      m = MockArray(Int32).new([2, 2], ArrayFlags::Fortran)
      m.flags.fortran?.should be_true
      m.flags.contiguous?.should be_false
    end

    it "picks up flags from block" do
      m = MockArray.new([2, 2], ArrayFlags::Contiguous) { |i| i }
      m.flags.contiguous?.should be_true
      m.flags.fortran?.should be_false
    end

    it "picks up fortran flags from block" do
      m = MockArray.new([2, 2], ArrayFlags::Fortran) { |i| i }
      m.flags.contiguous?.should be_false
      m.flags.fortran?.should be_true
    end

    it "correctly creates array from proc" do
      prok = ->(x : Int32) { x * 2 }
      m = MockArray.from_proc([2, 2], prok)
      expected = MockArray.from_array [[0, 2], [4, 6]]
      assert_array_equal m, expected
    end

    it "correctly creates matrix from block" do
      m = MockArray.new(2, 2) { |i, j| i + j }
      expected = MockArray.from_array [[0, 1], [1, 2]]
      assert_array_equal expected, m
    end
  end
end
