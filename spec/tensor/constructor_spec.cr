require "../spec_helper"
require "../../src/util/*"
include Bottle
include Bottle::Internal::Exceptions
include Bottle::Internal

describe Tensor do
  describe "Tensor#from_array" do
    it "Identifies an improperly shaped Array" do
      expect_raises(ShapeError) do
        Tensor.from_array([2, 2, 3], [1, 2])
      end
    end

    it "Initializes a Tensor from 1D Array" do
      result = Tensor.from_array([4], [0, 1, 2, 3])
      expected = Tensor.new([4]) { |i| i }
      Comparison.allclose(result, expected).should be_true
    end

    it "Initializes a Tensor from a 2D Array" do
      result = Tensor.from_array([2, 2], [[0, 1], [2, 3]])
      expected = Tensor.new([2, 2]) { |i| i }
      Comparison.allclose(result, expected).should be_true
    end

    it "Initializes different shape Tensor from array" do
      result = Tensor.from_array([2, 3], [0, 1, 2, 3, 4, 5])
      expected = Tensor.new([2, 3]) { |i| i }
      Comparison.allclose(result, expected).should be_true
    end

    it "Initializes a Tensor with proper flags" do
      result = Tensor.from_array([4], [1, 2, 3, 4])
      result.flags.fortran?.should be_true
      result.flags.contiguous?.should be_true
    end

    it "Initializes a 2D Tensor with proper flags" do
      result = Tensor.from_array([2, 2], [1, 2, 3, 4])
      result.flags.fortran?.should be_false
      result.flags.contiguous?.should be_true
    end

    it "Correctly handles an empty array" do
      result = Tensor.from_array([] of Int32, [] of Int32)
      result.shape.should eq [0]
      result.strides.should eq [1]
      result.should be_a(Tensor(Int32))
    end

    it "Initializes a Tensor from jagged nested array" do
      result = Tensor.from_array([2], [[[[0, [[[[[1]]]]]]]]])
      expected = Tensor.new([2]) { |i| i }
      Comparison.allclose(result, expected).should be_true
    end
  end

  describe "Tensor#from_block" do
    it "Creates empty Tensor from empty block" do
      result = Tensor.new([0]) { |i| i }
      result.shape.should eq [0]
      result.strides.should eq [1]
      result.should be_a(Tensor(Int32))
    end

    it "Infers float dtype from block" do
      result = Tensor.new([5]) { |i| i * 1.0 }
      result.should be_a(Tensor(Float64))
    end

    it "Correctly lays out memory from block" do
      result = Tensor.new([5]) { |i| i }
      result.flags.fortran?.should be_true
      result.flags.contiguous?.should be_true
    end

    it "Correctly lays out ND memory from block" do
      result = Tensor.new([2, 2, 3]) { |i| i }
      result.flags.fortran?.should be_false
      result.flags.contiguous?.should be_true
    end
  end

  describe "Tensor#matrix_block" do
    it "Infers dtype from block" do
      result = Tensor.new(2, 2) { |i, j| i / j }
      result.should be_a(Tensor(Float64))
    end

    it "Correctly lays out memory from block" do
      result = Tensor.new(2, 2) { |i, j| i / j }
      result.flags.fortran?.should be_false
      result.flags.contiguous?.should be_true
    end
  end

  describe "Tensor#random" do
    it "Creates empty tensor from empty shape" do
      result = Tensor.random(0...10, [] of Int32)
      result.shape.should eq [0]
      result.strides.should eq [1]
      result.should be_a(Tensor(Int32))
    end

    it "Infers dtype from range" do
      result = Tensor.random(0.0...10.0, [2, 2, 3])
      result.should be_a(Tensor(Float64))
    end
  end

  describe "Tensor#initialize" do
    it "Correctly creates empty tensor" do
      result = Tensor(Int32).new([] of Int32)
      result.shape.should eq [0]
      result.strides.should eq [1]
      result.should be_a(Tensor(Int32))
    end

    it "Lays out memory in Fortran order" do
      result = Tensor(Int32).new([2, 2], ArrayFlags::Fortran)
      result.flags.fortran?.should be_true
      result.flags.contiguous?.should be_false
    end

    it "Lays out memory in C style order" do
      result = Tensor(Int32).new([2, 2], ArrayFlags::Contiguous)
      result.flags.fortran?.should be_false
      result.flags.contiguous?.should be_true
    end
  end
end
