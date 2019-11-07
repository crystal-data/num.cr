require "../spec_helper"
require "../../src/util/*"
include Bottle
include Bottle::Internal::Exceptions
include Bottle::Internal

describe Tensor do
  describe "Tensor#getters" do
    it "Selects a single value from a Tensor" do
      t = Tensor.new([2, 2, 3]) { |i| i }
      result = t[[0, 1, 0]]
      expected = 3
      result.should eq expected
    end

    it "Selects a single value with a negative index" do
      t = Tensor.new([10]) { |i| i }
      result = t[[-1]]
      expected = 9
      result.should eq expected
    end

    it "Raises an IndexError when an index is out of range" do
      t = Tensor.new([2, 2, 3]) { |i| i }
      expect_raises(IndexError) do
        t[[0, 2, 0]]
      end
    end

    it "Selects a single entry from a dimension of a Tensor" do
      t = Tensor.new([2, 2, 3]) { |i| i }
      result = t[0]
      expected = Tensor.new([2, 3]) { |i| i }
      Comparison.allclose(result, expected).should be_true
    end

    it "Selects a single dimension with negative index" do
      t = Tensor.new([2, 2, 3]) { |i| i }
      result = t[-1]
      expected = Tensor.new([2, 3]) { |i| i + 6 }
      Comparison.allclose(result, expected).should be_true
    end

    it "Slices a tensor along a dimension" do
      t = Tensor.new([2, 2, 3]) { |i| i }
      result = t[1...]
      expected = Tensor.new([1, 2, 3]) { |i| i + 6 }
      Comparison.allclose(result, expected).should be_true
    end

    it "Slices a tensor using a negative index in a slice" do
      t = Tensor.new([2, 2, 3]) { |i| i }
      result = t[...-1]
      expected = Tensor.new([1, 2, 3]) { |i| i }
      Comparison.allclose(result, expected).should be_true
    end

    it "Slicing a Tensor raises an index error with bad dimension" do
      t = Tensor.new([2, 2, 3]) { |i| i }
      expect_raises(IndexError) do
        t[3, ...]
      end
    end

    it "Slicing a tensor with a bad slice raises an index error" do
      t = Tensor.new([2, 2, 3]) { |i| i }
      expect_raises(IndexError) do
        t[3...]
      end
    end

    it "A slice of a tensor does not won its own data" do
      t = Tensor.new([2, 2, 3]) { |i| i }
      t[1...].flags.own_data?.should be_false
    end
  end

  describe "Tensor#setters" do
    it "Sets a single element of a 1D tensor" do
      t = Tensor.new([5]) { |i| i }
      t[[0]] = 100
      expected = Tensor.from_array([5], [100, 1, 2, 3, 4])
      Comparison.allclose(t, expected).should be_true
    end

    it "Sets a slice of a 1D tensor" do
      t = Tensor.new([5]) { |i| i }
      t[1...] = Tensor.from_array([4], [99, 99, 99, 99])
      expected = Tensor.from_array([5], [0, 99, 99, 99, 99])
      Comparison.allclose(t, expected).should be_true
    end

    it "Sets all values of a view to a scalar" do
      t = Tensor.new([2, 2]) { |i| i }
      t[0] = 100
      expected = Tensor.from_array([2, 2], [100, 100, 2, 3])
      Comparison.allclose(t, expected).should be_true
    end

    it "Sets a more complex slice of ND Tensor" do
      t = Tensor.new([2, 2, 3]) { |i| i }
      t[0] = t[1]
      expected = Tensor.from_array([2, 2, 3], [6, 7, 8, 9, 10, 11] * 2)
      Comparison.allclose(t, expected).should be_true
    end

    it "Mutates base when a view of a Tensor is set" do
      t = Tensor.new([2, 2, 3]) { |i| i }
      slice = t[...]
      slice[0] = 99
      Comparison.allclose(t, slice).should be_true
    end
  end
end
