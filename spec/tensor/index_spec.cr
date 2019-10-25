require "../spec_helper"
require "../../src/util/*"
include Bottle
include Bottle::Core::Exceptions

describe Tensor do
  describe "Tensor#index" do
    it "correctly gets a single element" do
      f = Tensor.new [1, 2, 3, 4, 5]
      f[2].should eq(3)
    end

    it "correctly gets multiple elements as a copy" do
      f = Tensor.new [1, 2, 3, 4, 5]
      result = f[[0, 1, 2]]
      expected = Tensor.new [1, 2, 3]
      Testing.tensor_equal(result, expected).should be_true
    end

    it "correctly gets a view of a Tensor" do
      f = Tensor.new [1, 2, 3, 4, 5]
      result = f[2...]
      expected = Tensor.new [3, 4, 5]
      Testing.tensor_equal(result, expected).should be_true
    end

    it "correctly gets from a strided Tensor" do
      n = 10
      stride = 2
      slice = Pointer.malloc(n) { |i| i }
      f = Tensor.new slice, n // stride, stride, true
      f[3].should eq(6)
    end

    it "correctly gets a range from a strided Tensor" do
      n = 10
      stride = 2
      slice = Pointer.malloc(n) { |i| i }
      f = Tensor.new slice, n // stride, stride, true
      result = f[1...]
      expected = Tensor.new [2, 4, 6, 8]
      Testing.tensor_equal(result, expected).should be_true
    end

    it "correctly sets a single value" do
      f = Tensor.new [1, 2, 3, 4, 5]
      f[2] = 100
      f[2].should eq(100)
    end

    it "correctly sets multiple values" do
      f = Tensor.new [1, 2, 3, 4, 5]
      f[[0, 1]] = [100, 100]
      expected = Tensor.new [100, 100]
      result = f[[0, 1]]
      Testing.tensor_equal(expected, result).should be_true
    end

    it "correctly sets multiple values for range" do
      f = Tensor.new [1, 2, 3, 4, 5]
      f[...2] = [100, 100]
      expected = Tensor.new [100, 100]
      result = f[...2]
      Testing.tensor_equal(expected, result).should be_true
    end

    it "correctly sets multiple values for range with strided flask" do
      n = 10
      stride = 2
      slice = Pointer.malloc(n) { |i| i }
      f = Tensor.new slice, n//stride, stride, true
      f[...2] = [100, 100]
      result = f[...2]
      expected = Tensor.new [100, 100]
      Testing.tensor_equal(result, expected).should be_true
    end
  end
end
