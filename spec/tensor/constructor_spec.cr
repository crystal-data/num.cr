require "../spec_helper"
require "../../src/util/*"
include Bottle
include Bottle::Exceptions

describe Tensor do
  describe "Tensor#initialize" do
    it "correctly identifies dtype from data" do
      f = Tensor.new [1, 2, 3]
      f.should be_a(Tensor(Int32))
    end

    it "correctly creates flask from block" do
      f = Tensor.new(5) { |i| i }
      Testing.tensor_equal(f, Tensor.new [0, 1, 2, 3, 4]).should be_true
    end

    it "correctly allocates an empty flask" do
      n = 10
      f = Tensor.empty(n)
      f.size.should eq(n)
    end

    it "empty respects passed dtype" do
      f = Tensor.empty(10, Float32)
      f.should be_a(Tensor(Float32))
    end

    it "creates a valid flask from a slice, size and stride" do
      n = 5
      slice = Pointer.malloc(n) { |i| i }
      f = Tensor.new slice, n, 1, true
      Testing.tensor_equal(f, Tensor.new [0, 1, 2, 3, 4]).should be_true
    end

    it "creates a valid strided flask" do
      n = 10
      slice = Pointer.malloc(n) { |i| i }
      f = Tensor.new slice, 5, 2, true
      Testing.tensor_equal(f, Tensor.new [0, 2, 4, 6, 8]).should be_true
    end

    it "random returns correct type from range" do
      f = Tensor.random(0...10, 10)
      f.should be_a(Tensor(Int32))
    end

    # it "reverses a flask" do
    #   f = Tensor.new [1, 2, 3]
    #   Testing.tensor_equal(f.reverse, Tensor.new [3, 2, 1]).should be_true
    # end

    it "casts the type of a flask" do
      f = Tensor.new [1, 2, 3]
      fasfloat = f.astype(Float64)
      Testing.tensor_equal(fasfloat, Tensor.new [1.0, 2.0, 3.0]).should be_true
    end

    it "clones a flask that owns its own memory" do
      f = Tensor.new [1, 2, 3]
      g = f.clone
      g[0] = 100
      Testing.tensor_equal(f, g).should be_false
    end
  end
end
