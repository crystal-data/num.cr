require "../spec_helper"
require "../../src/util/*"
include Bottle
include Bottle::Core::Exceptions

describe Vector do
  describe "Vector#initialize" do
    it "correctly identifies dtype from data" do
      f = Vector.new [1, 2, 3]
      f.should be_a(Vector(Int32))
    end

    it "correctly creates flask from block" do
      f = Vector.new(5) { |i| i }
      Testing.vector_equal(f, Vector.new [0, 1, 2, 3, 4]).should be_true
    end

    it "correctly allocates an empty flask" do
      n = 10
      f = Vector.empty(n)
      f.size.should eq(n)
    end

    it "empty respects passed dtype" do
      f = Vector.empty(10, Float32)
      f.should be_a(Vector(Float32))
    end

    it "creates a valid flask from a slice, size and stride" do
      n = 5
      slice = Slice.new(n) { |i| i }
      f = Vector.new slice, 1, true
      Testing.vector_equal(f, Vector.new [0, 1, 2, 3, 4]).should be_true
    end

    it "creates a valid strided flask" do
      n = 10
      slice = Slice.new(n) { |i| i }
      f = Vector.new slice, 2, true
      Testing.vector_equal(f, Vector.new [0, 2, 4, 6, 8]).should be_true
    end

    it "random returns correct type from range" do
      f = Vector.random(0...10, 10)
      f.should be_a(Vector(Int32))
    end

    it "reverses a flask" do
      f = Vector.new [1, 2, 3]
      Testing.vector_equal(f.reverse, Vector.new [3, 2, 1]).should be_true
    end

    it "casts the type of a flask" do
      f = Vector.new [1, 2, 3]
      fasfloat = f.astype(Float64)
      Testing.vector_equal(fasfloat, Vector.new [1.0, 2.0, 3.0]).should be_true
    end

    it "clones a flask that owns its own memory" do
      f = Vector.new [1, 2, 3]
      g = f.clone
      g[0] = 100
      Testing.vector_equal(f, g).should be_false
    end
  end
end
