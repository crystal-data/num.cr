require "../spec_helper"
require "../../src/util/testing"

describe Flask do
  describe "#initialize" do
    it "correctly identifies integer flask" do
      f = Flask.new [1, 2, 3]
      f.should be_a(Flask(Int32))
    end

    it "correctly identifies float32 flask" do
      f = Flask.new [1_f32, 2_f32, 3_f32]
      f.should be_a(Flask(Float32))
    end

    it "correctly identifies float64 flask" do
      f = Flask.new [1.0, 2.0, 3.0]
      f.should be_a(Flask(Float64))
    end

    it "correctly identifies boolean flask" do
      f = Flask.new [true, true, false]
      f.should be_a(Flask(Bool))
    end

    it "correctly creates flask from block" do
      f = Flask(Int32).new(5) { |i| i }
      BottleTest.flask_equal(f, Flask.new [0, 1, 2, 3, 4]).should be_true
    end

    it "correctly allocates an empty flask" do
      n = 10
      f = Flask(Int32).empty(n)
      f.size.should eq(n)
    end

    it "empty respects passed dtype" do
      f = Flask(Float32).empty(10)
      f.should be_a(Flask(Float32))
    end

    it "creates a valid flask from a slice, size and stride" do
      n = 5
      slice = Slice(Int32).new(n) { |i| i }
      f = Flask.new slice, n, 1
      BottleTest.flask_equal(f, Flask.new [0, 1, 2, 3, 4]).should be_true
    end

    it "creates a valid strided flask" do
      n = 10
      slice = Slice(Int32).new(n) { |i| i }
      f = Flask.new slice, n // 2, 2
      BottleTest.flask_equal(f, Flask.new [0, 2, 4, 6, 8]).should be_true
    end

    it "random returns correct type from range" do
      f = Flask.random(0...10, 10)
      f.should be_a(Flask(Int32))
    end

    it "reverses a flask" do
      f = Flask.new [1, 2, 3]
      BottleTest.flask_equal(f.reverse, Flask.new [3, 2, 1]).should be_true
    end

    it "casts the type of a flask" do
      f = Flask.new [1, 2, 3]
      fasfloat = f.astype(Float64)
      BottleTest.flask_equal(fasfloat, Flask.new [1.0, 2.0, 3.0]).should be_true
    end

    it "clones a flask that owns its own memory" do
      f = Flask.new [1, 2, 3]
      g = f.clone
      g[0] = 100
      BottleTest.flask_equal(f, g).should be_false
    end
  end
end
