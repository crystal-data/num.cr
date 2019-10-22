require "../spec_helper"
require "../../src/util/*"
include Bottle
include Bottle::Core::Exceptions

describe Flask do
  describe "Flask#index" do
    it "correctly pours a single element" do
      f = Flask.new [1, 2, 3, 4, 5]
      f[2].should eq(3)
    end

    it "correctly pours multiple elements as a copy" do
      f = Flask.new [1, 2, 3, 4, 5]
      result = f[[0, 1, 2]]
      expected = Flask.new [1, 2, 3]
      Testing.flask_equal(result, expected).should be_true
    end

    it "correctly pours a view of a flask" do
      f = Flask.new [1, 2, 3, 4, 5]
      result = f[2...]
      expected = Flask.new [3, 4, 5]
      Testing.flask_equal(result, expected).should be_true
    end

    it "correctly pours from a strided flask" do
      n = 10
      stride = 2
      slice = Slice.new(n) { |i| i }
      f = Flask.new slice, n // stride, stride
      f[3].should eq(6)
    end

    it "correctly pours a range from a strided flask" do
      n = 10
      stride = 2
      slice = Slice.new(n) { |i| i }
      f = Flask.new slice, n // stride, stride
      result = f[1...]
      expected = Flask.new [2, 4, 6, 8]
      Testing.flask_equal(result, expected).should be_true
    end

    it "correctly sets a single value" do
      f = Flask.new [1, 2, 3, 4, 5]
      f[2] = 100
      f[2].should eq(100)
    end

    it "correctly sets multiple values" do
      f = Flask.new [1, 2, 3, 4, 5]
      f[[0, 1]] = [100, 100]
      expected = Flask.new [100, 100]
      result = f[[0, 1]]
      Testing.flask_equal(expected, result).should be_true
    end

    it "correctly sets multiple values for range" do
      f = Flask.new [1, 2, 3, 4, 5]
      f[...2] = [100, 100]
      expected = Flask.new [100, 100]
      result = f[...2]
      Testing.flask_equal(expected, result).should be_true
    end

    it "correctly sets multiple values for range with strided flask" do
      n = 10
      stride = 2
      slice = Slice.new(n) { |i| i }
      f = Flask.new slice, n // stride, stride
      f[...2] = [100, 100]
      result = f[...2]
      expected = Flask.new [100, 100]
      Testing.flask_equal(result, expected).should be_true
    end
  end
end
