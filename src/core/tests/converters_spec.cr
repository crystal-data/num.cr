require "../../__test__"

struct Foo(T) < BaseArray(T)
  def check_type
    T
  end
  def basetype
    Foo
  end
end

describe Bottle::Convert do
  describe "Convert#astensor" do
    it "leaves tensors alone" do
      t = Tensor(Int32).new([2, 2])
      res = B.astensor(t)
      res.is_a?(Tensor(Int32)).should be_true
      assert_array_equal res, t
    end

    it "coerces other base types to tensors" do
      t = Foo.new([2, 2]) { |i| i }
      res = B.astensor(t)
      res.is_a?(Tensor(Int32)).should be_true
    end

    it "upscales an array to a tensor" do
      t = [[1, 2, 3], [4, 5, 6]]
      res = B.astensor(t)
      res.is_a?(Tensor(Int32)).should be_true
    end

    it "upscales a number to a tensor" do
      t = 3
      res = B.astensor(t)
      res.is_a?(Tensor(Int32)).should be_true
    end
  end
end
