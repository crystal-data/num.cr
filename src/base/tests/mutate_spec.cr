require "../../__test__"

describe Num::BaseArray do
  describe "BaseArray#reshape" do
    it "reshapes a tensor to 2d" do
      a = Tensor.from_array [1, 2, 3, 4]
      expected = Tensor.from_array [[1, 2], [3, 4]]
      res = a.reshape(2, 2)
      assert_array_equal res, expected
    end

    it "infer reshape dimension" do
      a = Tensor.from_array [1, 2, 3, 4]
      expected = Tensor.from_array [[1, 2], [3, 4]]
      res = a.reshape(-1, 2)
      assert_array_equal res, expected
    end

    it "raise on bad reshape" do
      a = N.arange(10)
      expect_raises(ShapeError) do
        a.reshape(3, 3)
      end
    end

    it "raises on infer two dimensions" do
      a = N.arange(12)
      expect_raises(ValueError) do
        a.reshape(2, -1, -1)
      end
    end
  end

  describe "BaseArray#ravel" do
    it "flattens array" do
      desired = N.arange(12)
      result = desired.reshape(3, 2, 2).ravel
      assert_array_equal desired, result
    end
  end
end
