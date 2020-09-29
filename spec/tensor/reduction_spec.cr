require "../spec_helper"

describe Tensor do
  it "sorts a one dimensional Tensor" do
    a = [4, 3, 2, 1].to_tensor
    result = Num.sort(a)
    expected = [1, 2, 3, 4].to_tensor
    assert_array_equal(result, expected)
  end

  it "sorts a strided Tensor" do
    a = [4, 3, 2, 1].to_tensor[{..., 2}]
    result = Num.sort(a)
    expected = [2, 4]
    assert_array_equal(result, expected)
  end

  it "sorts a Tensor along an axis" do
    a = [[3, 5, 6], [1, 1, 2], [9, 2, 3]].to_tensor
    result = Num.sort(a, 0)
    expected = [[1, 1, 2], [3, 2, 3], [9, 5, 6]].to_tensor
    assert_array_equal(result, expected)
  end

  it "sorts a strided Tensor along an axis" do
    a = [[3, 4, 5, 1], [2, 1, 3, 2], [4, 7, 6, 2]].to_tensor[..., {..., 2}]
    result = Num.sort(a, 0)
    expected = [[2, 3], [3, 5], [4, 6]].to_tensor
    assert_array_equal(result, expected)
  end
end
