require "./spec_helper"

describe Num::Internal do
  it "infers RowMajor strides" do
    a = [3, 2, 2]
    expected = [4, 2, 1]
    result = Num::Internal.shape_to_strides(a, Num::RowMajor)
    expected.should eq result
  end

  it "infers ColMajor strides" do
    a = [3, 2, 2]
    expected = [1, 3, 6]
    result = Num::Internal.shape_to_strides(a, Num::ColMajor)
    expected.should eq result
  end

  it "finds the shape of a deeply nested array" do
    a = [[[[1]]]]
    expected = [1, 1, 1, 1]
    result = Num::Internal.recursive_array_shape(a)
    expected.should eq result

    a = [[1, 2], [3, 4]]
    expected = [2, 2]
    result = Num::Internal.recursive_array_shape(a)
    expected.should eq result
  end
end
