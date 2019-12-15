require "../spec_helper"

macro reduce_operation_axis(arr, output, axis, operation)
  it "reduces along axis {{axis}} using {{operation}}" do
    t = Tensor.from_array {{arr}}
    desired = Tensor.from_array {{output}}
    res = N.{{operation}}(t, axis: {{axis}})
    assert_array_equal res, desired
  end
end

describe Num::Statistics do
  describe "reduction to scalars" do
    it "sum reduces to scalar" do
      t = N.arange(100_000, dtype: Float64)
      t.sum.should eq 4999950000.0
    end

    it "strided sum reduces to scalar" do
      t = N.arange(100).reshape([10, 10])[..., 1]
      t.sum.should eq 460
    end

    it "prod reduces to scalar" do
      t = N.arange(1, 10, dtype: Float64)
      N.prod(t).should eq 362880.0
    end

    it "strided prod reduces to scalar" do
      t = (N.arange(10, 110)//10).reshape([10, 10])[..., 1]
      N.prod(t).should eq 3628800
    end

    it "all reduces to boolean" do
      t = Tensor.from_array [true, true, true, false]
      N.all(t).should be_false
    end

    it "any reduces to scalar" do
      t = Tensor.from_array [false, true, false, false]
      N.any(t).should be_true
    end

    it "mean reduces to scalar" do
      t = N.arange(1, 10)
      N.mean(t).should eq 5.0
    end

    it "std reduces to scalar" do
      t = N.arange(2)
      N.std(t).should eq 0.5
    end

    it "max reduces to scalar" do
      t = N.arange(100)
      N.max(t).should eq 99
    end

    it "min reduces to scalar" do
      t = N.arange(100)
      N.min(t).should eq 0
    end

    it "ptp reduces to scalar" do
      t = N.arange(100)
    end
  end

  describe "reductions along axes" do
    reduce_operation_axis [[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]], [[12, 15], [18, 21]], 0, sum
    reduce_operation_axis [[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]], [[2, 4], [10, 12], [18, 20]], 1, sum
    reduce_operation_axis [[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]], [[1, 5], [9, 13], [17, 21]], 2, sum
    reduce_operation_axis [[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]], [[0, 45], [120, 231]], 0, prod
    reduce_operation_axis [[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]], [[0, 3], [24, 35], [80, 99]], 1, prod
    reduce_operation_axis [[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]], [[0, 6], [20, 42], [72, 110]], 2, prod
    # reduce_operation_axis [[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]], [[false, true], [true, true]], 0, all  # This needs fixed
    reduce_operation_axis [[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]], [[false, true], [true, true], [true, true]], 1, all
    reduce_operation_axis [[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]], [[false, true], [true, true], [true, true]], 2, all
    reduce_operation_axis [[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]], [[true, true], [true, true]], 0, any
    reduce_operation_axis [[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]], [[true, true], [true, true], [true, true]], 1, any
    reduce_operation_axis [[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]], [[true, true], [true, true], [true, true]], 2, any
    reduce_operation_axis [[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]], [[4.0, 5.0], [6.0, 7.0]], 0, mean
    reduce_operation_axis [[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]], [[1.0, 2.0], [5.0, 6.0], [9.0, 10.0]], 1, mean
    reduce_operation_axis [[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]], [[0.5, 2.5], [4.5, 6.5], [8.5, 10.5]], 2, mean
    reduce_operation_axis [[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]], [[8, 9], [10, 11]], 0, max
    reduce_operation_axis [[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]], [[2, 3], [6, 7], [10, 11]], 1, max
    reduce_operation_axis [[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]], [[1, 3], [5, 7], [9, 11]], 2, max
    reduce_operation_axis [[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]], [[0, 1], [2, 3]], 0, min
    reduce_operation_axis [[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]], [[0, 1], [4, 5], [8, 9]], 1, min
    reduce_operation_axis [[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]], [[0, 2], [4, 6], [8, 10]], 2, min
    reduce_operation_axis [[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]], [[8, 8], [8, 8]], 0, ptp
    reduce_operation_axis [[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]], [[2, 2], [2, 2], [2, 2]], 1, ptp
    reduce_operation_axis [[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]], [[1, 1], [1, 1], [1, 1]], 2, ptp
  end
end
