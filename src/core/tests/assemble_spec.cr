require "../../__test__"

describe Num::Assemble do
  describe "Assemble#atleast_1d" do
    it "creates tensor from scalar" do
      a = 2
      t = N.atleast_1d(a)
      assert_array_equal t, Tensor.from_array [2]
    end

    it "leaves 1d tensor as is" do
      a = Tensor.from_array [1, 2]
      t = N.atleast_1d(a)
      assert_array_equal a, t
    end

    it "leaves 2d tensor as is" do
      a = Tensor.from_array [[1, 2], [1, 2]]
      t = N.atleast_1d(a)
      assert_array_equal a, t
    end

    it "leaves 3d tensor as is" do
      a = Tensor.new([3, 2, 2]) { |i| i }
      t = N.atleast_1d(a)
      assert_array_equal a, t
    end
  end

  describe "Assemble#atleast_2d" do
    it "creates tensor from scalar" do
      a = 2
      t = N.atleast_2d(a)
      assert_array_equal t, Tensor.from_array [[2]]
    end

    it "adds dimension to 1d tensor" do
      a = Tensor.from_array [1, 2]
      t = N.atleast_2d(a)
      assert_array_equal t, Tensor.from_array [[1, 2]]
    end

    it "leaves 2d tensor as is" do
      a = Tensor.from_array [[1, 2], [1, 2]]
      t = N.atleast_2d(a)
      assert_array_equal t, a
    end

    it "leaves 3d tensor as is" do
      a = Tensor.new([3, 2, 2]) { |i| i }
      t = N.atleast_2d(a)
      assert_array_equal t, a
    end
  end

  describe "Assemble#atleast_3d" do
    it "creates tensor from scalar" do
      a = 2
      t = N.atleast_3d(a)
      assert_array_equal t, Tensor.from_array [[[2]]]
    end

    it "adds dimension to 1d tensor" do
      a = Tensor.from_array [1, 2]
      t = N.atleast_3d(a)
      assert_array_equal t, Tensor.from_array [[[1, 2]]]
    end

    it "adds dimension to 2d" do
      a = Tensor.from_array [[1, 2], [1, 2]]
      t = N.atleast_3d(a)
      assert_array_equal t, Tensor.from_array [[[1, 2], [1, 2]]]
    end

    it "leaves 3d tensor as is" do
      a = Tensor.new([3, 2, 2]) { |i| i }
      t = N.atleast_3d(a)
      assert_array_equal t, a
    end
  end

  describe "Assemble#hstack" do
    it "test 1d tensors" do
      a = Tensor.new(1)
      b = Tensor.new(2)
      expected = Tensor.from_array [1, 2]
      assert_array_equal N.hstack([a, b]), expected
    end

    it "test 2d tensors" do
      a = Tensor.from_array [[1], [2]]
      b = Tensor.from_array [[1], [2]]
      res = N.hstack([a, b])
      desired = Tensor.from_array [[1, 1], [2, 2]]
      assert_array_equal res, desired
    end
  end

  describe "Assemble#vstack" do
    it "test 1d tensors" do
      a = Tensor.new(1)
      b = Tensor.new(2)
      expected = Tensor.from_array [[1], [2]]
      assert_array_equal N.vstack([a, b]), expected
    end

    it "test 2d tensors" do
      a = Tensor.from_array [[1], [2]]
      b = Tensor.from_array [[1], [2]]
      res = N.vstack([a, b])
      desired = Tensor.from_array [[1], [2], [1], [2]]
      assert_array_equal res, desired
    end

    it "test 2d tensors again" do
      a = Tensor.from_array [1, 2]
      b = Tensor.from_array [1, 2]
      res = N.vstack([a, b])
      desired = Tensor.from_array [[1, 2], [1, 2]]
      assert_array_equal res, desired
    end
  end

  describe "Assemble#concatenate" do
    it "test returns copy" do
      a = N.eye(3)
      b = N.concatenate([a], 0)
      b[0, 0] = 2
      (b[0, 0].value != a[0, 0].value).should be_true
    end

    it "test concatenate slices" do
      res = Tensor.new([2, 3, 7]) { |i| i }
      a0 = res[..., ..., ...4]
      a1 = res[..., ..., 4...6]
      a2 = res[..., ..., 6...]
      assert_array_equal N.concatenate([a0, a1, a2], 2), res
      assert_array_equal N.concatenate([a0, a1, a2], -1), res
      assert_array_equal N.concatenate([a0.transpose, a1.transpose, a2.transpose], 0), res.transpose
    end

    it "test shape error off axis" do
      a = Tensor.from_array [[1, 2, 3], [2, 3, 4]]
      b = Tensor.from_array [[1, 2], [3, 4]]
      expect_raises ShapeError do
        N.concatenate([a, b], 0)
      end
    end

    it "test shape error number of axis" do
      a = Tensor.new(3)
      b = Tensor.from_array [[1, 2]]
      expect_raises(ShapeError) do
        N.concatenate([a, b], 0)
      end
    end
  end

  describe "Assemble#dstack" do
    it "test single element tensors" do
      a = Tensor.new(1)
      b = Tensor.new(2)
      desired = Tensor.from_array [[[1, 2]]]
      assert_array_equal N.dstack([a, b]), desired
    end

    it "test 1d tensors" do
      a = N.arange(3)
      b = N.arange(3) ** 2
      desired = Tensor.from_array [[[0, 0], [1, 1], [2, 4]]]
      assert_array_equal N.dstack([a, b]), desired
    end

    it "test 2d tensors" do
      a = N.arange(4).reshape([2, 2])
      desired = Tensor.from_array [[[0, 0], [1, 1]], [[2, 2], [3, 3]]]
      assert_array_equal N.dstack([a, a]), desired
    end

    it "raises on higher dimensional tensors" do
      a = Tensor.new([3, 2, 2]) { |i| i }
      expect_raises(ShapeError) do
        N.dstack([a, a])
      end
    end
  end

  describe "Assemble#column_stack" do
    it "test single element tensors" do
      a = Tensor.new(1)
      b = Tensor.new(2)
      desired = Tensor.from_array [[1, 2]]
      assert_array_equal N.column_stack([a, b]), desired
    end

    it "test 1d tensors" do
      a = N.arange(3)
      b = N.arange(3) ** 2
      desired = Tensor.from_array [[0, 0], [1, 1], [2, 4]]
      assert_array_equal N.column_stack([a, b]), desired
    end

    it "test 2d tensors" do
      a = N.arange(4).reshape([2, 2])
      desired = Tensor.from_array [[0, 1, 0, 1], [2, 3, 2, 3]]
      assert_array_equal N.column_stack([a, a]), desired
    end

    it "raises on higher dimensional tensors" do
      a = Tensor.new([3, 2, 2]) { |i| i }
      expect_raises(ShapeError) do
        N.column_stack([a, a])
      end
    end
  end
end
