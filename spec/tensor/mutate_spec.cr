require "../spec_helper"

def get_tensor
  Tensor.new([2, 2, 3]) { |i| i }
end

def get_matrix
  Tensor.new([3, 3]) { |i| i }
end

describe Tensor do
  describe "Tensor#mutate" do
    it "duplicating an array equals base" do
      t = get_tensor
      result = t.dup
      assert_array_equal result, t
    end

    it "duplicating an array with Fortran ordering works" do
      t = get_tensor
      result = t.dup('F')
      result.flags.fortran?.should be_true
    end

    it "duplicating an array with C ordering works" do
      t = get_tensor
      result = t.dup('C')
      result.flags.contiguous?.should be_true
    end

    it "duplicate C array with F ordering equals base" do
      t = get_tensor
      result = t.dup('F')
      assert_array_equal result, t
    end

    it "duplicate F array with C ordering equals base" do
      t = Tensor.new([2, 2, 3], 'F') { |i| i }
      result = t.dup('C')
      assert_array_equal result, t
    end

    it "duplicate owns data" do
      t = get_tensor
      result = t.dup
      result.flags.own_data?.should be_true
    end

    it "dup view matches base" do
      t = get_tensor
      view = t.dup_view
      assert_array_equal t, view
    end

    it "dup view doesn't own its own data" do
      t = get_tensor
      view = t.dup_view
      view.flags.own_data?.should be_false
    end

    it "dup view mutates base" do
      t = get_tensor
      view = t.dup_view
      view[1] = 99
      assert_array_equal t, view
    end

    it "dup view is the same as full slice" do
      t = get_tensor
      view = t.dup_view
      view2 = t[...]
      assert_array_equal view, view2
    end

    it "diag view correct output C array" do
      t = get_matrix
      view = t.diag_view
      expected = Tensor.from_array([3], [0, 4, 8])
      assert_array_equal view, expected
    end

    it "diag view can mutate array" do
      t = get_matrix
      view = t.diag_view
      view[...] = 99
      expected = Tensor.from_array([3, 3], [[99, 1, 2], [3, 99, 5], [6, 7, 99]])
      assert_array_equal t, expected
    end

    it "diag view does not own memory" do
      t = get_matrix
      view = t.diag_view
      view.flags.own_data?.should be_false
    end

    it "diag view raises shape error for nd tensors" do
      t = get_tensor
      expect_raises(NumInternal::ShapeError) do
        t.diag_view
      end
    end
  end

  describe "Tensor#reshape" do
    it "basic reshape test" do
      t = get_tensor
      r = t.reshape([6, 2])
      expected = Tensor.new([6, 2]) { |i| i }
      assert_array_equal r, expected
    end

    it "reshape can infer a dimension" do
      t = get_tensor
      r = t.reshape([6, -1])
      expected = Tensor.new([6, 2]) { |i| i }
      assert_array_equal r, expected
    end

    it "reshape raises on multiple inferred dimensions" do
      t = get_tensor
      expect_raises(NumInternal::ValueError) do
        t.reshape([2, -1, -1])
      end
    end

    it "contiguous reshape does not own data" do
      t = get_tensor
      r = t.reshape([6, -1])
      r.flags.own_data?.should be_false
    end

    it "non-contiguous reshape owns data" do
      t = get_tensor
      view = t[..., 1]
      r = view.reshape([-1])
      r.flags.own_data?.should be_true
    end

    it "contiguous reshape mutates base" do
      t = get_tensor
      t.reshape([2, -1])[...] = 99
      expected = Tensor.new(t.shape) { |_| 99 }
      assert_array_equal t, expected
    end
  end

  describe "Tensor#transpose" do
    it "basic tranpose test" do
      t = get_matrix
      expected = Tensor.new([3, 3], 'F') { |i| i }
      result = t.transpose
      assert_array_equal result, expected
    end

    it "transpose returns a view" do
      t = Tensor.new([3, 3]) { |i| i }
      view = t.transpose
      view[...] = 99
      assert_array_equal t, view
    end

    it "explicit order transpose" do
      t = Tensor.new([2, 2, 3]) { |i| i }
      expected = Tensor.from_array([3, 2, 2],
        [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11])
      result = t.transpose([2, 0, 1])
      assert_array_equal result, expected
    end

    it "C to F transpose swaps memory order" do
      t = Tensor.new([2, 2, 3]) { |i| i }
      result = t.transpose
      result.flags.fortran?.should be_true
    end

    it "F to C transpose swaps memory order" do
      t = Tensor.new([2, 2, 3]) { |i| i }
      t = t.dup('F')
      result = t.transpose
      result.flags.contiguous?.should be_true
    end
  end
end
