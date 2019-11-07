require "../spec_helper"
require "../../src/util/*"
include Bottle
include Bottle::Internal::Exceptions
include Bottle::Internal

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
      Comparison.allclose(t, result).should be_true
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
      Comparison.allclose(t, result).should be_true
    end

    it "duplicate F array with C ordering equals base" do
      t = Tensor.new([2, 2, 3], ArrayFlags::Fortran) { |i| i }
      result = t.dup('C')
      Comparison.allclose(t, result).should be_true
    end

    it "duplicate owns data" do
      t = get_tensor
      result = t.dup
      result.flags.own_data?.should be_true
    end

    it "dup view matches base" do
      t = get_tensor
      view = t.dup_view
      Comparison.allclose(t, view).should be_true
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
      Comparison.allclose(t, view).should be_true
    end

    it "dup view is the same as full slice" do
      t = get_tensor
      view = t.dup_view
      view2 = t[...]
      Comparison.allclose(view2, view).should be_true
    end

    it "diag view correct output C array" do
      t = get_matrix
      view = t.diag_view
      expected = Tensor.from_array([3], [0, 4, 8])
      Comparison.allclose(view, expected).should be_true
    end

    it "diag view can mutate array" do
      t = get_matrix
      view = t.diag_view
      view[...] = 99
      expected = Tensor.from_array([3, 3], [[99, 1, 2], [3, 99, 5], [6, 7, 99]])
      Comparison.allclose(t, expected).should be_true
    end

    it "diag view does not own memory" do
      t = get_matrix
      view = t.diag_view
      view.flags.own_data?.should be_false
    end

    it "diag view raises shape error for nd tensors" do
      t = get_tensor
      expect_raises(ShapeError) do
        t.diag_view
      end
    end
  end

  describe "Tensor#reshape" do
    it "basic reshape test" do
      t = get_tensor
      r = t.reshape([6, 2])
      expected = Tensor.new([6, 2]) { |i| i }
      Comparison.allclose(r, expected).should be_true
    end

    it "reshape can infer a dimension" do
      t = get_tensor
      r = t.reshape([6, -1])
      expected = Tensor.new([6, 2]) { |i| i }
      Comparison.allclose(r, expected).should be_true
    end

    it "reshape raises on multiple inferred dimensions" do
      t = get_tensor
      expect_raises(ValueError) do
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
      Comparison.allclose(t, expected).should be_true
    end
  end
end
