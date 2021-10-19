# Copyright (c) 2021 Crystal Data Contributors
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# abstract struct Num::Einsum::SingletonViewer
#   abstract def view(arr : Tensor(U, CPU(U))) forall U
# end

abstract struct Num::Einsum::SingletonContractor
  abstract def contract(arr : Tensor(U, CPU(U))) forall U
end

# abstract struct Num::Einsum::SingletonViewerAndContractor
#   abstract def view(arr : Tensor(U, CPU(U))) forall U
#   abstract def contract(arr : Tensor(U, CPU(U))) forall U
# end

# Returns a view or clone of the input tensor.
#
# Example: `ij->ij`
struct Num::Einsum::Identity < Num::Einsum::SingletonContractor
  def contract(arr : Tensor(U, CPU(U))) forall U
    arr.dup(Num::RowMajor)
  end

  def view(arr : Tensor(U, CPU(U))) forall U
    arr.view
  end
end

# Permutes the axes of the input tensor and returns a view or clones
# the elements.
#
# Example: `ij->ji`
struct Num::Einsum::Permutation < Num::Einsum::SingletonContractor
  @permutation : Array(Int32)

  def initialize(sc : Num::Einsum::SizedContraction)
    @permutation = [] of Int32
    sc.contraction.output_indices.each do |c|
      val = sc.contraction.operand_indices[0].index { |x| x == c }
      if !val.nil?
        @permutation << val
      end
    end
  end

  def initialize(@permutation : Array(Int32))
  end

  def contract(arr : Tensor(U, CPU(U))) forall U
    arr.transpose(@permutation).dup
  end

  def view(arr : Tensor(U, CPU(U))) forall U
    arr.transpose(@permutation)
  end
end

# Sums across the elements of the input tensor that don't appear in the output
# tensor.
#
# Example: `ij->i`
struct Num::Einsum::Summation < Num::Einsum::SingletonContractor
  @orig_axis_list : Array(Int32)
  @adjusted_axis_list : Array(Int32)

  def initialize(sc : Num::Einsum::SizedContraction)
    output_indices = sc.contraction.output_indices
    input_indices = sc.contraction.operand_indices[0]

    start_index = output_indices.size
    num_summed_axes = input_indices.size - output_indices.size

    @orig_axis_list = (start_index...(start_index + num_summed_axes)).to_a
    @adjusted_axis_list = (0...num_summed_axes).map { |e| start_index }
  end

  def initialize(start_index : Int32, num_summed_axes : Int32)
    @orig_axis_list = (start_index...(start_index + num_summed_axes)).to_a
    @adjusted_axis_list = (0...num_summed_axes).map { |e| start_index }
  end

  def contract(arr : Tensor(U, CPU(U))) forall U
    result = arr.sum(axis: @adjusted_axis_list[0])
    @adjusted_axis_list[1...].each do |axis|
      result = result.sum(axis: axis)
    end
    result
  end
end

# Returns the elements of the input tensor where all instances of the repeated
# indices are equal to one another.
#
# Optionally permutes the axes of the tensor as well.
#
# Examples:
#
# 1. `ii->i`
# 2. `iij->ji`
struct Num::Einsum::Diagonalization < Num::Einsum::SingletonContractor
  @input_to_output_mapping : Array(Int32)
  @output_shape : Array(Int32)

  def initialize(sc : Num::Einsum::SizedContraction)
    adjusted_output_indices = sc.contraction.output_indices.dup
    @input_to_output_mapping = [] of Int32

    sc.contraction.operand_indices[0].each do |c|
      current_length = adjusted_output_indices.size
      val = adjusted_output_indices.index { |x| x == c }
      case val
      when nil
        adjusted_output_indices << c
        @input_to_output_mapping << current_length
      else
        @input_to_output_mapping << val
      end
    end

    @output_shape = adjusted_output_indices.map { |c| sc.output_size[c] }
  end

  def contract(arr : Tensor(U, CPU(U))) forall U
    newshape = @output_shape
    newstrides = [0] * @output_shape.size

    arr.strides.each_with_index do |stride, idx|
      newstrides[@input_to_output_mapping[idx]] += stride
    end

    arr.as_strided(newshape, newstrides).dup
  end
end

# Permutes the elements of the input tensor and sums across elements that don't appear in the output.
#
# Example: `ijk->kj`
struct Num::Einsum::PermutationAndSummation < Num::Einsum::SingletonContractor
  @permutation : Num::Einsum::Permutation
  @summation : Num::Einsum::Summation

  def initialize(sc : Num::Einsum::SizedContraction)
    output_order = [] of Int32
    sc.contraction.output_indices.each do |output_char|
      input_pos = sc.contraction.operand_indices[0].index do |x|
        x == output_char
      end
      if input_pos.nil?
        raise Num::Exceptions::ValueError.new("Bad input")
      end
      output_order << input_pos
    end

    sc.contraction.operand_indices[0].each_with_index do |input_char, i|
      val = sc.contraction.output_indices.find { |x| x == input_char }
      if val.nil?
        output_order << i
      end
    end

    @permutation = Num::Einsum::Permutation.new(output_order)
    @summation = Num::Einsum::Summation.new(sc)
  end

  def contract(arr : Tensor(U, CPU(U))) forall U
    permuted = @permutation.contract(arr)
    @summation.contract(permuted)
  end
end

# Returns the elements of the input tensor where all instances of the repeated
# indices are equal to one another, optionally permuting the axes, and
# sums across indices that don't appear in the output.
#
# Examples:
#
# 1. `iijk->ik` (Diagonalizes the `i` axes and sums over `j`)
# 2. `jijik->ki` (Diagonalizes `i` and `j` and sums over `j` after diagonalization)
struct Num::Einsum::DiagonalizationAndSummation < Num::Einsum::SingletonContractor
  @diagonalization : Num::Einsum::Diagonalization
  @summation : Num::Einsum::Summation

  def initialize(sc : Num::Einsum::SizedContraction)
    @diagonalization = Num::Einsum::Diagonalization.new(sc)
    @summation = Num::Einsum::Summation.new(
      sc.contraction.output_indices.size,
      @diagonalization.@output_shape.size - sc.contraction.output_indices.size
    )
  end

  def contract(arr : Tensor(U, CPU(U))) forall U
    contracted_singleton = @diagonalization.contract(arr)
    @summation.contract(contracted_singleton)
  end
end
