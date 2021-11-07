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

# :nodoc:
abstract struct Num::Einsum::PairContractor
  abstract def contract(lhs : Tensor(U, CPU(U)), rhs : Tensor(U, CPU(U))) forall U
end

# :nodoc:
# Performs tensor dot product for two tensors where no permutation needs to be performed,
# e.g. `ijk,jkl->il` or `ijk,klm->ijlm`.
#
# The axes to be contracted must be the last axes of the LHS tensor and the first axes
# of the RHS tensor, and the axis order for the output tensor must be all the uncontracted
# axes of the LHS tensor followed by all the uncontracted axes of the RHS tensor, in the
# orders those originally appear in the LHS and RHS tensors.
#
# The contraction is performed by reshaping the LHS into a matrix (2-D tensor) of shape
# [len_uncontracted_lhs, len_contracted_axes], reshaping the RHS into shape
# [len_contracted_axes, len_contracted_rhs], matrix-multiplying the two reshaped tensor,
# and then reshaping the result into [...self.output_shape].
struct Num::Einsum::TensordotFixedPosition < Num::Einsum::PairContractor
  # The product of the lengths of all the uncontracted axes in the LHS (or 1 if all of the
  # LHS axes are contracted)
  @len_uncontracted_lhs : Int32

  # The product of the lengths of all the uncontracted axes in the RHS (or 1 if all of the
  # RHS axes are contracted)
  @len_uncontracted_rhs : Int32

  # The product of the lengths of all the contracted axes (or 1 if no axes are contracted,
  # i.e. the outer product is computed)
  @len_contracted_axes : Int32

  # The shape that the tensor dot product will be recast to
  @output_shape : Array(Int32)

  def initialize(sc : Num::Einsum::SizedContraction)
    lhs_indices = sc.contraction.operand_indices[0]
    rhs_indices = sc.contraction.operand_indices[1]
    output_indices = sc.contraction.output_indices

    twice_num_contracted_axes = lhs_indices.size + rhs_indices.size - \
      output_indices.size

    num_contracted_axes = twice_num_contracted_axes / 2

    lhs_shape = lhs_indices.map { |c| sc.output_size[c] }
    rhs_shape = rhs_indices.map { |c| sc.output_size[c] }

    @len_uncontracted_lhs = 1
    @len_uncontracted_rhs = 1
    len_contracted_lhs = 1
    len_contracted_rhs = 1
    @output_shape = [] of Int32

    num_axes_lhs = lhs_shape.size
    lhs_shape.each_with_index do |axis_length, axis|
      if axis < (num_axes_lhs - num_contracted_axes)
        @len_uncontracted_lhs *= axis_length
        @output_shape << axis_length
      else
        len_contracted_lhs *= axis_length
      end
    end

    rhs_shape.each_with_index do |axis_length, axis|
      if axis < num_contracted_axes
        len_contracted_rhs *= axis_length
      else
        @len_uncontracted_rhs *= axis_length
        @output_shape << axis_length
      end
    end

    @len_contracted_axes = len_contracted_lhs
  end

  # Compute the uncontracted and contracted axis lengths and the output shape based on the
  # input shapes and how many axes should be contracted from each tensor.
  def initialize(
    lhs_shape : Array(Int32),
    rhs_shape : Array(Int32),
    num_contracted_axes : Int32
  )
    @len_uncontracted_lhs = 1
    @len_uncontracted_rhs = 1
    len_contracted_lhs = 1
    len_contracted_rhs = 1
    @output_shape = [] of Int32

    num_axes_lhs = lhs_shape.size
    lhs_shape.each_with_index do |axis_length, axis|
      if axis < (num_axes_lhs - num_contracted_axes)
        @len_uncontracted_lhs *= axis_length
        @output_shape << axis_length
      else
        len_contracted_lhs *= axis_length
      end
    end

    rhs_shape.each_with_index do |axis_length, axis|
      if axis < num_contracted_axes
        len_contracted_rhs *= axis_length
      else
        @len_uncontracted_rhs *= axis_length
        @output_shape << axis_length
      end
    end

    @len_contracted_axes = len_contracted_lhs
  end

  def contract(lhs : Tensor(U, CPU(U)), rhs : Tensor(U, CPU(U))) forall U
    lhs = lhs.reshape(@len_uncontracted_lhs, @len_contracted_axes)
    rhs = rhs.reshape(@len_contracted_axes, @len_uncontracted_rhs)
    lhs.matmul(rhs).reshape(@output_shape)
  end
end

# :nodoc:
# Computes the tensor dot product of two tensors, with individual permutations of the
# LHS and RHS performed as necessary, as well as a final permutation of the output.
#
# Examples that qualify for TensordotGeneral but not TensordotFixedPosition:
#
# 1. `jik,jkl->il` LHS tensor needs to be permuted `jik->ijk`
# 2. `ijk,klm->imlj` Output tensor needs to be permuted `ijlm->imlj`
struct Num::Einsum::TensordotGeneral < Num::Einsum::PairContractor
  @lhs_permutation : Num::Einsum::Permutation
  @rhs_permutation : Num::Einsum::Permutation
  @tensordot_fixed_position : Num::Einsum::TensordotFixedPosition
  @output_permutation : Num::Einsum::Permutation

  def initialize(sc : Num::Einsum::SizedContraction)
    lhs_indices = sc.contraction.operand_indices[0]
    rhs_indices = sc.contraction.operand_indices[1]
    contracted_indices = sc.contraction.summation_indices
    output_indices = sc.contraction.output_indices
    lhs_shape = lhs_indices.map { |e| sc.output_size[e] }
    rhs_shape = rhs_indices.map { |e| sc.output_size[e] }

    lhs_contracted_axes = Num::Einsum::Inputter.find_outputs_in_inputs_unique(
      contracted_indices, lhs_indices
    )
    rhs_contracted_axes = Num::Einsum::Inputter.find_outputs_in_inputs_unique(
      contracted_indices, rhs_indices
    )

    uncontracted_chars = lhs_indices.dup
    uncontracted_chars.select! do |input_char|
      val = output_indices.index { |output_char| input_char == output_char }
      !val.nil?
    end

    rhs_uncontracted_chars = rhs_indices.dup
    rhs_uncontracted_chars.select! do |input_char|
      val = output_indices.index { |output_char| input_char == output_char }
      !val.nil?
    end

    uncontracted_chars += rhs_uncontracted_chars
    output_order = Num::Einsum::Inputter.find_outputs_in_inputs_unique(
      output_indices, uncontracted_chars
    )

    num_contracted_axes = lhs_contracted_axes.size
    lhs_uniques = lhs_contracted_axes.to_set
    rhs_uniques = rhs_contracted_axes.to_set

    adjusted_lhs_shape = [] of Int32
    adjusted_rhs_shape = [] of Int32

    permutation_lhs = [] of Int32
    lhs_shape.each_with_index do |axis_length, i|
      if !lhs_uniques.includes?(i)
        permutation_lhs << i
        adjusted_lhs_shape << axis_length
      end
    end

    lhs_contracted_axes.each do |i|
      permutation_lhs << i
      adjusted_lhs_shape << lhs_shape[i]
    end

    permutation_rhs = [] of Int32
    rhs_contracted_axes.each do |i|
      permutation_rhs << i
      adjusted_rhs_shape << rhs_shape[i]
    end

    rhs_shape.each_with_index do |axis_length, i|
      if !rhs_uniques.includes?(i)
        permutation_rhs << i
        adjusted_rhs_shape << axis_length
      end
    end

    @lhs_permutation = Num::Einsum::Permutation.new(permutation_lhs)
    @rhs_permutation = Num::Einsum::Permutation.new(permutation_rhs)
    @tensordot_fixed_position = Num::Einsum::TensordotFixedPosition.new(
      adjusted_lhs_shape, adjusted_rhs_shape, num_contracted_axes
    )
    @output_permutation = Num::Einsum::Permutation.new(output_order)
  end

  def contract(lhs : Tensor(U, CPU(U)), rhs : Tensor(U, CPU(U))) forall U
    permuted_lhs = @lhs_permutation.contract(lhs)
    permuted_rhs = @rhs_permutation.contract(rhs)
    tensordotted = @tensordot_fixed_position.contract(
      permuted_lhs, permuted_rhs
    )
    @output_permutation.contract(tensordotted)
  end
end

# :nodoc:
# Computes the Hadamard (element-wise) product of two tensors.
#
# All instances of `SizedContraction` making use of this contractor must have the form
# `ij,ij->ij`.
#
# Contractions of the form `ij,ji->ij` need to use `HadamardProductGeneral` instead.
struct Num::Einsum::HadamardProduct < Num::Einsum::PairContractor
  def contract(lhs : Tensor(U, CPU(U)), rhs : Tensor(U, CPU(U))) forall U
    lhs * rhs
  end
end

# :nodoc:
# Permutes the axes of the LHS and RHS tensors to the order in which those axes appear in the
# output before computing the Hadamard (element-wise) product.
#
# Examples of contractions that fit this category:
#
# 1. `ij,ij->ij` (Can also can use `HadamardProduct`)
# 2. `ij,ji->ij` (Can only use `HadamardProductGeneral`)
struct Num::Einsum::HadamardProductGeneral < Num::Einsum::PairContractor
  @lhs_permutation : Num::Einsum::Permutation
  @rhs_permutation : Num::Einsum::Permutation
  @hadamard_product : Num::Einsum::HadamardProduct

  def initialize(sc : Num::Einsum::SizedContraction)
    lhs_indices = sc.contraction.operand_indices[0]
    rhs_indices = sc.contraction.operand_indices[1]
    output_indices = sc.contraction.output_indices

    @lhs_permutation = Num::Einsum::Permutation.new(
      Inputter.find_outputs_in_inputs_unique(output_indices, lhs_indices)
    )
    @rhs_permutation = Num::Einsum::Permutation.new(
      Inputter.find_outputs_in_inputs_unique(output_indices, rhs_indices)
    )
    @hadamard_product = Num::Einsum::HadamardProduct.new
  end

  def contract(lhs : Tensor(U, CPU(U)), rhs : Tensor(U, CPU(U))) forall U
    @hadamard_product.contract(
      @lhs_permutation.contract(lhs),
      @rhs_permutation.contract(rhs)
    )
  end
end

# :nodoc:
# Multiplies every element of the RHS tensor by the single scalar in the 0-d LHS tensor.
#
# This contraction can arise when the simplification of the LHS tensor results in all the
# axes being summed before the two tensors are contracted. For example, in the contraction
# `i,jk->jk`, every element of the RHS tensor is simply multiplied by the sum of the elements
# of the LHS tensor.
struct Num::Einsum::ScalarMatrixProduct < Num::Einsum::PairContractor
  def contract(lhs : Tensor(U, CPU(U)), rhs : Tensor(U, CPU(U))) forall U
    rhs * lhs.value
  end
end

# :nodoc:
# Permutes the axes of the RHS tensor to the output order and multiply all elements by the single
# scalar in the 0-d LHS tensor.
#
# This contraction can arise when the simplification of the LHS tensor results in all the
# axes being summed before the two tensors are contracted. For example, in the contraction
# `i,jk->kj`, the output matrix is equal to the RHS matrix, transposed and then scalar-multiplied
# by the sum of the elements of the LHS tensor.
struct Num::Einsum::ScalarMatrixProductGeneral < Num::Einsum::PairContractor
  @rhs_permutation : Num::Einsum::Permutation
  @scalar_matrix_product : Num::Einsum::ScalarMatrixProduct

  def initialize(sc : Num::Einsum::SizedContraction)
    lhs_indices = sc.contraction.operand_indices[0]
    rhs_indices = sc.contraction.operand_indices[1]
    output_indices = sc.contraction.output_indices

    @rhs_permutation = Num::Einsum::Permutation.new(
      Num::Einsum::Inputter.find_outputs_in_inputs_unique(output_indices, rhs_indices)
    )
    @scalar_matrix_product = Num::Einsum::ScalarMatrixProduct.new
  end

  def contract(lhs : Tensor(U, CPU(U)), rhs : Tensor(U, CPU(U))) forall U
    @scalar_matrix_product.contract(lhs, @rhs_permutation.contract(rhs))
  end
end

# :nodoc:
# Multiplies every element of the LHS tensor by the single scalar in the 0-d RHS tensor.
#
# This contraction can arise when the simplification of the LHS tensor results in all the
# axes being summed before the two tensors are contracted. For example, in the contraction
# `ij,k->ij`, every element of the LHS tensor is simply multiplied by the sum of the elements
# of the RHS tensor.
struct Num::Einsum::MatrixScalarProduct < Num::Einsum::PairContractor
  def contract(lhs : Tensor(U, CPU(U)), rhs : Tensor(U, CPU(U))) forall U
    lhs * rhs.value
  end
end

# :nodoc:
# Permutes the axes of the LHS tensor to the output order and multiply all elements by the single
# scalar in the 0-d RHS tensor.
#
# This contraction can arise when the simplification of the RHS tensor results in all the
# axes being summed before the two tensors are contracted. For example, in the contraction
# `ij,k->ji`, the output matrix is equal to the LHS matrix, transposed and then scalar-multiplied
# by the sum of the elements of the RHS tensor.
struct Num::Einsum::MatrixScalarProductGeneral < Num::Einsum::PairContractor
  @lhs_permutation : Num::Einsum::Permutation
  @matrix_scalar_product : Num::Einsum::MatrixScalarProduct

  def initialize(sc : Num::Einsum::SizedContraction)
    lhs_indices = sc.contraction.operand_indices[0]
    rhs_indices = sc.contraction.operand_indices[1]
    output_indices = sc.contraction.output_indices

    @lhs_permutation = Num::Einsum::Permutation.new(
      Num::Einsum::Inputter.find_outputs_in_inputs_unique(lhs_indices, output_indices)
    )
    @matrix_scalar_product = Num::Einsum::MatrixScalarProduct.new
  end

  def contract(lhs : Tensor(U, CPU(U)), rhs : Tensor(U, CPU(U))) forall U
    @matrix_scalar_product.contract(
      @lhs_permutation.contract(lhs), rhs
    )
  end
end

# :nodoc:
# Repeatedly computes the tensor dot of subviews of the two tensors, iterating over
# indices which appear in the LHS, RHS, and output.
#
# The indices appearing in all three places are referred to here as the "stack" indices.
# For example, in the contraction `ijk,ikl->ijl`, `i` would be the (only) "stack" index.
# This contraction is an instance of batch matrix multiplication. The LHS and RHS are both
# 3-D tensors and the `i`th (2-D) subview of the output is the matrix product of the `i`th
# subview of the LHS matrix-multiplied by the `i`th subview of the RHS.
#
# This is the most general contraction and in theory could handle all pairwise contractions,
# but is less performant than special-casing when there are no "stack" indices. It is also
# currently the only case that requires `.outer_iter_mut()` (which might make parallelizing
# operations more difficult).
struct Num::Einsum::StackedTensordotGeneral < Num::Einsum::PairContractor
  @lhs_permutation : Num::Einsum::Permutation
  @rhs_permutation : Num::Einsum::Permutation
  @lhs_output_shape : Array(Int32)
  @rhs_output_shape : Array(Int32)
  @intermediate_shape : Array(Int32)
  @tensordot_fixed_position : Num::Einsum::TensordotFixedPosition
  @output_shape : Array(Int32)
  @output_permutation : Num::Einsum::Permutation

  def initialize(sc : Num::Einsum::SizedContraction)
    lhs_order = [] of Int32
    rhs_order = [] of Int32
    @lhs_output_shape = [] of Int32
    @rhs_output_shape = [] of Int32
    @intermediate_shape = [] of Int32

    lhs_indices = sc.contraction.operand_indices[0]
    rhs_indices = sc.contraction.operand_indices[1]
    output_indices = sc.contraction.output_indices

    maybe_lhs_axes = Num::Einsum::Inputter.maybe_find_outputs_in_inputs_unique(
      output_indices, lhs_indices
    )
    maybe_rhs_axes = Num::Einsum::Inputter.maybe_find_outputs_in_inputs_unique(
      output_indices, rhs_indices
    )
    lhs_stack_axes = [] of Int32
    rhs_stack_axes = [] of Int32
    stack_indices = [] of Char
    lhs_outer_axes = [] of Int32
    lhs_outer_indices = [] of Char
    rhs_outer_axes = [] of Int32
    rhs_outer_indices = [] of Char
    lhs_contracted_axes = [] of Int32
    rhs_contracted_axes = [] of Int32
    intermediate_indices = [] of Char
    lhs_uncontracted_shape = [] of Int32
    rhs_uncontracted_shape = [] of Int32
    contracted_shape = [] of Int32

    @lhs_output_shape << 1
    @rhs_output_shape << 1

    maybe_lhs_axes.zip(maybe_rhs_axes, output_indices) do |maybe_lhs_pos, maybe_rhs_pos, output_char|
      case {maybe_lhs_pos.nil?, maybe_rhs_pos.nil?}
      when {false, false}
        lhs_stack_axes << maybe_lhs_pos.not_nil!
        rhs_stack_axes << maybe_rhs_pos.not_nil!
        stack_indices << output_char
        @lhs_output_shape[0] *= sc.output_size[output_char]
        @rhs_output_shape[0] *= sc.output_size[output_char]
      when {false, true}
        lhs_outer_axes << maybe_lhs_pos.not_nil!
        lhs_outer_indices << output_char
        lhs_uncontracted_shape << sc.output_size[output_char]
      when {true, false}
        rhs_outer_axes << maybe_rhs_pos.not_nil!
        rhs_outer_indices << output_char
        rhs_uncontracted_shape << sc.output_size[output_char]
      else
        raise Num::Exceptions::ValueError.new("Invalid inputs")
      end
    end

    lhs_indices.each_with_index do |lhs_char, lhs_pos|
      if output_indices.index { |x| x == lhs_char }.nil?
        lhs_contracted_axes << lhs_pos
        other = rhs_indices.index { |x| x == lhs_char }.not_nil!
        rhs_contracted_axes << other
        contracted_shape << sc.output_size[lhs_char]
      end
    end

    lhs_order += lhs_stack_axes
    lhs_order += lhs_outer_axes
    @lhs_output_shape += lhs_uncontracted_shape
    lhs_order += lhs_contracted_axes
    @lhs_output_shape += contracted_shape

    rhs_order += rhs_stack_axes.dup
    rhs_order += rhs_contracted_axes
    @rhs_output_shape += contracted_shape
    rhs_order += rhs_outer_axes
    @rhs_output_shape += rhs_uncontracted_shape

    intermediate_indices += stack_indices
    intermediate_indices += lhs_outer_indices
    intermediate_indices += rhs_outer_indices

    @intermediate_shape << @lhs_output_shape[0]
    lhs_outer_indices.each do |e|
      @intermediate_shape << sc.output_size[e]
    end
    rhs_outer_indices.each do |e|
      @intermediate_shape << sc.output_size[e]
    end

    output_order = Num::Einsum::Inputter.find_outputs_in_inputs_unique(output_indices, intermediate_indices)
    @output_shape = intermediate_indices.map { |c| sc.output_size[c] }

    @tensordot_fixed_position = Num::Einsum::TensordotFixedPosition.new(
      @lhs_output_shape[1...], @rhs_output_shape[1...], lhs_contracted_axes.size
    )
    @lhs_permutation = Num::Einsum::Permutation.new(lhs_order)
    @rhs_permutation = Num::Einsum::Permutation.new(rhs_order)
    @output_permutation = Num::Einsum::Permutation.new(output_order)
  end

  def contract(lhs : Tensor(U, CPU(U)), rhs : Tensor(U, CPU(U))) forall U
    lhs_permuted = @lhs_permutation.contract(lhs)
    lhs_reshaped = lhs_permuted.reshape(@lhs_output_shape)

    rhs_permuted = @rhs_permutation.contract(rhs)
    rhs_reshaped = rhs_permuted.reshape(@rhs_output_shape)

    intermediate_result = Tensor(U, CPU(U)).zeros(@intermediate_shape)

    intermediate_result.shape[0].times do |i|
      intermediate_result[i] = @tensordot_fixed_position.contract(
        lhs_reshaped[i], rhs_reshaped[i]
      )
    end

    @output_permutation.contract(intermediate_result.reshape(@output_shape))
  end
end
