require "../tensor/tensor"

module Num::Einsum
  abstract struct SingletonContraction
    abstract def contract_singleton(tensor : Tensor)
  end

  struct Identity < SingletonContraction
    def contract_singleton(tensor : Tensor)
      tensor.dup
    end
  end

  struct Permutation < SingletonContraction
    @permutation : Array(Int32)

    def initialize(sc : SizedContraction)
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

    def contract_singleton(tensor : Tensor)
      tensor.transpose(@permutation).dup
    end
  end

  struct Summation < SingletonContraction
    @orig_axis_list : Array(Int32)
    @adjusted_axis_list : Array(Int32)

    def initialize(sc : SizedContraction)
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

    def contract_singleton(tensor : Tensor)
      result = tensor.sum(axis: @adjusted_axis_list[0])
      @adjusted_axis_list[1...].each do |axis|
        result = result.sum(axis: axis)
      end
      result
    end
  end

  struct Diagonalization < SingletonContraction
    @input_to_output_mapping : Array(Int32)
    @output_shape : Array(Int32)

    def initialize(sc : SizedContraction)
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

    def contract_singleton(tensor : Tensor)
      newshape = @output_shape
      newstrides = [0] * @output_shape.size

      tensor.strides.each_with_index do |stride, idx|
        newstrides[@input_to_output_mapping[idx]] += stride
      end

      tensor.as_strided(newshape, newstrides).dup
    end
  end

  struct PermutationAndSummation < SingletonContraction
    @permutation : Permutation
    @summation : Summation

    def initialize(sc : SizedContraction)
      output_order = [] of Int32
      sc.contraction.output_indices.each do |output_char|
        input_pos = sc.contraction.operand_indices[0].index { |x| x == output_char }
        if input_pos.nil?
          raise "Bad input"
        end
        output_order << input_pos
      end

      sc.contraction.operand_indices[0].each_with_index do |input_char, i|
        val = sc.contraction.output_indices.find { |x| x == input_char }
        if val.nil?
          output_order << i
        end
      end

      @permutation = Permutation.new(output_order)
      @summation = Summation.new(sc)
    end

    def contract_singleton(tensor : Tensor)
      permuted = @permutation.contract_singleton(tensor)
      @summation.contract_singleton(permuted)
    end
  end

  struct DiagonalizationAndSummation < SingletonContraction
    @diagonalization : Diagonalization
    @summation : Summation

    def initialize(sc : SizedContraction)
      @diagonalization = Diagonalization.new(sc)
      @summation = Summation.new(sc.contraction.output_indices.size, @diagonalization.@output_shape.size - sc.contraction.output_indices.size)
    end

    def contract_singleton(tensor : Tensor)
      contracted_singleton = @diagonalization.contract_singleton(tensor)
      @summation.contract_singleton(contracted_singleton)
    end
  end

  struct Inputter
    def self.find_outputs_in_inputs_unique(output_indices : Array(Char), input_indices : Array(Char))
      ret = maybe_find_outputs_in_inputs_unique(output_indices, input_indices)
      valid = [] of Int32
      ret.each do |e|
        if !e.nil?
          valid << e
        end
      end
      valid
    end

    def self.maybe_find_outputs_in_inputs_unique(output_indices, input_indices)
      ret = output_indices.map do |output_char|
        input_indices.index { |input_char| input_char == output_char }
      end
    end
  end

  abstract struct PairContraction
    abstract def contract_pair(lhs : Tensor, rhs : Tensor)
  end

  struct TensordotFixedPosition < PairContraction
    @len_uncontracted_lhs : Int32
    @len_uncontracted_rhs : Int32
    @len_contracted_axes : Int32
    @output_shape : Array(Int32)

    def initialize(sc : SizedContraction)
      lhs_indices = sc.contraction.operand_indices[0]
      rhs_indices = sc.contraction.operand_indices[1]
      output_indices = sc.contraction.output_indices

      twice_num_contracted_axes = lhs_indices.size + rhs_indices.size - output_indices.size

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

    def initialize(lhs_shape : Array(Int32), rhs_shape : Array(Int32), num_contracted_axes : Int32)
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

    def contract_pair(lhs : Tensor, rhs : Tensor)
      lhs = lhs.reshape(@len_uncontracted_lhs, @len_contracted_axes)
      rhs = rhs.reshape(@len_contracted_axes, @len_uncontracted_rhs)
      lhs.matmul(rhs)
    end
  end

  struct TensordotGeneral < PairContraction
    @lhs_permutation : Permutation
    @rhs_permutation : Permutation
    @tensordot_fixed_position : TensordotFixedPosition
    @output_permutation : Permutation

    def initialize(sc : SizedContraction)
      lhs_indices = sc.contraction.operand_indices[0]
      rhs_indices = sc.contraction.operand_indices[1]
      contracted_indices = sc.contraction.summation_indices
      output_indices = sc.contraction.output_indices
      lhs_shape = lhs_indices.map { |e| sc.output_size[e] }
      rhs_shape = rhs_indices.map { |e| sc.output_size[e] }

      lhs_contracted_axes = Inputter.find_outputs_in_inputs_unique(contracted_indices, lhs_indices)
      rhs_contracted_axes = Inputter.find_outputs_in_inputs_unique(contracted_indices, rhs_indices)

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
      output_order = Inputter.find_outputs_in_inputs_unique(output_indices, uncontracted_chars)

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

      @lhs_permutation = Permutation.new(permutation_lhs)
      @rhs_permutation = Permutation.new(permutation_rhs)
      @tensordot_fixed_position = TensordotFixedPosition.new(adjusted_lhs_shape, adjusted_rhs_shape, num_contracted_axes)
      @output_permutation = Permutation.new(output_order)
    end

    def contract_pair(lhs : Tensor, rhs : Tensor)
      permuted_lhs = @lhs_permutation.contract_singleton(lhs)
      permuted_rhs = @rhs_permutation.contract_singleton(rhs)
      tensordotted = @tensordot_fixed_position.contract_pair(permuted_lhs, permuted_rhs)
      @output_permutation.contract_singleton(tensordotted)
    end
  end

  struct HadamardProduct < PairContraction
    def contract_pair(lhs : Tensor, rhs : Tensor)
      lhs * rhs
    end
  end

  struct HadamardProductGeneral < PairContraction
    @lhs_permutation : Permutation
    @rhs_permutation : Permutation
    @hadamard_product : HadamardProduct

    def initialize(sc : SizedContraction)
      lhs_indices = sc.contraction.operand_indices[0]
      rhs_indices = sc.contraction.operand_indices[1]
      output_indices = sc.contraction.output_indices

      @lhs_permutation = Permutation.new(Inputter.find_outputs_in_inputs_unique(output_indices, lhs_indices))
      @rhs_permutation = Permutation.new(Inputter.find_outputs_in_inputs_unique(output_indices, rhs_indices))
      @hadamard_product = HadamardProduct.new
    end

    def contract_pair(lhs : Tensor, rhs : Tensor)
      @hadamard_product.contract_pair(
        @lhs_permutation.contract_singleton(lhs),
        @rhs_permutation.contract_singleton(rhs)
      )
    end
  end

  struct ScalarMatrixProduct < PairContraction
    def contract_pair(lhs : Tensor, rhs : Tensor)
      rhs * lhs.value
    end
  end

  struct ScalarMatrixProductGeneral < PairContraction
    @rhs_permutation : Permutation
    @scalar_matrix_product : ScalarMatrixProduct

    def initialize(sc : SizedContraction)
      lhs_indices = sc.contraction.operand_indices[0]
      rhs_indices = sc.contraction.operand_indices[1]
      output_indices = sc.contraction.output_indices

      @rhs_permutation = Permutation.new(Inputter.find_outputs_in_inputs_unique(output_indices, rhs_indices))
      @scalar_matrix_product = ScalarMatrixProduct.new
    end

    def contract_pair(lhs : Tensor, rhs : Tensor)
      @scalar_matrix_product.contract_pair(lhs, @rhs_permutation.contract_singleton(rhs))
    end
  end

  struct MatrixScalarProduct < PairContraction
    def contract_pair(lhs : Tensor, rhs : Tensor)
      lhs * rhs.value
    end
  end

  struct MatrixScalarProductGeneral
    @lhs_permutation : Permutation
    @matrix_scalar_product : MatrixScalarProduct

    def initialize(sc : SizedContraction)
      lhs_indices = sc.contraction.operand_indices[0]
      rhs_indices = sc.contraction.operand_indices[1]
      output_indices = sc.contraction.output_indices

      @lhs_permutation = Permutation.new(find_outputs_in_inputs_unique(output_indices, input_indices))
      @matrix_scalar_product = MatrixScalarProduct.new
    end

    def contract_pair(lhs : Tensor, rhs : Tensor)
      @matrix_scalar_product.contract_pair(@lhs_permutation.contract_singleton(lhs), rhs)
    end
  end

  struct StackedTensordotGeneral < PairContraction
    @lhs_permutation : Permutation
    @rhs_permutation : Permutation
    @lhs_output_shape : Array(Int32)
    @rhs_output_shape : Array(Int32)
    @intermediate_shape : Array(Int32)
    @tensordot_fixed_position : TensordotFixedPosition
    @output_shape : Array(Int32)
    @output_permutation : Permutation

    def initialize(sc : SizedContraction)
      lhs_order = [] of Int32
      rhs_order = [] of Int32
      @lhs_output_shape = [] of Int32
      @rhs_output_shape = [] of Int32
      @intermediate_shape = [] of Int32

      lhs_indices = sc.contraction.operand_indices[0]
      rhs_indices = sc.contraction.operand_indices[1]
      output_indices = sc.contraction.output_indices

      maybe_lhs_axes = Inputter.maybe_find_outputs_in_inputs_unique(output_indices, lhs_indices)
      maybe_rhs_axes = Inputter.maybe_find_outputs_in_inputs_unique(output_indices, rhs_indices)
      lhs_stack_axes = [] of Int32
      rhs_stack_axes = [] of Int32
      stack_indices = [] of Int32
      lhs_outer_axes = [] of Int32
      lhs_outer_indices = [] of Char
      rhs_outer_axes = [] of Int32
      rhs_outer_indices = [] of Char
      lhs_contracted_axes = [] of Int32
      rhs_contracted_axes = [] of Int32
      intermediate_indices = [] of Int32
      lhs_uncontracted_shape = [] of Int32
      rhs_uncontracted_shape = [] of Int32
      contracted_shape = [] of Int32

      lhs_output_shape << 1
      rhs_output_shape << 1

      maybe_lhs_axes.zip(maybe_rhs_axes, output_indices) do |maybe_lhs_pos, maybe_rhs_pos, output_char|
        case {maybe_lhs_pos.nil?, maybe_rhs_pos.nil?}
        when {false, false}
          lhs_stack_axes << maybe_lhs_pos.as?(Int32)
          rhs_stack_axes << maybe_rhs_pos.as?(Int32)
          stack_indices << output_char
          lhs_output_shape[0] *= sc.output_size[output_char]
          rhs_output_shape[0] *= sc.output_size[output_char]
        when {false, true}
          lhs_outer_axes << maybe_lhs_pos.as?(Int32)
          lhs_outer_indices << output_char
          lhs_uncontracted_shape << sc.output_size[output_char]
        when {true, false}
          rhs_outer_axes << maybe_rhs_pos.as?(Int32)
          rhs_outer_indices << output_char
          rhs_uncontracted_shape << sc.output_size[output_char]
        else
          raise "You done messed up AA-RON"
        end
      end

      lhs_indices.each_with_index do |lhs_char, lhs_pos|
        lhs_contracted_axes << lhs_pos
        other = rhs_contracted_axes.index { |x| x == lhs_char }
        if !other.nil?
          rhs_contracted_axes << other
        end
        contracted_shape << sc.output_size[lhs_char]
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

      @intermediate_shape << lhs_output_shape[0]
      lhs_outer_indices.each do |e|
        @intermediate_shape << sc.output_size[e]
      end
      rhs_outer_indices.each do |e|
        @intermediate_shape << sc.output_size[e]
      end

      output_order = find_outputs_in_inputs_unique(output_indices, intermediate_indices)
      @output_shape = intermediate_indices.map { |e| sc.output_size[c] }

      @tensordot_fixed_position = TensordotFixedPosition.new(lhs_output_shape[1...], rhs_output_shape[1...], lhs_contracted_axes.size)
      @lhs_permutation = Permutation.new(lhs_order)
      @rhs_permutation = Permutation.new(rhs_order)
      @output_permutation = Permutation.new(output_order)
    end

    def contract_pair(lhs : Tensor, rhs : Tensor)
      lhs_permuted = @lhs_permutation.contract_singleton(lhs)
      lhs_reshaped = lhs_permuted.reshape(@lhs_output_shape)

      rhs_permuted = @rhs_permutation.contract_singleton(rhs)
      rhs_reshaped = lhs_permuted.reshape(@rhs_output_shape)

      intermediate_result = Creation.zeros(@intermediate_shape)
    end
  end
end
