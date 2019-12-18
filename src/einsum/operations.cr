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

      num_contracted_axes = twice_num_contracted_axes // 2

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
          len_uncontracted_lhs *= axis_length
          output_shape << axis_length
        else
          len_contracted_lhs *= axis_length
        end
      end

      rhs_shape.each_with_index do |axis_length, axis|
        if axis < num_contracted_axes
          len_uncontracted_rhs *= axis_length
          output_shape << axis_length
        else
          len_contracted_rhs *= axis_length
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
end
