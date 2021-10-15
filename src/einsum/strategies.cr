module Num::Einsum
  enum SingletonMethod
    Identity
    Permutation
    Summation
    Diagonalization
    PermutationAndSummation
    DiagonalizationAndSummation
  end

  struct SingletonSummary
    getter num_summed_axes : Int32
    getter num_diagonalized_axes : Int32
    getter num_reordered_axes : Int32

    def initialize(sc : SizedContraction)
      output_indices = sc.contraction.output_indices
      input_indices = sc.contraction.operand_indices[0]

      input_counts = Hash(Char, Int32).new
      input_indices.each do |c|
        input_counts[c] = input_counts.fetch(c, 0) + 1
      end

      @num_summed_axes = input_counts.size - output_indices.size
      @num_diagonalized_axes = input_counts.select { |k, v| v > 1 }.size
      tmp = output_indices.zip(input_indices)
      tmp.select! { |i, j| i != j }
      @num_reordered_axes = tmp.size
    end

    def get_strategy
      case {@num_summed_axes, @num_diagonalized_axes, @num_reordered_axes}
      when {0, 0, 0}
        SingletonMethod::Identity
      when {0, 0, _}
        SingletonMethod::Permutation
      when {_, 0, 0}
        SingletonMethod::Summation
      when {0, _, _}
        SingletonMethod::Diagonalization
      when {_, 0, _}
        SingletonMethod::PermutationAndSummation
      else
        SingletonMethod::DiagonalizationAndSummation
      end
    end
  end

  enum PairMethod
    HadamardProduct
    HadamardProductGeneral
    TensordotFixedPosition
    TensordotGeneral
    ScalarMatrixProduct
    ScalarMatrixProductGeneral
    MatrixScalarProduct
    MatrixScalarProductGeneral
    BroadcastProductGeneral
    StackedTensordotGeneral
  end

  struct PairSummary
    @num_stacked_axes : Int32
    @num_lhs_outer_axes : Int32
    @num_rhs_outer_axes : Int32
    @num_contracted_axes : Int32

    def initialize(sc : SizedContraction)
      output_indices = sc.contraction.output_indices
      lhs_indices = sc.contraction.operand_indices[0]
      rhs_indices = sc.contraction.operand_indices[1]

      lhs_uniques = lhs_indices.to_set
      rhs_uniques = rhs_indices.to_set
      output_uniques = output_indices.to_set

      lhs_and_rhs = lhs_uniques & rhs_uniques
      stacked = lhs_and_rhs & output_uniques

      @num_stacked_axes = stacked.size
      @num_contracted_axes = lhs_and_rhs.size - @num_stacked_axes
      @num_lhs_outer_axes = lhs_uniques.size - @num_stacked_axes - @num_contracted_axes
      @num_rhs_outer_axes = rhs_uniques.size - @num_stacked_axes - @num_contracted_axes
    end

    def get_strategy
      case {@num_contracted_axes, @num_lhs_outer_axes, @num_rhs_outer_axes, @num_stacked_axes}
      when {0, 0, 0, _}
        PairMethod::HadamardProductGeneral
      when {0, 0, _, 0}
        PairMethod::ScalarMatrixProductGeneral
      when {0, _, 0, 0}
        PairMethod::MatrixScalarProductGeneral
      when {_, _, _, 0}
        PairMethod::TensordotGeneral
      else
        PairMethod::StackedTensordotGeneral
      end
    end
  end

  private def single_einsum_dispatch(sc : SizedContraction, operand)
    strategy = SingletonSummary.new(sc).get_strategy
    case strategy
    when SingletonMethod::Identity
      op = Identity.new
    when SingletonMethod::Permutation
      op = Permutation.new(sc)
    when SingletonMethod::Summation
      op = Summation.new(sc)
    when SingletonMethod::Diagonalization
      op = Diagonalization.new(sc)
    when SingletonMethod::PermutationAndSummation
      op = PermutationAndSummation.new(sc)
    when SingletonMethod::DiagonalizationAndSummation
      op = DiagonalizationAndSummation.new(sc)
    else
      raise "Bad inputs"
    end
    op.contract(operand)
  end

  private def multiple_einsum_dispatch(sc : SizedContraction, lhs, rhs)
    strategy = PairSummary.new(sc).get_strategy
    case strategy
    when PairMethod::HadamardProductGeneral
      op = HadamardProductGeneral.new(sc)
    when PairMethod::ScalarMatrixProductGeneral
      op = ScalarMatrixProductGeneral.new(sc)
    when PairMethod::MatrixScalarProductGeneral
      op = MatrixScalarProductGeneral.new(sc)
    when PairMethod::TensordotGeneral
      op = TensordotGeneral.new(sc)
    when PairMethod::StackedTensordotGeneral
      op = StackedTensordotGeneral.new(sc)
    else
      raise Num::Exceptions::ValueError.new("Bad inputs")
    end
    op.contract(lhs, rhs)
  end

  private def einsum_dispatcher(input_string, operands : Array(Tensor(U, CPU(U)))) forall U
    co = validate_and_optimize_order(input_string, operands)
    case co.ctype
    when ContractionOrderType::Singleton
      single_einsum_dispatch(co.item[0].sized_contraction, operands[0])
    else
      intermediate_results = [] of Tensor(U, CPU(U))
      steps = co.item.unsafe_as(Array(Pair))
      steps.each do |i|
        lhs_t = i.operand_nums.lhs
        case lhs_t.flag
        when OperandType::Input
          lhs = operands[lhs_t.value]
        else
          lhs = intermediate_results[lhs_t.value]
        end

        rhs_t = i.operand_nums.rhs
        case rhs_t.flag
        when OperandType::Input
          rhs = operands[rhs_t.value]
        else
          rhs = intermediate_results[rhs_t.value]
        end
        ir = multiple_einsum_dispatch(i.sized_contraction, lhs, rhs)
        intermediate_results << ir
      end
      intermediate_results.pop
    end
  end

  def einsum(input_string : String, *operands : Tensor(U, CPU(U))) forall U
    einsum_dispatcher(input_string, operands.to_a)
  end
end
