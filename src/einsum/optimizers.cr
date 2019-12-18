require "./validation"

module Num::Einsum
  enum OperandType
    Input
    Intermediate
  end

  # Either an input operand or an intermediate result
  struct OperandNumber
    getter flag : OperandType
    getter value : Int32

    def initialize(@flag : OperandType, @value : Int32)
    end
  end

  # Which two tensors to contract
  struct OperandNumPair
    getter lhs : OperandNumber
    getter rhs : OperandNumber

    def initialize(@lhs : OperandNumber, @rhs : OperandNumber)
    end
  end

  # A single pairwise contraction between two input operands, an input operand and an intermediate
  # result, or two intermediate results.
  struct Pair
    getter sized_contraction : SizedContraction
    getter operand_nums : OperandNumPair

    def initialize(@sized_contraction : SizedContraction, @operand_nums : OperandNumPair)
    end
  end

  struct Singleton
    getter sized_contraction : SizedContraction

    def initialize(@sized_contraction : SizedContraction)
    end
  end

  enum ContractionOrderType
    Singleton
    Pair
  end

  struct ContractionOrder(T)
    getter ctype : ContractionOrderType
    getter item : Array(T)

    private def initialize(@item : Array(T), @ctype)
    end

    def self.from_singleton(singleton : T)
      new([singleton], ContractionOrderType::Singleton)
    end

    def self.from_pairs(pairs : Array(T))
      new(pairs, ContractionOrderType::Pair)
    end
  end

  enum OptimizationMethod
    Naive
    Greedy
    Optimal
    Branch
  end

  def get_remaining_operand_indices(operand_indices : Array(Char), output_indices : Array(Char))
    (operand_indices + output_indices).to_set
  end

  def get_existing_indices(lhs_indices : Array(Char), rhs_indices : Array(Char))
    (lhs_indices + rhs_indices).to_set
  end

  def generate_permutated_contraction(sized_contraction : SizedContraction, tensor_order : Array(Int32))
    new_operand_indices = tensor_order.map { |i| sized_contraction.contraction.operand_indices[i] }
    sized_contraction.subset(new_operand_indices, sized_contraction.contraction.output_indices)
  end

  def generate_sized_contraction_pair(lhs_operand_indices : Array(Char), rhs_operand_indices : Array(Char), output_indices : Array(Char), orig_contraction : SizedContraction)
    orig_contraction.subset([lhs_operand_indices, rhs_operand_indices], output_indices)
  end

  def generate_path(sized_contraction : SizedContraction, tensor_order : Array(Int32))
    permuted_contraction = generate_permutated_contraction(sized_contraction, tensor_order)

    case permuted_contraction.contraction.operand_indices.size
    when 1
      singleton = Singleton.new(sized_contraction)
      ContractionOrder.from_singleton(singleton)
    when 2
      sc = generate_sized_contraction_pair(
        permuted_contraction.contraction.operand_indices[0],
        permuted_contraction.contraction.operand_indices[1],
        permuted_contraction.contraction.output_indices,
        permuted_contraction,
      )

      lhs = OperandNumber.new(OperandType::Input, tensor_order[0])
      rhs = OperandNumber.new(OperandType::Input, tensor_order[1])
      operand_num_pair = OperandNumPair.new(lhs, rhs)
      only_step = Pair.new(sized_contraction: sc, operand_nums: operand_num_pair)
      ContractionOrder.from_pairs([only_step])
    else
      steps = [] of Pair
      output_indices = permuted_contraction.contraction.operand_indices[0].dup

      (0...(permuted_contraction.contraction.operand_indices[0].size - 1)).each do |idx_of_lhs|
        lhs_indices = output_indices.dup
        idx_of_rhs = idx_of_lhs + 1
        rhs_indices = permuted_contraction.contraction.operand_indices[idx_of_rhs]

        if idx_of_rhs == (permuted_contraction.contraction.operand_indices.size - 1)
          output_indices = permuted_contraction.contraction.output_indices.dup
        else
          existing_indices = get_existing_indices(lhs_indices, rhs_indices)
          remaining_indices = get_remaining_operand_indices(
            permuted_contraction.contraction.operand_indices[(idx_of_rhs + 1)...].flatten,
            permuted_contraction.contraction.output_indices,
          )
          output_indices = (existing_indices & remaining_indices).to_a
        end

        sc = generate_sized_contraction_pair(lhs_indices, rhs_indices, output_indices, permuted_contraction)

        if idx_of_lhs == 0
          lhs = OperandNumber.new(OperandType::Input, tensor_order[idx_of_lhs])
          rhs = OperandNumber.new(OperandType::Input, tensor_order[idx_of_rhs])
          operand_nums = OperandNumPair.new(lhs, rhs)
        else
          lhs = OperandNumber.new(OperandType::Intermediate, idx_of_lhs - 1)
          rhs = OperandNumber.new(OperandType::Input, tensor_order[idx_of_rhs])
          operand_nums = OperandNumPair.new(lhs, rhs)
        end

        steps << Pair.new(sized_contraction: sc, operand_nums: operand_nums)
      end

      ContractionOrder.from_pairs(steps)
    end
  end

  def naive_order(sized_contraction : SizedContraction)
    (0...sized_contraction.contraction.operand_indices.size).to_a
  end

  def generate_optimized_order(sized_contraction : SizedContraction, strategy : OptimizationMethod = OptimizationMethod::Naive)
    case strategy
    when OptimizationMethod::Naive
      tensor_order = naive_order(sized_contraction)
    else
      raise "Unsupported Optimization Method. PRs Welcome :)"
    end
    generate_path(sized_contraction, tensor_order)
  end
end
