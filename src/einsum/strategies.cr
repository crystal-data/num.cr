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
enum Num::Einsum::SingletonMethod
  Identity
  Permutation
  Summation
  Diagonalization
  PermutationAndSummation
  DiagonalizationAndSummation
end

# :nodoc:
struct Num::Einsum::SingletonSummary
  getter num_summed_axes : Int32
  getter num_diagonalized_axes : Int32
  getter num_reordered_axes : Int32

  def initialize(sc : Num::Einsum::SizedContraction)
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
      Num::Einsum::SingletonMethod::Identity
    when {0, 0, _}
      Num::Einsum::SingletonMethod::Permutation
    when {_, 0, 0}
      Num::Einsum::SingletonMethod::Summation
    when {0, _, _}
      Num::Einsum::SingletonMethod::Diagonalization
    when {_, 0, _}
      Num::Einsum::SingletonMethod::PermutationAndSummation
    else
      Num::Einsum::SingletonMethod::DiagonalizationAndSummation
    end
  end
end

# :nodoc:
enum Num::Einsum::PairMethod
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

# :nodoc:
struct Num::Einsum::PairSummary
  @num_stacked_axes : Int32
  @num_lhs_outer_axes : Int32
  @num_rhs_outer_axes : Int32
  @num_contracted_axes : Int32

  def initialize(sc : Num::Einsum::SizedContraction)
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
      Num::Einsum::PairMethod::HadamardProductGeneral
    when {0, 0, _, 0}
      Num::Einsum::PairMethod::ScalarMatrixProductGeneral
    when {0, _, 0, 0}
      Num::Einsum::PairMethod::MatrixScalarProductGeneral
    when {_, _, _, 0}
      Num::Einsum::PairMethod::TensordotGeneral
    else
      Num::Einsum::PairMethod::StackedTensordotGeneral
    end
  end
end

# :nodoc:
struct Num::Einsum::EinsumPath(T)
  getter order : T

  def initialize(@order : T)
  end

  def self.new(input_string : String, operands : Array(Tensor(U, CPU(U)))) forall U
    new(
      Num::Einsum.validate_and_optimize_order(input_string, operands)
    )
  end

  def contract_operands(operands : Array(Tensor(U, CPU(U)))) : Tensor(U, CPU(U)) forall U
    case @order.ctype
    when ContractionOrderType::Singleton
      sized_contraction = @order.item.first.sized_contraction
      contraction = Num::Einsum::SingletonContraction.new(sized_contraction)
      contraction.contract(operands[0])
    when Num::Einsum::ContractionOrderType::Pair
      buffer = [] of Tensor(U, CPU(U))
      steps = @order.item.unsafe_as(Array(Num::Einsum::Pair))
      steps.each do |step|
        lhs_info = step.operand_nums.lhs
        rhs_info = step.operand_nums.rhs

        lhs = case lhs_info.flag
              when Num::Einsum::OperandType::Input
                operands[lhs_info.value]
              else
                buffer[lhs_info.value]
              end

        rhs = case rhs_info.flag
              when Num::Einsum::OperandType::Input
                operands[rhs_info.value]
              else
                buffer[rhs_info.value]
              end

        contraction = Num::Einsum::PairContraction.new(step.sized_contraction)
        buffer << contraction.contract(lhs, rhs)
      end
      buffer.pop
    else
      raise Num::Exceptions::ValueError.new("InvalidContraction OrderType")
    end
  end
end

module Num::Einsum
  # Evaluates the Einstein summation convention on the operands.
  #
  # The Einstein summation convention can be used to compute many
  # multi-dimensional, linear algebraic array operations. einsum provides a
  # succinct way of representing these.
  #
  # A non-exhaustive list of these operations, which can be computed by
  # einsum, is shown below along with examples:
  #
  #     Trace of an array
  #     Return a diagonal
  #     Array axis summations
  #     Transpositions and permutations
  #     Matrix multiplication and dot product
  #     Vector inner and outer products
  #     Broadcasting, element-wise and scalar multiplication
  #     Tensor contractions
  #
  # The subscripts string is a comma-separated list of subscript labels,
  # where each label refers to a dimension of the corresponding operand.
  # Whenever a label is repeated it is summed, so
  # `Num::Einsum.einsum("i,i", a, b)` is equivalent to an inner operation.
  # If a label appears only once, it is not summed, so
  # `Num::Einsum.einsum("i", a)` produces a view of a with no changes.
  # A further example `Num::Einsum.einsum("ij,jk", a, b)` describes traditional
  # matrix multiplication and is equivalent to a.matmul(b). Repeated
  # subscript labels in one operand take the diagonal. For example,
  # `Num::Einsum.einsum("ii", a)` gets the trace of a matrix
  def einsum(input_string : String, *operands : Tensor(U, CPU(U))) forall U
    einsum(input_string, operands.to_a)
  end

  # :ditto:
  def einsum(input_string : String, operands : Array(Tensor(U, CPU(U)))) forall U
    path = Num::Einsum::EinsumPath.new(input_string, operands)
    path.contract_operands(operands.to_a)
  end

  # :nodoc:
  def einsum_path(input_string : String, *operands : Tensor(U, CPU(U))) forall U
    einsum_path(input_string, operands.to_a)
  end

  # :nodoc:
  def einsum_path(input_string : String, operands : Array(Tensor(U, CPU(U)))) forall U
    Num::Einsum::EinsumPath.new(input_string, operands)
  end
end
