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
enum Num::Einsum::OperandType
  Input
  Intermediate
end

# :nodoc:
# Either an input operand or an intermediate result
struct Num::Einsum::OperandNumber
  getter flag : OperandType
  getter value : Int32

  def initialize(@flag : OperandType, @value : Int32)
  end
end

# :nodoc:
# Which two tensors to contract
struct Num::Einsum::OperandNumPair
  getter lhs : OperandNumber
  getter rhs : OperandNumber

  def initialize(@lhs : OperandNumber, @rhs : OperandNumber)
  end
end

# :nodoc:
# A single pairwise contraction between two input operands, an input
# operand and an intermediate result, or two intermediate results.
struct Num::Einsum::Pair
  # The contraction to be performed
  getter sized_contraction : SizedContraction

  # Which two tensors to contract
  getter operand_nums : OperandNumPair

  def initialize(
    @sized_contraction : SizedContraction,
    @operand_nums : OperandNumPair
  )
  end
end

# :nodoc:
struct Num::Einsum::Singleton
  getter sized_contraction : SizedContraction

  def initialize(@sized_contraction : SizedContraction)
  end
end

# :nodoc:
enum Num::Einsum::ContractionOrderType
  Singleton
  Pair
end

# :nodoc:
# The order in which to contract pairs of tensors and the specific
# contractions to be performed between the pairs.
#
# Either a singleton contraction, in the case of a single input operand,
# or a list of pair contractions, given two or more input operands
struct Num::Einsum::ContractionOrder(T)
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

# :nodoc:
# Strategy for optimizing the contraction. The only currently supported
# options are "Naive" and "Reverse".
enum Num::Einsum::OptimizationMethod
  # Contracts each pair of tensors in the order given in the input and
  # uses the intermediate result as the LHS of the next contraction.
  Naive

  # Contracts each pair of tensors in the reverse of the order given in the input and uses the
  # intermediate result as the LHS of the next contraction. Only implemented to help test
  # that this is actually functioning properly.
  Reverse
end

module Num::Einsum
  extend self

  # :nodoc:
  # Returns a set of all the indices in any of the remaining operands
  # or in the output
  def get_remaining_operand_indices(
    operand_indices : Array(Char),
    output_indices : Array(Char)
  ) : Set(Char)
    operand_indices.to_set + output_indices.to_set
  end

  # :nodoc:
  # Returns a set of all the indices in the LHS or the RHS
  def get_existing_indices(lhs_indices : Array(Char), rhs_indices : Array(Char))
    lhs_indices.to_set + rhs_indices.to_set
  end

  # :nodoc:
  # Returns a permuted version of `sized_contraction`, specified by
  # `tensor_order`
  def generate_permutated_contraction(
    sized_contraction : Num::Einsum::SizedContraction,
    order : Array(Int32)
  ) : Num::Einsum::SizedContraction
    indices = order.map do |i|
      sized_contraction.contraction.operand_indices[i]
    end
    sized_contraction.subset(
      indices,
      sized_contraction.contraction.output_indices
    )
  end

  # :nodoc:
  # Generates a mini-contraction corresponding to `lhs_operand_indices`,
  # `rhs_operand_indices`->`output_indices`
  def generate_sized_contraction_pair(
    lhs_operand_indices : Array(Char),
    rhs_operand_indices : Array(Char),
    output_indices : Array(Char),
    original_contraction : Num::Einsum::SizedContraction
  )
    original_contraction.subset(
      [lhs_operand_indices, rhs_operand_indices], output_indices
    )
  end

  # :nodoc:
  # Generate the actual path consisting of all the mini-contractions. Currently always
  # contracts two input operands and then repeatedly uses the result as the LHS of the
  # next pairwise contraction.
  def generate_path(
    sized_contraction : Num::Einsum::SizedContraction,
    tensor_order : Array(Int32)
  )
    permuted_contraction = generate_permutated_contraction(
      sized_contraction, tensor_order
    )

    case permuted_contraction.contraction.operand_indices.size
    when 1
      singleton = Num::Einsum::Singleton.new(sized_contraction)
      Num::Einsum::ContractionOrder.from_singleton(singleton)
    when 2
      sc = generate_sized_contraction_pair(
        permuted_contraction.contraction.operand_indices[0],
        permuted_contraction.contraction.operand_indices[1],
        permuted_contraction.contraction.output_indices,
        permuted_contraction,
      )

      lhs = Num::Einsum::OperandNumber.new(OperandType::Input, tensor_order[0])
      rhs = Num::Einsum::OperandNumber.new(OperandType::Input, tensor_order[1])
      operand_num_pair = OperandNumPair.new(lhs, rhs)
      only_step = Num::Einsum::Pair.new(
        sized_contraction: sc, operand_nums: operand_num_pair
      )
      Num::Einsum::ContractionOrder.from_pairs([only_step])
    else
      steps = [] of Num::Einsum::Pair
      output_indices = permuted_contraction.contraction.operand_indices[0].dup
      n = permuted_contraction.contraction.operand_indices.size - 1

      (0...n).each do |idx_of_lhs|
        lhs_indices = output_indices.dup
        idx_of_rhs = idx_of_lhs + 1
        rhs_indices = permuted_contraction.contraction.operand_indices[
          idx_of_rhs,
        ]

        if idx_of_rhs == n
          output_indices = permuted_contraction.contraction.output_indices.dup
        else
          existing_indices = get_existing_indices(lhs_indices, rhs_indices)
          remaining_indices = get_remaining_operand_indices(
            permuted_contraction.contraction.operand_indices[
              (idx_of_rhs + 1)...,
            ].flatten,
            permuted_contraction.contraction.output_indices,
          )
          output_indices = (existing_indices & remaining_indices).to_a
        end

        sc = generate_sized_contraction_pair(
          lhs_indices, rhs_indices, output_indices, permuted_contraction
        )

        if idx_of_lhs == 0
          lhs = Num::Einsum::OperandNumber.new(
            Num::Einsum::OperandType::Input, tensor_order[idx_of_lhs]
          )
          rhs = Num::Einsum::OperandNumber.new(
            Num::Einsum::OperandType::Input, tensor_order[idx_of_rhs]
          )
          operand_nums = OperandNumPair.new(lhs, rhs)
        else
          lhs = Num::Einsum::OperandNumber.new(
            Num::Einsum::OperandType::Intermediate, idx_of_lhs - 1
          )
          rhs = Num::Einsum::OperandNumber.new(
            Num::Einsum::OperandType::Input, tensor_order[idx_of_rhs]
          )
          operand_nums = Num::Einsum::OperandNumPair.new(lhs, rhs)
        end

        steps << Num::Einsum::Pair.new(
          sized_contraction: sc, operand_nums: operand_nums
        )
      end

      Num::Einsum::ContractionOrder.from_pairs(steps)
    end
  end

  # :nodoc:
  # Contracts the first two operands, then contracts the result with the
  # third operand, etc.
  def naive_order(sized_contraction : Num::Einsum::SizedContraction)
    (0...sized_contraction.contraction.operand_indices.size).to_a
  end

  # :nodoc:
  # Contracts the last two operands, then contracts the result with the
  # third-to-last operand, etc.
  def reverse_order(sized_contraction : Num::Einsum::SizedContraction)
    naive_order(sized_contraction).reverse
  end

  # :nodoc:
  # Given a `SizedContraction` and an optimization strategy, returns an
  # order in which to perform pairwise contractions in order to produce
  # the final result
  def generate_optimized_order(
    sized_contraction : Num::Einsum::SizedContraction,
    strategy : Num::Einsum::OptimizationMethod = Num::Einsum::OptimizationMethod::Naive
  )
    case strategy
    when Num::Einsum::OptimizationMethod::Naive
      tensor_order = naive_order(sized_contraction)
    when Num::Einsum::OptimizationMethod::Reverse
      tensor_order = naive_order(sized_contraction)
    else
      raise Num::Exceptions::ValueError.new(
        "Invalid optimization type"
      )
    end
    generate_path(sized_contraction, tensor_order)
  end
end
