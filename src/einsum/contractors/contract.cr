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
# Holds a `Box`ed `SingletonContractor` trait object.
# Constructed at runtime based on the number of diagonalized, summed, and permuted axes
# in the input. Reimplements the `SingletonContractor` trait by delegating to the inner
# object.
#
# For example, the contraction `iij->i` will be performed by assigning a `Box`ed
# `DiagonalizationAndSummation` to `op`. The contraction `ijk->kij` will be performed
# by assigning a `Box`ed `Permutation` to `op`.
struct Num::Einsum::SingletonContraction(T)
  getter method : Num::Einsum::SingletonMethod
  getter op : T

  def initialize(@method : Num::Einsum::SingletonMethod, @op : T)
  end

  def self.new(sc : Num::Einsum::SizedContraction)
    singleton_summary = Num::Einsum::SingletonSummary.new(sc)
    method = singleton_summary.get_strategy

    op = case method
         when SingletonMethod::Identity
           Identity.new
         when SingletonMethod::Permutation
           Permutation.new(sc)
         when SingletonMethod::Summation
           Summation.new(sc)
         when SingletonMethod::Diagonalization
           Diagonalization.new(sc)
         when SingletonMethod::PermutationAndSummation
           PermutationAndSummation.new(sc)
         when SingletonMethod::DiagonalizationAndSummation
           DiagonalizationAndSummation.new(sc)
         else
           raise Num::Exceptions::ValueError.new("Invalid singleton method")
         end

    new(method, op)
  end

  delegate contract, to: @op
end

# :nodoc:
# Holds a `SingletonContractor` and the resulting simplified indices.
struct Num::Einsum::SimplificationMethodAndOutput(T)
  getter method : Num::Einsum::SingletonMethod
  getter op : T
  getter new_indices : Array(Char)
  getter einsum_string : String

  def initialize(
    @method : Num::Einsum::SingletonMethod,
    @op : T,
    @new_indices : Array(Char),
    @einsum_string : String
  )
  end

  # Based on the number of diagonalized, permuted, and summed axes, chooses a struct implementing
  # `SingletonContractor` to simplify the tensor (or `None` if the tensor doesn't need simplification)
  # and computes the indices of the simplified tensor.
  # :nodoc:
  def self.from_indices_and_sizes(
    this_input_indices : Array(Char),
    other_input_indices : Array(Char),
    output_indices : Array(Char),
    orig_contraction : Num::Einsum::SizedContraction
  )
    this_input_uniques = this_input_indices.to_set
    other_input_uniques = other_input_indices.to_set
    output_uniques = output_indices.to_set

    other_and_output = other_input_uniques | output_uniques
    desired_uniques = this_input_uniques & other_and_output

    new_indices = desired_uniques.to_a

    simplification_sc = orig_contraction.subset(
      [this_input_indices], new_indices
    )

    sc = Num::Einsum::SingletonContraction.new(simplification_sc)

    case sc.method
    when Num::Einsum::SingletonMethod::Identity, Num::Einsum::SingletonMethod::Permutation
      nil
    else
      new(sc.method, sc.op, new_indices, simplification_sc.as_einsum_string)
    end
  end
end

# :nodoc:
# Holds a `Box`ed `PairContractor` trait object and two `Optional` simplifications for the LHS and RHS tensors.
# # For example, the contraction `ijk,kj->jk` will currently be performed as follows:
# # 1. Simplify the LHS with the contraction `ijk->jk`
# 2. Don't simplify the RHS
# 3. Use HadamardProductGeneral to compute `jk,kj->jk`
# # A second example is the contraction `iij,jkk->ik`:
# # 1. Simplify the LHS with the contraction `iij->ij`
# 2. Simplify the RHS with the contraction `jkk->jk`
# 3. Use TensordotGeneral to compute `ij,jk->ik`
# # Since the axis lengths aren't known until runtime, and the actual einsum string may not
# be either, it is generally not possible to know at compile time which specific PairContractor
# will be used to perform a given contraction, or even which contractions will be performed;
# the optimizer could choose a different order.
struct Num::Einsum::PairContraction(T, U)
  getter lhs_simplification : Num::Einsum::SimplificationMethodAndOutput(U)?
  getter rhs_simplification : Num::Einsum::SimplificationMethodAndOutput(U)?
  getter method : Num::Einsum::PairMethod
  getter op : T
  getter simplified_einsum_string : String

  def initialize(
    @lhs_simplification : Num::Einsum::SimplificationMethodAndOutput(U)?,
    @rhs_simplification : Num::Einsum::SimplificationMethodAndOutput(U)?,
    @method : Num::Einsum::PairMethod,
    @op : T,
    @simplified_einsum_string : String
  )
  end

  def self.new(sc : Num::Einsum::SizedContraction)
    lhs_indices = sc.contraction.operand_indices[0]
    rhs_indices = sc.contraction.operand_indices[1]
    output_indices = sc.contraction.output_indices

    lhs_simplification = \
       Num::Einsum::SimplificationMethodAndOutput.from_indices_and_sizes(
        lhs_indices,
        rhs_indices,
        output_indices,
        sc
      )

    rhs_simplification = \
       Num::Einsum::SimplificationMethodAndOutput.from_indices_and_sizes(
        rhs_indices,
        lhs_indices,
        output_indices,
        sc
      )

    new_lhs_indices = case lhs_simplification
                      in Num::Einsum::SimplificationMethodAndOutput
                        lhs_simplification.new_indices.dup
                      in nil
                        lhs_indices.dup
                      end

    new_rhs_indices = case rhs_simplification
                      in Num::Einsum::SimplificationMethodAndOutput
                        rhs_simplification.new_indices.dup
                      in nil
                        rhs_indices.dup
                      end

    reduced_sc = sc.subset([new_lhs_indices, new_rhs_indices], output_indices)

    pair_summary = Num::Einsum::PairSummary.new(reduced_sc)
    method = pair_summary.get_strategy

    op = case method
         when Num::Einsum::PairMethod::HadamardProductGeneral
           Num::Einsum::HadamardProductGeneral.new(reduced_sc)
         when Num::Einsum::PairMethod::ScalarMatrixProductGeneral
           Num::Einsum::ScalarMatrixProductGeneral.new(reduced_sc)
         when Num::Einsum::PairMethod::MatrixScalarProductGeneral
           Num::Einsum::MatrixScalarProductGeneral.new(reduced_sc)
         when Num::Einsum::PairMethod::TensordotGeneral
           Num::Einsum::TensordotGeneral.new(reduced_sc)
         when Num::Einsum::PairMethod::StackedTensordotGeneral
           Num::Einsum::StackedTensordotGeneral.new(reduced_sc)
         else
           raise Num::Exceptions::ValueError.new("Invalid PairMethod")
         end

    new(
      lhs_simplification,
      rhs_simplification,
      method,
      op,
      simplified_einsum_string: reduced_sc.as_einsum_string
    )
  end

  def contract(
    lhs : Tensor(U, CPU(U)),
    rhs : Tensor(U, CPU(U))
  ) : Tensor(U, CPU(U)) forall U
    case {@lhs_simplification, @rhs_simplification}
    when {nil, nil}
      @op.contract(lhs, rhs)
    when {_, nil}
      @op.contract(@lhs_simplification.not_nil!.op.contract(lhs), rhs)
    when {nil, _}
      @op.contract(lhs, @rhs_simplification.not_nil!.op.contract(rhs))
    else
      @op.contract(
        @lhs_simplification.not_nil!.op.contract(lhs),
        @rhs_simplification.not_nil!.op.contract(rhs)
      )
    end
  end
end
