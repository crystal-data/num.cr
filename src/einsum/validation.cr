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

struct Num::Einsum::Parse
  getter operand_indices : Array(String)
  getter output_indices : String?

  def initialize(@operand_indices : Array(String), @output_indices : String?)
  end
end

struct Num::Einsum::Contraction
  # An array with as many elements as input operands, where each
  # member of the array is an Array(Char) with each char representing the label for
  # each axis of the operand.
  getter operand_indices : Array(Array(Char))

  # Specifies which axes the resulting tensor will contain
  # (corresponding to axes in one or more of the input operands).
  getter output_indices : Array(Char)

  # Contains the axes that will be summed over (a.k.a contracted) by the operation.
  getter summation_indices : Array(Char)

  private def initialize(
    @operand_indices : Array(Array(Char)),
    @output_indices : Array(Char),
    @summation_indices : Array(Char)
  )
  end

  # Validates and creates a `Contraction` from an `einsum`-formatted string.
  def self.new(input_string : String) : Num::Einsum::Contraction
    parse = Num::Einsum.parse_einsum_string(input_string)
    from_parse(parse)
  end

  # If output_indices has been specified in the parse (i.e. explicit case),
  # e.g. "ij,jk->ik", simply converts the strings to Array(Char) and passes
  # them to Contraction.from_indices. If the output indices haven't been specified,
  # e.g. "ij,jk", figures out which ones aren't duplicated and hence summed over,
  # sorts them alphabetically, and uses those as the output indices.
  def self.from_parse(parse : Num::Einsum::Parse) : Num::Einsum::Contraction
    if parse.output_indices.nil?
      input_indices = Hash(Char, Int32).new(0)
      parse.operand_indices.join("").each_char do |c|
        input_indices[c] += 1
      end
      output_indices = input_indices.to_a.select do |_, v|
        v == 1
      end.map { |k, _| k }
    else
      output_indices = parse.output_indices.not_nil!.chars
    end
    operand_indices = parse.operand_indices.map { |el| el.chars }
    self.from_indices(operand_indices, output_indices)
  end

  # Validates and creates a `Contraction` from an Array(Array(Char)) containing
  # the operand indices, and a slice of `char` containing the desired output indices.
  def self.from_indices(
    operand_indices : Array(Array(Char)),
    output_indices : Array(Char)
  ) : Num::Einsum::Contraction
    input_char_counts = Hash(Char, Int32).new(0)
    operand_indices.flatten.each do |c|
      input_char_counts[c] += 1
    end

    distinct_output_indices = Hash(Char, Int32).new(0)
    output_indices.each do |c|
      distinct_output_indices[c] += 1
    end

    distinct_output_indices.each do |k, v|
      if v > 1
        raise Num::Exceptions::ValueError.new(
          "Requested output has duplicate index"
        )
      end

      if input_char_counts[k]?.nil?
        raise Num::Exceptions::ValueError.new(
          "Requested output contains an index not found in inputs"
        )
      end
    end

    summation_indices = input_char_counts.keys.select do |c|
      distinct_output_indices[c]?.nil?
    end
    summation_indices.sort!

    new(operand_indices.dup, output_indices.dup, summation_indices)
  end
end

# Alias for `Hash(Char, Int32)`. Contains the axis lengths for all indices
# in the contraction.
#
# Contrary to the name, does not only hold the sizes for output indices.
class Num::Einsum::OutputSize < Hash(Char, Int32)
  # Build the Hash containing the axis lengths
  def self.from_contraction_and_shapes(
    contraction : Num::Einsum::Contraction,
    operand_shapes : Array(Array(Int32))
  ) : Num::Einsum::OutputSize
    unless contraction.operand_indices.size == operand_shapes.size
      raise Num::Exceptions::ValueError.new(
        "Number of operands in contraction does not match number of " \
        "operands supplied"
      )
    end
    index_lengths = Num::Einsum::OutputSize.new(0)
    contraction.operand_indices.zip(operand_shapes) do |indices, operand_shape|
      unless indices.size == operand_shape.size
        raise Num::Exceptions::ValueError.new(
          "Number of indices in one or more operands not match " \
          "dimensions of operand"
        )
      end
      indices.zip(operand_shape) do |c, n|
        existing_n = index_lengths.fetch(c, n)
        unless existing_n == n
          raise Num::Exceptions::ValueError.new(
            "Repeated index with different size"
          )
        end
        index_lengths[c] = n
      end
    end
    index_lengths
  end
end

# A `SizedContraction` contains a `Contraction` as well as a `Hash(Char, Int32)`
# specifying the axis lengths for each index in the contraction.
#
# Note that output_size is a misnomer (to be changed); it contains all the axis lengths,
# including the ones that will be contracted (i.e. not just the ones in
# contraction.output_indices).
struct Num::Einsum::SizedContraction
  getter contraction : Num::Einsum::Contraction
  getter output_size : Num::Einsum::OutputSize

  def initialize(
    @contraction : Num::Einsum::Contraction,
    @output_size : Num::Einsum::OutputSize
  )
  end

  # Creates a new SizedContraction based on a subset of the operand indices and/or output
  # indices. Not intended for general use; used internally in the library when compiling
  # a multi-tensor contraction into a set of tensor simplifications (a.k.a. singleton
  # contractions) and pairwise contractions.
  def subset(
    new_operand_indices : Array(Array(Char)),
    new_output_indices : Array(Char)
  ) : Num::Einsum::SizedContraction
    all_operand_indices = new_operand_indices.flatten.to_set
    if all_operand_indices.any? { |c| @output_size[c]?.nil? }
      raise Num::Exceptions::ValueError.new(
        "Character found in new_operand_indices but not in output_size"
      )
    end

    new_contraction = Num::Einsum::Contraction.from_indices(
      new_operand_indices, new_output_indices
    )
    new_output_size = Num::Einsum::OutputSize.new

    @output_size.each do |k, v|
      if all_operand_indices.includes?(k)
        new_output_size[k] = v
      end
    end

    self.class.new(new_contraction, new_output_size)
  end

  # Create a SizedContraction from an already-created Contraction and a list
  # of shapes.
  def self.from_contraction_and_shapes(
    contraction : Num::Einsum::Contraction,
    operand_shapes : Array(Array(Int32))
  ) : Num::Einsum::SizedContraction
    output_size = Num::Einsum::OutputSize.from_contraction_and_shapes(
      contraction, operand_shapes
    )
    new(contraction.dup, output_size)
  end

  # Create a SizedContraction from an already-created Contraction and a list
  # of operands.
  def self.from_contraction_and_operands(
    contraction : Num::Einsum::Contraction,
    operands : Array(Tensor(U, CPU(U)))
  ) : Num::Einsum::SizedContraction forall U
    from_contraction_and_shapes(contraction.dup, operands.map &.shape)
  end

  # Create a SizedContraction from an `einsum`-formatted input string and
  # an array of the shapes of each operand
  def self.from_string_and_shapes(
    input_string : String,
    operand_shapes : Array(Array(Int32))
  ) : Num::Einsum::SizedContraction
    contraction = Num::Einsum::Contraction.new(input_string)
    from_contraction_and_shapes(contraction, operand_shapes)
  end

  # Create a SizedContraction from an `einsum`-formatted input string and a list
  # of operands.
  def self.new(
    input_string : String,
    operands : Array(Tensor(U, CPU(U)))
  ) : Num::Einsum::SizedContraction forall U
    from_string_and_shapes(input_string, operands.map &.shape)
  end

  # Show as an `einsum`-formatted string.
  def as_einsum_string : String
    operands = @contraction.operand_indices.map &.join("")
    s = operands.join(",")
    s += "->"
    s += @contraction.output_indices.join("")
  end
end

module Num::Einsum
  extend self

  # Runs an input string through a regex and convert it to an EinsumParse.
  def parse_einsum_string(input_string : String)
    rgx = /^([a-z]+)((?:,[a-z]+)*)(?:->([a-z]*))?$/
    captures = rgx.match(input_string)
    operand_indices = [] of String
    output = captures.try &.[3]?
    first_operands = captures.try &.[1]?
    other_operands = captures.try &.[2]?

    if first_operands.nil? || other_operands.nil?
      raise Num::Exceptions::ValueError.new("Invalid einsum string")
    end

    operand_indices << first_operands
    other_operands.split(",")[1...].each do |s|
      operand_indices << s
    end

    Num::Einsum::Parse.new(operand_indices, output)
  end

  # Create a SizedContraction and then optimize the order in which pairs of
  # inputs will be contracted.
  def validate_and_optimize_order(input_string : String, operands : Array(Tensor))
    Num::Einsum.generate_optimized_order(
      Num::Einsum::SizedContraction.new(input_string, operands)
    )
  end
end
