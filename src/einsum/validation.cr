require "./optimizers"

module Num::Einsum

  struct EinsumParse
    getter operand_indices : Array(String)
    getter output_indices : String

    def initialize(@operand_indices : Array(String), @output_indices : String)
    end
  end

  alias OutputSize = Hash(Char, Int32)

  struct Contraction
    getter operand_indices : Array(Array(Char))
    getter output_indices : Array(Char)
    getter summation_indices : Array(Char)

    def initialize(@operand_indices, @output_indices, @summation_indices)
    end

    def self.from_parse(inp, outp)
      operand_indices = inp.map { |e| e.each_char.to_a }
      if outp.size > 0
        requested_output_indices = outp
      else
        input_indices = Hash(Char, Int32).new
        inp.join().each_char do |c|
          if input_indices.has_key?(c)
            input_indices[c] += 1
          else
            input_indices[c] = 1
          end
        end
        unique_indices = input_indices.select { |k, v| v == 1 }.keys
        requested_output_indices = unique_indices.sort.join()
      end
      from_indices(operand_indices, requested_output_indices)
    end

    def self.from_indices(operand_indices, output_indices)
      input_char_counts = Hash(Char, Int32).new
      operand_indices.flatten.each do |c|
        input_char_counts[c] = 1
      end

      distinct_output_indices = Hash(Char, Int32).new

      if output_indices.is_a?(Array(Char))
        output_indices = output_indices.join()
      end

      output_indices.each_char do |c|
        distinct_output_indices[c] = distinct_output_indices.fetch(c, 0) + 1
      end

      distinct_output_indices.each do |c, n|
        if n > 1
          raise "Requested output has duplicate index"
        end

        if !input_char_counts.has_key?(c)
          raise "Output contains an index not found in inputs"
        end
      end

      summation_indices = input_char_counts.select { |k, v| !distinct_output_indices.has_key?(k) }.keys
      new(operand_indices, output_indices.each_char.to_a, summation_indices)
    end
  end

  struct SizedContraction
    getter contraction : Contraction
    getter output_size : OutputSize

    def initialize(@contraction, @output_size)
    end

    def self.from_contraction_and_shapes(contraction : Contraction, shapes : Array(Array(Int32)))
      if contraction.operand_indices.size != shapes.size
        raise "Number of operands in contraction deos not match number of operands supplied"
      end

      index_lengths = OutputSize.new

      contraction.operand_indices.zip(shapes) do |indices, operand_shape|
        if indices.size != operand_shape.size
          raise "Number of indices in one or more operands does not match dimension of operand"
        end

        indices.zip(operand_shape) do |c, n|
          existing_n = index_lengths.fetch(c, nil)
          if existing_n.nil?
            index_lengths[c] = n
            existing_n = n
          end
          if existing_n != n
            raise "Repeated index with different size"
          end
        end
      end
      new(contraction, index_lengths)
    end

    def self.from_contraction_and_operands(contraction : Contraction, operands : Array(Tensor))
      shapes = operands.map { |e| e.shape }
      from_contraction_and_shapes(contraction, shapes)
    end

    def subset(new_operand_indices : Array(Array(Char)), new_output_indices : Array(Char))
      all_operand_indices = new_operand_indices.flatten.to_set

      if all_operand_indices.any? { |c| !output_size.has_key?(c) }
        raise "Character found in new operand indices but not in output size"
      end

      new_contraction = Contraction.from_indices(new_operand_indices, new_output_indices)
      new_output_size = output_size.select { |k, v| all_operand_indices.includes?(k) }
      SizedContraction.new(new_contraction, new_output_size)
    end
  end

  def parse_einsum_string(input_string : String)
    parsed = /^(?P<first_operand>[a-z]+)(?P<more_operands>(?:,[a-z]+)*)(?:->(?P<output>[a-z]*))?$/.match(input_string)
    begin
      output = parsed.try &.["output"]
      output_indices = output.nil? ? "" : output
    rescue
      output_indices = ""
    end
    operand_indices = [] of String
    first = parsed.try &.["first_operand"]
    if first.nil?
      raise "Bad input"
    else
      operand_indices << first
    end
    rest = parsed.try &.["more_operands"]
    if !rest.nil?
      operand_indices += rest.split(",")[1...]
    end
    EinsumParse.new(operand_indices, output_indices)
  end

  def validate(input_string : String)
    ep = parse_einsum_string(input_string)
    Contraction.from_parse(ep.operand_indices, ep.output_indices)
  end

  def get_operand_shapes(operands : Array(Tensor))
    operands.map { |e| e.shape }
  end

  def validate_and_size(input_string : String, *operands : Tensor)
    contraction = validate(input_string)
    SizedContraction.from_contraction_and_operands(contraction, operands.to_a)
  end

  def validate_and_optimize_order(input_string : String, *operands : Tensor)
    sc = validate_and_size(input_string, *operands)
    generate_optimized_order(sc)
  end
end
