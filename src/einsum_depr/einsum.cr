require "../tensor/tensor"

module Num::Einsum
  extend self
  EINSUM_SYMBOLS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
  EINSUM_SYMBOLS_SET = EINSUM_SYMBOLS.each_char.to_set

  private def _parse_einsum_input(operands, *args : Tensor)
    subscripts = operands.gsub(" ", "")
    subscripts.each_char do |c|
      if ".,->".includes?(c)
        next
      end
      if !EINSUM_SYMBOLS_SET.includes?(c)
        raise "Character #{c} is not a valid symbol"
      end
    end

    if subscripts.includes?('-') || subscripts.includes?('>')
      invalid = (subscripts.count('-') > 1) || (subscripts.count('>') > 1)
      if invalid
        raise "Subscripts can only contain one '->'."
      end
    end

    if subscripts.includes?('.')
      used = subscripts.gsub('.', "").gsub(',', "").gsub("->", "")
      unused = (EINSUM_SYMBOLS_SET - used.each_char.to_set).to_a
      ellipse_inds = unused.join()
      longest = 0

      if subscripts.includes?("->")
        input_tmp, output_sub = subscripts.split("->")
        split_subscripts = input_tmp.split(',')
        out_sub = true
      else
        split_subscripts = subscripts.split(',')
        output_sub = ""
        out_sub = false
      end

      split_subscripts.each_with_index do |sub, num|
        if sub.includes?('.')
          if sub.count('.') != 3
            raise "Invalid ellipses"
          end

          if args[num].shape == [] of Int32
            ellipse_count = 0
          else
            ellipse_count = {args[num].ndims, 1}.max
            ellipse_count -= sub.size - 3
          end

          if ellipse_count > longest
            longest = ellipse_count
          end

          if ellipse_count < 0
            raise "Ellipses lengths do not match"
          elsif ellipse_count == 0
            split_subscripts[num] = sub.gsub("...", "")
          else
            rep_inds = ellipse_inds[-ellipse_count...]
            split_subscripts[num] = sub.gsub("...", rep_inds)
          end
        end
      end

      subscripts = split_subscripts.join(",")
      if longest == 0
        out_ellipse = ""
      else
        out_ellipse = ellipse_inds[-longest...]
      end

      if out_sub
        subscripts += "->" + output_sub.gsub("...", out_ellipse)
      else
        output_subscript = ""
        tmp_subscripts = subscripts.gsub(",", "")
        tmp_subscripts.each_char.to_set.to_a.sort.each do |s|
          if !EINSUM_SYMBOLS.includes?(s)
            raise "Character #{s} is not a valid symbol."
          end
          if tmp_subscripts.count(s) == 1
            output_subscript += s
          end
        end

        normal_inds = (output_subscript.each_char.to_set - out_ellipse.each_char.to_set).to_a.sort.join()
        subscripts += "->" + out_ellipse + normal_inds
      end
    end

    if subscripts.includes?("->")
      input_subscripts, output_subscript = subscripts.split("->")
    else
      input_subscripts = subscripts
      tmp_subscripts = subscripts.gsub(",", "")
      output_subscript = ""
      tmp_subscripts.each_char.to_set.to_a.sort.each do |s|
        if !input_subscripts.includes?(s)
          raise "Character #{s} is not a valid symbol"
        end
        if tmp_subscripts.count(s) == 1
          output_subscript += s
        end
      end
    end

    output_subscript.each_char do |char|
      if !input_subscripts.includes?(char)
        raise "Output character #{char} did not appear in the input"
      end
    end

    if input_subscripts.split(',').size != args.size
      raise "Number o feinsum subscripts must be equal to the number of operands."
    end

    return {input_subscripts.split(","), output_subscript.each_char.to_a, args.to_a}
  end

  enum OptimizationMethod
    Naive
    Greedy
    Optimal
  end

  enum CacheType
    Result
    Intermediate
  end

  struct OperandNumber(T)
    getter item : Int32
    def initialize(@item)
    end
  end

  struct Singleton
    getter sized_contraction : SizedContraction
    def initialize(@sized_contraction)
    end
  end

  struct OperandNumPair(T)
    getter lhs : OperandNumber(T)
    getter rhs : OperandNumber(T)
    def initialize(@lhs : OperandNumber(T), @rhs : OperandNumber(T))
    end
  end

  struct Pair(T)
    getter sized_contraction : SizedContraction
    getter operand_nums : OperandNumPair(T)
    def initialize(@sized_contraction, @operand_nums : OperandNumPair(T))
    end
  end

  struct Pairs(T)
    getter items : Array(Pair(T))
    def initialize(@items : Array(Pair(T)))
    end
  end

  struct ContractionOrder(T)
    getter item : T
    def initialize(@item : T)
    end
  end

  def naive_order(sc : SizedContraction)
    (0...sc.contraction.operand_indices.size).to_a
  end

  def generate_permutated_contraction(sized_contraction : SizedContraction, order : Array(Int32))
    new_operand_indices = [] of Array(Char)
    order.each do |i|
      new_operand_indices << sized_contraction.contraction.operand_indices[i].dup
    end
    sized_contraction.subset(new_operand_indices, sized_contraction.contraction.output_indices)
  end

  def generate_sized_contraction_pair(lhs : Array(Char), rhs : Array(Char), outp : Array(Char), orig : SizedContraction)
    orig.subset([lhs, rhs], outp)
  end

  def generate_path(sc : SizedContraction, order : Array(Int32))
    permuted_contraction = generate_permutated_contraction(sc, order)

    case permuted_contraction.contraction.operand_indices.size
    when 1
      ContractionOrder.new(Singleton.new(permuted_contraction))
    when 2
      sc = generate_sized_contraction_pair(
        permuted_contraction.contraction.operand_indices[0],
        permuted_contraction.contraction.operand_indices[1],
        permuted_contraction.contraction.output_indices,
        permuted_contraction
      )
      operand_num_pair = OperandNumPair.new(
        lhs: OperandNumber(CacheType::Result).new(order[0]),
        rhs: OperandNumber(CacheType::Result).new(order[1])
      )
      only_step = Pair.new(
        sized_contraction: sc,
        operand_nums: operand_num_pair
      )
      pairs = Pairs.new([only_step])
      ContractionOrder.new(pairs)
    else
      raise "Only two operands currently supported"
    end
  end

  def generate_optimized_order(sc : SizedContraction, strategy : OptimizationMethod = OptimizationMethod::Naive)
    case strategy
    when OptimizationMethod::Naive
      tensor_order = naive_order(sc)
    else
      raise "#{strategy} is currently not supported"
    end
    generate_path(sc, tensor_order)
  end

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
        requested_output_indices = unique_indices.sort
      end
      from_indices(operand_indices, requested_output_indices)
    end

    def self.from_indices(operand_indices, output_indices)
      input_char_counts = Hash(Char, Int32).new
      operand_indices.flatten.each do |c|
        input_char_counts[c] = 1
      end

      distinct_output_indices = Hash(Char, Int32).new
      output_indices.each do |c|
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
      new(operand_indices, output_indices, summation_indices)
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

  alias OutputSize = Hash(Char, Int32)

  def einsum(input_str, *args : Tensor)
    input_chars, output_chars, operands = _parse_einsum_input(input_str, *args)
    contraction = Contraction.from_parse(input_chars, output_chars)
    sc = SizedContraction.from_contraction_and_operands(contraction, operands)
    generate_optimized_order(sc)
  end
end
