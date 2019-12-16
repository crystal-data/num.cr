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

    return {input_subscripts, output_subscript, args.to_a}
  end

  private def _compute_size_by_dict(indices, idx_dict)
    if indices.is_a?(String)
      indices = indices.each_char.to_set
    end
    ret = 1
    indices.each do |i|
      ret *= idx_dict[i]
    end
    ret
  end

  private def _flop_count(idx_contraction, inner, num_terms, size_dictionary)
    overall = _compute_size_by_dict(idx_contraction, size_dictionary)
    op_factor = {1, num_terms - 1}.max
    if inner
      op_factor += 1
    end

    overall * op_factor
  end

  private def _greedy_path(input_sets, output_set, idx_dict, memory_limit)
    if input_sets.size == 1
      return [[0]]
    elsif input_sets.size == 2
      return [[0, 1]]
    end
  end

  private def _find_contraction(positions, input_sets, output_set)
    idx_contract = Set(Char).new
    idx_remain = output_set.clone
    remaining = [] of Set(Char)
  end

  def einsum_path(operands, *args : Tensor, path_type = "greedy")
    input_subscripts, output_subscript, operands = _parse_einsum_input(operands, *args)

    input_list = input_subscripts.split(',')
    input_sets = input_list.map { |x| x.each_char.to_set }
    output_set = output_subscript.each_char.to_set
    indices = input_subscripts.gsub(",", "").each_char.to_set

    dimension_dict = Hash(Char, Int32).new
    broadcast_indices = (0...input_list.size).map { |_| [] of Char }

    input_list.each_with_index do |term, tnum|
      sh = operands[tnum].shape
      if sh.size != term.size
        raise "Einstein sum subscript #{input_subscripts[tnum]} does not contain the correct number of indices for operand #{tnum}"
      end

      term.each_char_with_index do |char, cnum|
        dim = sh[cnum]

        if dim == 1
          broadcast_indices[tnum] << char
        end

        if dimension_dict.has_key?(char)
          if dimension_dict[char] == 1
            dimension_dict[char] = dim
          elsif dim != 1 && dim != dimension_dict[char]
            raise "Size of label #{char} for operand #{tnum} does not match previous terms #{dim}"
          end
        else
          dimension_dict[char] = dim
        end
      end
    end

    broadcast_indices_set = broadcast_indices.map { |e| e.to_set }
    size_list = (input_list + [output_subscript]).map { |term| _compute_size_by_dict(term, dimension_dict) }

    max_size = size_list.max

    inner_product = ((input_sets.map &.size).sum - indices.size) > 0
    naive_cost = _flop_count(indices, inner_product, input_list.size, dimension_dict)
    path = _greedy_path(input_sets, output_set, dimension_dict, max_size)
  end
end
