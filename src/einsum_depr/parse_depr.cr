require "../tensor/tensor"

module Num::Einsum
  extend self
  EINSUM_SYMBOLS     = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
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
      ellipse_inds = unused.join
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

        normal_inds = (output_subscript.each_char.to_set - out_ellipse.each_char.to_set).to_a.sort.join
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

    contract = _find_contraction(0...(input_sets.size), input_sets, output_set)
    idx_result, new_input_sets, idx_removed, idx_contract = contract
    naive_cost = _flop_count(idx_contract, idx_removed, input_sets.size, idx_dict)

    comb_iter = (0...(input_sets.size)).to_a.each_combination
    known_contractions = [] of Tuple(Tuple(Int32, Int32), Array(Int32), Array(Set(Char)))

    path_cost = 0
    path = [] of Array(Int32)

    (input_sets.size - 1).times do |iteration|
      comb_iter.each do |positions|
        if !input_sets[positions[0]].intersects?(input_sets[positions[1]])
          next
        end
        result = _parse_possible_contraction(positions, input_sets, output_set, idx_dict, memory_limit, path_cost, naive_cost)
        if !result.nil?
          known_contractions << result
        end
      end

      if known_contractions.size == 0
        (0...(input_sets.size)).to_a.each_combination do |positions|
          result = _parse_possible_contraction(positions, input_sets, output_set, idx_dict, memory_limit, path_cost, naive_cost)
          if !result.nil?
            known_contractions << result
          end
        end

        if known_contractions.size == 0
          path << (0...(input_sets.size)).to_a
          break
        end
      end

      best = known_contractions.min_by { |e| e[0] }
      known_contractions = _update_other_results(known_contractions, best)

      input_sets = best[2]
      new_tensor_pos = input_sets.size - 1
      comb_iter = (0...new_tensor_pos).map { |e| [e, new_tensor_pos] }

      path << best[1]
      path_cost += best[0][1]
    end

    path
  end

  private def _find_contraction(positions, input_sets, output_set)
    idx_contract = Set(Char).new
    idx_remain = output_set.clone
    remaining = [] of Set(Char)

    input_sets.each_with_index do |value, ind|
      if positions.includes?(ind)
        idx_contract |= value
      else
        remaining << value
        idx_remain |= value
      end
    end

    new_result = idx_remain & idx_contract
    idx_removed = (idx_contract - new_result)
    remaining << new_result
    {new_result, remaining, idx_removed, idx_contract}
  end

  private def _parse_possible_contraction(positions, input_sets, output_set, idx_dict, memory_limit, path_cost, naive_cost)
    contract = _find_contraction(positions, input_sets, output_set)
    idx_result, new_input_sets, idx_removed, idx_contract = contract

    new_size = _compute_size_by_dict(idx_result, idx_dict)
    if new_size > memory_limit
      return nil
    end

    old_sizes = positions.map { |p| _compute_size_by_dict(input_sets[p], idx_dict) }
    removed_size = old_sizes.sum - new_size

    cost = _flop_count(idx_contract, idx_removed, positions.size, idx_dict)
    sort = {-removed_size, cost}

    if (path_cost + cost) > naive_cost
      return nil
    end

    {sort, positions, new_input_sets}
  end

  private def _update_other_results(results, best)
    best_con = best[1]
    bx, by = best_con
    mod_results = [] of Tuple(Tuple(Int32, Int32), Array(Int32), Array(Set(Char)))

    results.each do |cost, (x, y), con_sets|
      if best_con.includes?(x) || best_con.includes?(y)
        next
      end
      con_sets.delete_at(by - (by > x ? 1 : 0) - (by > y ? 1 : 0))
      con_sets.delete_at(bx - (bx > x ? 1 : 0) - (bx > y ? 1 : 0))
      con_sets.insert(-1, best[2][-1])

      mod_con = [x - (x > bx ? 1 : 0) - (x > by ? 1 : 0), y - (y > bx ? 1 : 0) - (y > by ? 1 : 0)]
      mod_results << {cost, mod_con, con_sets}
    end

    mod_results
  end

  def einsum_path(operands, *args : Tensor, optimize = "greedy")
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

    cost_list = [] of Int32
    scale_list = [] of Int32
    size_list = [] of Int32

    path.each_with_index do |contract_inds, cnum|
      contract_inds = contract_inds.sort.reverse
      contract = _find_contraction(contract_inds, input_sets, output_set)
      out_inds, input_sets, idx_removed, idx_contract = contract
      cost = _flop_count(idx_contract, idx_removed, contract_inds.size, dimension_dict)
      cost_list << cost
      scale_list << idx_contract.size
      size_list << _compute_size_by_dict(out_inds, dimension_dict)

      contract_inds.each do |x|
        popped = input_list[x]
        input_list.delete_at(x)
      end
    end
  end
end
