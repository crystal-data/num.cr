module Num::Einsum
  EINSUM_SYMBOLS_BASE = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

  private def parse_einsum_input(operands, *args : Tensor)
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
end
