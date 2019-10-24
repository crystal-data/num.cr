require "../core/vector"

module Bottle::B::Util
  extend self

  # This is a complete mess and needs to be redone, I was just
  # so tired of debugging ridiculously hard to read outputs
  # and this made my life so much easier.
  def matrix_print(io, data : Matrix(U), prefix = "Matrix[") forall U
  end

  # This is a complete mess and needs to be redone, I was just
  # so tired of debugging ridiculously hard to read outputs
  # and this made my life so much easier.
  def vector_print(io, data : Vector(U), prefix = "Vector[") forall U
    too_big = data.size > 1000
    if too_big
      data = data[[0, 1, 2, data.size - 3, data.size - 2, data.size - 1]]
    end

    longest = uninitialized U

    {% if U == Bool %}
      longest = false
    {% else %}
      longest = data.max.round(3)
    {% end %}
    rj = "#{longest}".size + (U == Int32 || U == Bool ? 2 : 5)

    emptyp = " " * prefix.size

    lw = 75 - prefix.size
    epl = lw // rj
    nl = rj * data.size // lw + 1

    index = 0
    io << prefix

    if too_big
      3.times do |i|
        {% if U == Bool %}
          io << data[i].to_s.rjust(rj)
        {% else %}
          io << data[i].round(3).to_s.rjust(rj)
        {% end %}
      end
      io << "  ...  "
      3.times do |i|
        {% if U == Bool %}
          io << data[i + 3].to_s.rjust(rj)
        {% else %}
          io << data[i + 3].round(3).to_s.rjust(rj)
        {% end %}
      end
    else
      (nl + 1).times do |l|
        epl.times do |e|
          if index < data.size
            {% if U == Bool %}
              io << data[index].to_s.rjust(rj)
            {% else %}
              io << data[index].round(3).to_s.rjust(rj)
            {% end %}
            index += 1
          end
        end
        if index < data.size
          io << "\n" << emptyp
        end
      end
    end
    io << "]"
  end
end
