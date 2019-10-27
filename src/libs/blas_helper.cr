require "./cblas"
require "../core/tensor"

module Bottle::Internal
  module BlasHelper
    ARG_NORMAL = 0
    ARG_BUFFER = 1
    ARG_INTOUT = 2
  end

  module Blas(T)
    include BlasHelper
    extend self

    macro blas(name, *args)
      {%
        blas_args = {
          "dot" => {2 => ARG_BUFFER, 4 => ARG_BUFFER},
        }
      %}

      {%
        if T == Float32
          typ = :s.id
        elsif T == Float64
          typ = :d.id
        elsif T == Complex
          typ = :z.id
        end
      %}

      {% func_args = blas_args[name.stringify] %}

      {% for arg, index in args %}
        {% argtype = func_args[index + 1] %}
        {% if argtype == ARG_BUFFER %}
        {% elsif argtype == ARG_INTOUT %}
          {{arg}} = 0
        {% else %}
        %var{index} = {{arg}}
        {% end %}
      {% end %}

      LibCblas.{{typ}}{{name}}(
        {% for arg, index in args %}
          {% argtype = func_args[index + 1] %}
          {% if argtype == ARG_BUFFER %}
            {{arg}},
          {% elsif argtype == ARG_INTOUT %}
            pointerof({{arg}}),
          {% else %}
            %var{index},
          {% end %}
        {% end %}
      )
    end
  end
end
