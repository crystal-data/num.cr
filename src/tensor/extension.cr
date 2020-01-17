require "../libs/lapack"
require "./work"

module NumInternal::LapackHelper
  ARG_NORMAL          = 0
  ARG_MATRIX          = 1
  ARG_INTOUT          = 2
  WORK_NONE           = 0
  WORK_DETECT         = 1
  WORK_DETECT_SPECIAL = 2
  WORK_EMPTY          = 3
  WORK_PARAM1         = 4
  WORK_PARAM2         = 5
end

class Tensor(T)
  private macro of_real_type(size)
    {% if T == Complex %}
      Tensor(Float64).new([size])
    {% else %}
      Tensor(T).new([size])
    {% end %}
  end

  private macro alloc_real_type(size)
    {% if T == Float32 %}
      WORK_POOL.get_f32({{size}})
    {% else %}
      WORK_POOL.get_f64({{size}})
    {% end %}
  end

  private macro alloc_type(size)
    {% if T == Complex %}
      WORK_POOL.get_cmplx({{size}})
    {% elsif T == Float32 %}
      WORK_POOL.get_f32({{size}})
    {% else %}
      WORK_POOL.get_f64({{size}})
    {% end %}
  end

  macro lapack_util(name, worksize, *args)
    WORK_POOL.reallocate(worksize*{% if T == Complex %} sizeof(Float64) {% else %} sizeof(T) {% end %})
    %buf = alloc_real_type(worksize)
    {% if T == Float32
         typ = :s.id
       elsif T == Float64
         typ = :d.id
       elsif T == Complex
         typ = :z.id
       end %}

    {% for arg, index in args %}
      {% if !(arg.stringify =~ /^tensor\(.*\)$/) %}
        %var{index} = {{arg}}
      {% end %}
    {% end %}

    %result = LibLapack.{{typ}}{{name}}(
      {% for arg, index in args %}
        {% if !(arg.stringify =~ /^tensor\(.*\)$/) %}
          pointerof(%var{index}),
        {% else %}
          {{arg.stringify.gsub(/^tensor\((.*)\)$/, "(\\1)").id}},
        {% end %}
      {% end %}
      %buf)
    WORK_POOL.release
    %result
  end

  macro lapack(name, *args, worksize = nil)
      {%
        lapack_args = {
          "gbtrf" => {5 => ARG_MATRIX, 7 => ARG_MATRIX},
          "gebal" => {3 => ARG_MATRIX, 5 => ARG_INTOUT, 6 => ARG_INTOUT, 7 => ARG_MATRIX},
          "gees"  => {3 => ARG_MATRIX, 5 => ARG_MATRIX, 7 => ARG_INTOUT, 8 => ARG_MATRIX, 9 => ARG_MATRIX, 10 => ARG_MATRIX},
          "geev"  => {4 => ARG_MATRIX, 6 => ARG_MATRIX, 7 => ARG_MATRIX, 8 => ARG_MATRIX, 10 => ARG_MATRIX},
          "gehrd" => {4 => ARG_MATRIX, 6 => ARG_MATRIX},
          "gels"  => {5 => ARG_MATRIX, 7 => ARG_MATRIX},
          "gelsd" => {4 => ARG_MATRIX, 6 => ARG_MATRIX, 8 => ARG_MATRIX, 10 => ARG_INTOUT},
          "gelsy" => {4 => ARG_MATRIX, 6 => ARG_MATRIX, 8 => ARG_MATRIX, 10 => ARG_INTOUT},
          "geqp3" => {3 => ARG_MATRIX, 5 => ARG_MATRIX, 6 => ARG_MATRIX},
          "geqrf" => {3 => ARG_MATRIX, 5 => ARG_MATRIX},
          "gerqf" => {3 => ARG_MATRIX, 5 => ARG_MATRIX},
          "gelqf" => {3 => ARG_MATRIX, 5 => ARG_MATRIX},
          "geqlf" => {3 => ARG_MATRIX, 5 => ARG_MATRIX},
          "gesdd" => {4 => ARG_MATRIX, 6 => ARG_MATRIX, 7 => ARG_MATRIX, 9 => ARG_MATRIX},
          "gesv"  => {3 => ARG_MATRIX, 5 => ARG_MATRIX, 6 => ARG_MATRIX},
          "getrf" => {3 => ARG_MATRIX, 5 => ARG_MATRIX},
          "getri" => {2 => ARG_MATRIX, 4 => ARG_MATRIX},
          "getrs" => {4 => ARG_MATRIX, 6 => ARG_MATRIX, 7 => ARG_MATRIX},
          "gges"  => {4 => ARG_MATRIX, 6 => ARG_MATRIX, 8 => ARG_MATRIX, 10 => ARG_INTOUT, 11 => ARG_MATRIX, 12 => ARG_MATRIX, 13 => ARG_MATRIX, 14 => ARG_MATRIX, 16 => ARG_MATRIX},
          "ggev"  => {4 => ARG_MATRIX, 6 => ARG_MATRIX, 8 => ARG_MATRIX, 9 => ARG_MATRIX, 10 => ARG_MATRIX, 11 => ARG_MATRIX, 13 => ARG_MATRIX},
          "heevr" => {5 => ARG_MATRIX, 12 => ARG_INTOUT, 13 => ARG_MATRIX, 14 => ARG_MATRIX, 16 => ARG_MATRIX},
          "hegvd" => {5 => ARG_MATRIX, 7 => ARG_MATRIX, 9 => ARG_MATRIX},
          "hesv"  => {4 => ARG_MATRIX, 6 => ARG_MATRIX, 7 => ARG_MATRIX},
          "hetrf" => {3 => ARG_MATRIX, 5 => ARG_MATRIX},
          "hetri" => {3 => ARG_MATRIX, 5 => ARG_MATRIX},
          "orghr" => {4 => ARG_MATRIX, 6 => ARG_MATRIX},
          "orgqr" => {4 => ARG_MATRIX, 6 => ARG_MATRIX},
          "orgrq" => {4 => ARG_MATRIX, 6 => ARG_MATRIX},
          "orglq" => {4 => ARG_MATRIX, 6 => ARG_MATRIX},
          "orgql" => {4 => ARG_MATRIX, 6 => ARG_MATRIX},
          "posv"  => {4 => ARG_MATRIX, 6 => ARG_MATRIX},
          "potrf" => {3 => ARG_MATRIX},
          "potri" => {3 => ARG_MATRIX},
          "potrs" => {4 => ARG_MATRIX, 6 => ARG_MATRIX},
          "syevr" => {5 => ARG_MATRIX, 12 => ARG_INTOUT, 13 => ARG_MATRIX, 14 => ARG_MATRIX, 16 => ARG_MATRIX},
          "sygvd" => {5 => ARG_MATRIX, 7 => ARG_MATRIX, 9 => ARG_MATRIX},
          "sysv"  => {4 => ARG_MATRIX, 6 => ARG_MATRIX, 7 => ARG_MATRIX},
          "sytrf" => {3 => ARG_MATRIX, 5 => ARG_MATRIX},
          "sytri" => {3 => ARG_MATRIX, 5 => ARG_MATRIX},
          "trtri" => {4 => ARG_MATRIX},
          "trtrs" => {6 => ARG_MATRIX, 8 => ARG_MATRIX},
          "syev"  => {4 => ARG_MATRIX, 6 => ARG_MATRIX},
          "gecon" => {3 => ARG_MATRIX},
        }

        lapack_args_complex = {
          "gees" => {3 => ARG_MATRIX, 5 => ARG_MATRIX, 7 => ARG_INTOUT, 8 => ARG_MATRIX, 9 => ARG_MATRIX},
          "geev" => {4 => ARG_MATRIX, 6 => ARG_MATRIX, 7 => ARG_MATRIX, 9 => ARG_MATRIX},
          "gges" => {4 => ARG_MATRIX, 6 => ARG_MATRIX, 8 => ARG_MATRIX, 10 => ARG_INTOUT, 11 => ARG_MATRIX, 12 => ARG_MATRIX, 13 => ARG_MATRIX, 15 => ARG_MATRIX},
          "ggev" => {4 => ARG_MATRIX, 6 => ARG_MATRIX, 8 => ARG_MATRIX, 9 => ARG_MATRIX, 10 => ARG_MATRIX, 12 => ARG_MATRIX},
        }

        lapack_worksize = {
          "gees"  => {"cwork" => WORK_DETECT, "rwork" => WORK_PARAM1, "bwork" => WORK_EMPTY},
          "geev"  => {"cwork" => WORK_DETECT, "rwork" => WORK_PARAM1},
          "gehrd" => {"cwork" => WORK_DETECT},
          "gels"  => {"cwork" => WORK_DETECT},
          "gelsd" => {"cwork" => WORK_DETECT, "rwork" => WORK_DETECT_SPECIAL, "iwork" => WORK_DETECT_SPECIAL},
          "gelsy" => {"cwork" => WORK_DETECT, "rwork" => WORK_PARAM1},
          "geqp3" => {"cwork" => WORK_DETECT, "rwork" => WORK_PARAM1},
          "geqrf" => {"cwork" => WORK_DETECT},
          "gerqf" => {"cwork" => WORK_DETECT},
          "gelqf" => {"cwork" => WORK_DETECT},
          "geqlf" => {"cwork" => WORK_DETECT},
          "gesdd" => {"cwork" => WORK_DETECT, "rwork" => WORK_PARAM1, "iwork" => WORK_PARAM2},
          "getri" => {"cwork" => WORK_DETECT},
          "gges"  => {"cwork" => WORK_DETECT, "rwork" => WORK_PARAM1, "bwork" => WORK_EMPTY},
          "ggev"  => {"cwork" => WORK_DETECT, "rwork" => WORK_PARAM1},
          "heevr" => {"cwork" => WORK_DETECT, "rwork" => WORK_DETECT, "iwork" => WORK_DETECT},
          "hegvd" => {"cwork" => WORK_DETECT, "rwork" => WORK_DETECT, "iwork" => WORK_DETECT},
          "hesv"  => {"cwork" => WORK_DETECT},
          "hetrf" => {"cwork" => WORK_DETECT},
          "hetri" => {"cwork" => WORK_PARAM1},
          "orghr" => {"cwork" => WORK_DETECT},
          "orgqr" => {"cwork" => WORK_DETECT},
          "orgrq" => {"cwork" => WORK_DETECT},
          "orglq" => {"cwork" => WORK_DETECT},
          "orgql" => {"cwork" => WORK_DETECT},
          "syevr" => {"cwork" => WORK_DETECT, "iwork" => WORK_DETECT},
          "sygvd" => {"cwork" => WORK_DETECT, "iwork" => WORK_DETECT},
          "sysv"  => {"cwork" => WORK_DETECT},
          "sytrf" => {"cwork" => WORK_DETECT},
          "sytri" => {"cwork" => WORK_PARAM1},
          "syev"  => {"cwork" => WORK_DETECT, "rwork" => WORK_PARAM1},
          "gecon" => {"cwork" => WORK_DETECT, "rwork" => WORK_PARAM1},
        }
      %}


      {% if T == Float32
           typ = :s.id
         elsif T == Float64
           typ = :d.id
         elsif T == Complex
           typ = :z.id
         end %}
      {% if T == Complex
           func_args = lapack_args_complex[name.stringify] || lapack_args[name.stringify]
         else
           func_args = lapack_args[name.stringify]
         end %}
      {% func_worksize = lapack_worksize[name.stringify] %}

      {% if T == Complex
           name = name.stringify.gsub(/^(or)/, "un").id
         end %}

      {% for arg, index in args %}
        {% argtype = func_args[index + 1] %}
        {% if argtype == ARG_MATRIX %}
        {% elsif argtype == ARG_INTOUT %}
          {{arg}} = 0
        {% else %}
        %var{index} = {{arg}}
        {% end %}
      {% end %}

      {% if func_worksize && (func_worksize.values.includes?(WORK_DETECT) || func_worksize.values.includes?(WORK_DETECT_SPECIAL)) %}
        #let's detect sizes
        #1. init vars
        {% if func_worksize["cwork"] %}
          %csize = -1
          %cresult = T.new(0.0)
        {% end %}
        {% if func_worksize["rwork"] %}
          %rsize = -1
          %rresult = of_real_type(0.0)
        {% end %}
        {% if func_worksize["iwork"] %}
          %isize = -1
          %iresult = 0
        {% end %}

        # 2. do workspace query
        %info = 0
        LibLapack.{{typ}}{{name}}(
          {% for arg, index in args %}
          {% argtype = func_args[index + 1] %}
          {% if argtype == ARG_MATRIX %}
            {{arg}},
          {% elsif argtype == ARG_INTOUT %}
            pointerof({{arg}}),
          {% else %}
           pointerof(%var{index}),
          {% end %}
          {% end %}

          {% if func_worksize %}
            {% if func_worksize["cwork"] %}
              {% if T == Complex %} pointerof(%cresult).as(LibCblas::ComplexDouble*) {% else %}pointerof(%cresult) {% end %},
               {% if func_worksize["cwork"] == WORK_DETECT %}
                 pointerof(%csize),
               {% end %}
            {% end %}
            {% if T == Complex && func_worksize["rwork"] %}
              pointerof(%rresult),
               {% if func_worksize["rwork"] == WORK_DETECT %}
                 pointerof(%rsize),
               {% end %}
            {% end %}
            {% if func_worksize["iwork"] %}
              pointerof(%iresult),
               {% if func_worksize["iwork"] == WORK_DETECT %}
                 pointerof(%isize),
               {% end %}
            {% end %}
            {% if func_worksize["bwork"] %}
               nil,
            {% end %}
          {% end %}

          pointerof(%info))
         #3. set sizes
         {% if func_worksize["cwork"] == WORK_DETECT %}
           %csize = {% if T == Complex %} %cresult.real.to_i {% else %}%cresult.to_i {% end %}
         {% end %}
         {% if T == Complex && func_worksize["rwork"] == WORK_DETECT || func_worksize["rwork"] == WORK_DETECT_SPECIAL %}
           %rsize = %rresult.to_i
         {% end %}
         {% if func_worksize["iwork"] == WORK_DETECT || func_worksize["iwork"] == WORK_DETECT_SPECIAL %}
           %isize = %iresult
         {% end %}
      {% end %}


      {% if func_worksize %}
        %asize = 0
        {% if func_worksize["cwork"] %}
          {% if func_worksize["cwork"] == WORK_PARAM1 %}
            %csize = {{worksize[0]}}
          {% elsif func_worksize["cwork"] == WORK_PARAM2 %}
            %csize = {{worksize[1]}}
          {% end %}
          %asize += %csize*sizeof(T)
        {% end %}

        {% if T == Complex && func_worksize["rwork"] %}
          {% if func_worksize["rwork"] == WORK_PARAM1 %}
            %rsize = {{worksize[0]}}
          {% elsif func_worksize["rwork"] == WORK_PARAM2 %}
            %rsize = {{worksize[1]}}
          {% end %}
          %asize += %rsize*{% if T == Complex %} sizeof(Float64) {% else %} sizeof(T) {% end %}
        {% end %}

        {% if func_worksize["iwork"] %}
          {% if func_worksize["iwork"] == WORK_PARAM1 %}
            %isize = {{worksize[0]}}
          {% elsif func_worksize["iwork"] == WORK_PARAM2 %}
            %isize = {{worksize[1]}}
          {% end %}
          %asize += %isize*sizeof(Int32)
        {% end %}

        WORK_POOL.reallocate(%asize)

        {% if func_worksize["cwork"] %}
          %cbuf = alloc_type(%csize)
        {% end %}

        {% if T == Complex && func_worksize["rwork"] %}
          %rbuf = alloc_real_type(%rsize)
        {% end %}

        {% if func_worksize["iwork"] %}
          %ibuf = WORK_POOL.get_i32(%isize)
        {% end %}

      {% end %}

       %info = 0
       LibLapack.{{typ}}{{name}}(
         {% for arg, index in args %}
         {% argtype = func_args[index + 1] %}
         {% if argtype == ARG_MATRIX %}
           {{arg}},
         {% elsif argtype == ARG_INTOUT %}
           pointerof({{arg}}),
         {% else %}
          pointerof(%var{index}),
         {% end %}
         {% end %}

         {% if func_worksize %}
           {% if func_worksize["cwork"] %}
              %cbuf,
              {% if func_worksize["cwork"] == WORK_DETECT %}
                pointerof(%csize),
              {% end %}
           {% end %}
           {% if T == Complex && func_worksize["rwork"] %}
              %rbuf,
              {% if func_worksize["rwork"] == WORK_DETECT %}
                pointerof(%rsize),
              {% end %}
           {% end %}
           {% if func_worksize["iwork"] %}
              %ibuf,
              {% if func_worksize["iwork"] == WORK_DETECT %}
                pointerof(%isize),
              {% end %}
           {% end %}
           {% if func_worksize["bwork"] %}
              nil,
           {% end %}
         {% end %}

         pointerof(%info))

         {% if func_worksize %}
           WORK_POOL.release
         {% end %}
    raise "LAPACK.{{typ}}{{name}} returned #{%info}" if %info != 0
  end

  macro blas(storage, name, *args)
    {%
      if T == Float32
        typ = :s.id
      elsif T == Float64
        typ = :d.id
      elsif T == Complex
        typ = :z.id
      end
    %}
    LibCblas.{{typ}}{{storage}}{{name}}(LibCblas::ROW_MAJOR, {{*args}})
  end

  private macro blas_const(x)
    {% if T == Complex %}
      pointerof({{x}}).as(LibCblas::ComplexDouble*)
    {% else %}
      {{x}}
    {% end %}
  end
end
