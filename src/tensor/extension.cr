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
          "gbtrf" => {5 => NumInternal::ARG_MATRIX, 7 => NumInternal::ARG_MATRIX},
          "gebal" => {3 => NumInternal::ARG_MATRIX, 5 => NumInternal::ARG_INTOUT, 6 => NumInternal::ARG_INTOUT, 7 => NumInternal::ARG_MATRIX},
          "gees"  => {3 => NumInternal::ARG_MATRIX, 5 => NumInternal::ARG_MATRIX, 7 => NumInternal::ARG_INTOUT, 8 => NumInternal::ARG_MATRIX, 9 => NumInternal::ARG_MATRIX, 10 => NumInternal::ARG_MATRIX},
          "geev"  => {4 => NumInternal::ARG_MATRIX, 6 => NumInternal::ARG_MATRIX, 7 => NumInternal::ARG_MATRIX, 8 => NumInternal::ARG_MATRIX, 10 => NumInternal::ARG_MATRIX},
          "gehrd" => {4 => NumInternal::ARG_MATRIX, 6 => NumInternal::ARG_MATRIX},
          "gels"  => {5 => NumInternal::ARG_MATRIX, 7 => NumInternal::ARG_MATRIX},
          "gelsd" => {4 => NumInternal::ARG_MATRIX, 6 => NumInternal::ARG_MATRIX, 8 => NumInternal::ARG_MATRIX, 10 => NumInternal::ARG_INTOUT},
          "gelsy" => {4 => NumInternal::ARG_MATRIX, 6 => NumInternal::ARG_MATRIX, 8 => NumInternal::ARG_MATRIX, 10 => NumInternal::ARG_INTOUT},
          "geqp3" => {3 => NumInternal::ARG_MATRIX, 5 => NumInternal::ARG_MATRIX, 6 => NumInternal::ARG_MATRIX},
          "geqrf" => {3 => NumInternal::ARG_MATRIX, 5 => NumInternal::ARG_MATRIX},
          "gerqf" => {3 => NumInternal::ARG_MATRIX, 5 => NumInternal::ARG_MATRIX},
          "gelqf" => {3 => NumInternal::ARG_MATRIX, 5 => NumInternal::ARG_MATRIX},
          "geqlf" => {3 => NumInternal::ARG_MATRIX, 5 => NumInternal::ARG_MATRIX},
          "gesdd" => {4 => NumInternal::ARG_MATRIX, 6 => NumInternal::ARG_MATRIX, 7 => NumInternal::ARG_MATRIX, 9 => NumInternal::ARG_MATRIX},
          "gesv"  => {3 => NumInternal::ARG_MATRIX, 5 => NumInternal::ARG_MATRIX, 6 => NumInternal::ARG_MATRIX},
          "getrf" => {3 => NumInternal::ARG_MATRIX, 5 => NumInternal::ARG_MATRIX},
          "getri" => {2 => NumInternal::ARG_MATRIX, 4 => NumInternal::ARG_MATRIX},
          "getrs" => {4 => NumInternal::ARG_MATRIX, 6 => NumInternal::ARG_MATRIX, 7 => NumInternal::ARG_MATRIX},
          "gges"  => {4 => NumInternal::ARG_MATRIX, 6 => NumInternal::ARG_MATRIX, 8 => NumInternal::ARG_MATRIX, 10 => NumInternal::ARG_INTOUT, 11 => NumInternal::ARG_MATRIX, 12 => NumInternal::ARG_MATRIX, 13 => NumInternal::ARG_MATRIX, 14 => NumInternal::ARG_MATRIX, 16 => NumInternal::ARG_MATRIX},
          "ggev"  => {4 => NumInternal::ARG_MATRIX, 6 => NumInternal::ARG_MATRIX, 8 => NumInternal::ARG_MATRIX, 9 => NumInternal::ARG_MATRIX, 10 => NumInternal::ARG_MATRIX, 11 => NumInternal::ARG_MATRIX, 13 => NumInternal::ARG_MATRIX},
          "heevr" => {5 => NumInternal::ARG_MATRIX, 12 => NumInternal::ARG_INTOUT, 13 => NumInternal::ARG_MATRIX, 14 => NumInternal::ARG_MATRIX, 16 => NumInternal::ARG_MATRIX},
          "hegvd" => {5 => NumInternal::ARG_MATRIX, 7 => NumInternal::ARG_MATRIX, 9 => NumInternal::ARG_MATRIX},
          "hesv"  => {4 => NumInternal::ARG_MATRIX, 6 => NumInternal::ARG_MATRIX, 7 => NumInternal::ARG_MATRIX},
          "hetrf" => {3 => NumInternal::ARG_MATRIX, 5 => NumInternal::ARG_MATRIX},
          "hetri" => {3 => NumInternal::ARG_MATRIX, 5 => NumInternal::ARG_MATRIX},
          "orghr" => {4 => NumInternal::ARG_MATRIX, 6 => NumInternal::ARG_MATRIX},
          "orgqr" => {4 => NumInternal::ARG_MATRIX, 6 => NumInternal::ARG_MATRIX},
          "orgrq" => {4 => NumInternal::ARG_MATRIX, 6 => NumInternal::ARG_MATRIX},
          "orglq" => {4 => NumInternal::ARG_MATRIX, 6 => NumInternal::ARG_MATRIX},
          "orgql" => {4 => NumInternal::ARG_MATRIX, 6 => NumInternal::ARG_MATRIX},
          "posv"  => {4 => NumInternal::ARG_MATRIX, 6 => NumInternal::ARG_MATRIX},
          "potrf" => {3 => NumInternal::ARG_MATRIX},
          "potri" => {3 => NumInternal::ARG_MATRIX},
          "potrs" => {4 => NumInternal::ARG_MATRIX, 6 => NumInternal::ARG_MATRIX},
          "syevr" => {5 => NumInternal::ARG_MATRIX, 12 => NumInternal::ARG_INTOUT, 13 => NumInternal::ARG_MATRIX, 14 => NumInternal::ARG_MATRIX, 16 => NumInternal::ARG_MATRIX},
          "sygvd" => {5 => NumInternal::ARG_MATRIX, 7 => NumInternal::ARG_MATRIX, 9 => NumInternal::ARG_MATRIX},
          "sysv"  => {4 => NumInternal::ARG_MATRIX, 6 => NumInternal::ARG_MATRIX, 7 => NumInternal::ARG_MATRIX},
          "sytrf" => {3 => NumInternal::ARG_MATRIX, 5 => NumInternal::ARG_MATRIX},
          "sytri" => {3 => NumInternal::ARG_MATRIX, 5 => NumInternal::ARG_MATRIX},
          "trtri" => {4 => NumInternal::ARG_MATRIX},
          "trtrs" => {6 => NumInternal::ARG_MATRIX, 8 => NumInternal::ARG_MATRIX},
          "syev"  => {4 => NumInternal::ARG_MATRIX, 6 => NumInternal::ARG_MATRIX},
          "gecon" => {3 => NumInternal::ARG_MATRIX},
        }

        lapack_args_complex = {
          "gees" => {3 => NumInternal::ARG_MATRIX, 5 => NumInternal::ARG_MATRIX, 7 => NumInternal::ARG_INTOUT, 8 => NumInternal::ARG_MATRIX, 9 => NumInternal::ARG_MATRIX},
          "geev" => {4 => NumInternal::ARG_MATRIX, 6 => NumInternal::ARG_MATRIX, 7 => NumInternal::ARG_MATRIX, 9 => NumInternal::ARG_MATRIX},
          "gges" => {4 => NumInternal::ARG_MATRIX, 6 => NumInternal::ARG_MATRIX, 8 => NumInternal::ARG_MATRIX, 10 => NumInternal::ARG_INTOUT, 11 => NumInternal::ARG_MATRIX, 12 => NumInternal::ARG_MATRIX, 13 => NumInternal::ARG_MATRIX, 15 => NumInternal::ARG_MATRIX},
          "ggev" => {4 => NumInternal::ARG_MATRIX, 6 => NumInternal::ARG_MATRIX, 8 => NumInternal::ARG_MATRIX, 9 => NumInternal::ARG_MATRIX, 10 => NumInternal::ARG_MATRIX, 12 => NumInternal::ARG_MATRIX},
        }

        lapack_worksize = {
          "gees"  => {"cwork" => NumInternal::WORK_DETECT, "rwork" => NumInternal::WORK_PARAM1, "bwork" => NumInternal::WORK_EMPTY},
          "geev"  => {"cwork" => NumInternal::WORK_DETECT, "rwork" => NumInternal::WORK_PARAM1},
          "gehrd" => {"cwork" => NumInternal::WORK_DETECT},
          "gels"  => {"cwork" => NumInternal::WORK_DETECT},
          "gelsd" => {"cwork" => NumInternal::WORK_DETECT, "rwork" => NumInternal::WORK_DETECT_SPECIAL, "iwork" => NumInternal::WORK_DETECT_SPECIAL},
          "gelsy" => {"cwork" => NumInternal::WORK_DETECT, "rwork" => NumInternal::WORK_PARAM1},
          "geqp3" => {"cwork" => NumInternal::WORK_DETECT, "rwork" => NumInternal::WORK_PARAM1},
          "geqrf" => {"cwork" => NumInternal::WORK_DETECT},
          "gerqf" => {"cwork" => NumInternal::WORK_DETECT},
          "gelqf" => {"cwork" => NumInternal::WORK_DETECT},
          "geqlf" => {"cwork" => NumInternal::WORK_DETECT},
          "gesdd" => {"cwork" => NumInternal::WORK_DETECT, "rwork" => NumInternal::WORK_PARAM1, "iwork" => NumInternal::WORK_PARAM2},
          "getri" => {"cwork" => NumInternal::WORK_DETECT},
          "gges"  => {"cwork" => NumInternal::WORK_DETECT, "rwork" => NumInternal::WORK_PARAM1, "bwork" => NumInternal::WORK_EMPTY},
          "ggev"  => {"cwork" => NumInternal::WORK_DETECT, "rwork" => NumInternal::WORK_PARAM1},
          "heevr" => {"cwork" => NumInternal::WORK_DETECT, "rwork" => NumInternal::WORK_DETECT, "iwork" => NumInternal::WORK_DETECT},
          "hegvd" => {"cwork" => NumInternal::WORK_DETECT, "rwork" => NumInternal::WORK_DETECT, "iwork" => NumInternal::WORK_DETECT},
          "hesv"  => {"cwork" => NumInternal::WORK_DETECT},
          "hetrf" => {"cwork" => NumInternal::WORK_DETECT},
          "hetri" => {"cwork" => NumInternal::WORK_PARAM1},
          "orghr" => {"cwork" => NumInternal::WORK_DETECT},
          "orgqr" => {"cwork" => NumInternal::WORK_DETECT},
          "orgrq" => {"cwork" => NumInternal::WORK_DETECT},
          "orglq" => {"cwork" => NumInternal::WORK_DETECT},
          "orgql" => {"cwork" => NumInternal::WORK_DETECT},
          "syevr" => {"cwork" => NumInternal::WORK_DETECT, "iwork" => NumInternal::WORK_DETECT},
          "sygvd" => {"cwork" => NumInternal::WORK_DETECT, "iwork" => NumInternal::WORK_DETECT},
          "sysv"  => {"cwork" => NumInternal::WORK_DETECT},
          "sytrf" => {"cwork" => NumInternal::WORK_DETECT},
          "sytri" => {"cwork" => NumInternal::WORK_PARAM1},
          "syev"  => {"cwork" => NumInternal::WORK_DETECT, "rwork" => NumInternal::WORK_PARAM1},
          "gecon" => {"cwork" => NumInternal::WORK_DETECT, "rwork" => NumInternal::WORK_PARAM1},
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
        {% if argtype == NumInternal::ARG_MATRIX %}
        {% elsif argtype == NumInternal::ARG_INTOUT %}
          {{arg}} = 0
        {% else %}
        %var{index} = {{arg}}
        {% end %}
      {% end %}

      {% if func_worksize && (func_worksize.values.includes?(NumInternal::WORK_DETECT) || func_worksize.values.includes?(NumInternal::WORK_DETECT_SPECIAL)) %}
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
          {% if argtype == NumInternal::ARG_MATRIX %}
            {{arg}},
          {% elsif argtype == NumInternal::ARG_INTOUT %}
            pointerof({{arg}}),
          {% else %}
           pointerof(%var{index}),
          {% end %}
          {% end %}

          {% if func_worksize %}
            {% if func_worksize["cwork"] %}
              {% if T == Complex %} pointerof(%cresult).as(LibCblas::ComplexDouble*) {% else %}pointerof(%cresult) {% end %},
               {% if func_worksize["cwork"] == NumInternal::WORK_DETECT %}
                 pointerof(%csize),
               {% end %}
            {% end %}
            {% if T == Complex && func_worksize["rwork"] %}
              pointerof(%rresult),
               {% if func_worksize["rwork"] == NumInternal::WORK_DETECT %}
                 pointerof(%rsize),
               {% end %}
            {% end %}
            {% if func_worksize["iwork"] %}
              pointerof(%iresult),
               {% if func_worksize["iwork"] == NumInternal::WORK_DETECT %}
                 pointerof(%isize),
               {% end %}
            {% end %}
            {% if func_worksize["bwork"] %}
               nil,
            {% end %}
          {% end %}

          pointerof(%info))
         #3. set sizes
         {% if func_worksize["cwork"] == NumInternal::WORK_DETECT %}
           %csize = {% if T == Complex %} %cresult.real.to_i {% else %}%cresult.to_i {% end %}
         {% end %}
         {% if T == Complex && func_worksize["rwork"] == NumInternal::WORK_DETECT || func_worksize["rwork"] == NumInternal::WORK_DETECT_SPECIAL %}
           %rsize = %rresult.to_i
         {% end %}
         {% if func_worksize["iwork"] == NumInternal::WORK_DETECT || func_worksize["iwork"] == NumInternal::WORK_DETECT_SPECIAL %}
           %isize = %iresult
         {% end %}
      {% end %}


      {% if func_worksize %}
        %asize = 0
        {% if func_worksize["cwork"] %}
          {% if func_worksize["cwork"] == NumInternal::WORK_PARAM1 %}
            %csize = {{worksize[0]}}
          {% elsif func_worksize["cwork"] == NumInternal::WORK_PARAM2 %}
            %csize = {{worksize[1]}}
          {% end %}
          %asize += %csize*sizeof(T)
        {% end %}

        {% if T == Complex && func_worksize["rwork"] %}
          {% if func_worksize["rwork"] == NumInternal::WORK_PARAM1 %}
            %rsize = {{worksize[0]}}
          {% elsif func_worksize["rwork"] == NumInternal::WORK_PARAM2 %}
            %rsize = {{worksize[1]}}
          {% end %}
          %asize += %rsize*{% if T == Complex %} sizeof(Float64) {% else %} sizeof(T) {% end %}
        {% end %}

        {% if func_worksize["iwork"] %}
          {% if func_worksize["iwork"] == NumInternal::WORK_PARAM1 %}
            %isize = {{worksize[0]}}
          {% elsif func_worksize["iwork"] == NumInternal::WORK_PARAM2 %}
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
         {% if argtype == NumInternal::ARG_MATRIX %}
           {{arg}},
         {% elsif argtype == NumInternal::ARG_INTOUT %}
           pointerof({{arg}}),
         {% else %}
          pointerof(%var{index}),
         {% end %}
         {% end %}

         {% if func_worksize %}
           {% if func_worksize["cwork"] %}
              %cbuf,
              {% if func_worksize["cwork"] == NumInternal::WORK_DETECT %}
                pointerof(%csize),
              {% end %}
           {% end %}
           {% if T == Complex && func_worksize["rwork"] %}
              %rbuf,
              {% if func_worksize["rwork"] == NumInternal::WORK_DETECT %}
                pointerof(%rsize),
              {% end %}
           {% end %}
           {% if func_worksize["iwork"] %}
              %ibuf,
              {% if func_worksize["iwork"] == NumInternal::WORK_DETECT %}
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
