# Copyright (c) 2020 Crystal Data Contributors
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

module Num::Internal::LapackHelper
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

class Tensor(T, S)
  private macro of_real_type(size)
    {% if T == Complex %}
      Tensor(Float64).new([size])
    {% else %}
      Tensor(T).new([size])
    {% end %}
  end

  private macro alloc_real_type(size)
    {% if T == Float32 %}
      Num::WORK_POOL.get_f32({{size}})
    {% else %}
      Num::WORK_POOL.get_f64({{size}})
    {% end %}
  end

  private macro alloc_type(size)
    {% if T == Complex %}
      Num::WORK_POOL.get_cmplx({{size}})
    {% elsif T == Float32 %}
      Num::WORK_POOL.get_f32({{size}})
    {% else %}
      Num::WORK_POOL.get_f64({{size}})
    {% end %}
  end

  macro lapack_util(name, worksize, *args)
    Num::WORK_POOL.reallocate(worksize*{% if T == Complex %} sizeof(Float64) {% else %} sizeof(T) {% end %})
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
    Num::WORK_POOL.release
    %result
  end

  macro lapack(name, *args, worksize = nil)
      {% unless S < CPU %}
        {% raise "LAPACK is only supported for CPUs" %}
      {% end %}
      {%
        lapack_args = {
          "gbtrf" => {5 => Num::Internal::LapackHelper::ARG_MATRIX, 7 => Num::Internal::LapackHelper::ARG_MATRIX},
          "gebal" => {3 => Num::Internal::LapackHelper::ARG_MATRIX, 5 => Num::Internal::LapackHelper::ARG_INTOUT, 6 => Num::Internal::LapackHelper::ARG_INTOUT, 7 => Num::Internal::LapackHelper::ARG_MATRIX},
          "gees"  => {3 => Num::Internal::LapackHelper::ARG_MATRIX, 5 => Num::Internal::LapackHelper::ARG_MATRIX, 7 => Num::Internal::LapackHelper::ARG_INTOUT, 8 => Num::Internal::LapackHelper::ARG_MATRIX, 9 => Num::Internal::LapackHelper::ARG_MATRIX, 10 => Num::Internal::LapackHelper::ARG_MATRIX},
          "geev"  => {4 => Num::Internal::LapackHelper::ARG_MATRIX, 6 => Num::Internal::LapackHelper::ARG_MATRIX, 7 => Num::Internal::LapackHelper::ARG_MATRIX, 8 => Num::Internal::LapackHelper::ARG_MATRIX, 10 => Num::Internal::LapackHelper::ARG_MATRIX},
          "gehrd" => {4 => Num::Internal::LapackHelper::ARG_MATRIX, 6 => Num::Internal::LapackHelper::ARG_MATRIX},
          "gels"  => {5 => Num::Internal::LapackHelper::ARG_MATRIX, 7 => Num::Internal::LapackHelper::ARG_MATRIX},
          "gelsd" => {4 => Num::Internal::LapackHelper::ARG_MATRIX, 6 => Num::Internal::LapackHelper::ARG_MATRIX, 8 => Num::Internal::LapackHelper::ARG_MATRIX, 10 => Num::Internal::LapackHelper::ARG_INTOUT},
          "gelsy" => {4 => Num::Internal::LapackHelper::ARG_MATRIX, 6 => Num::Internal::LapackHelper::ARG_MATRIX, 8 => Num::Internal::LapackHelper::ARG_MATRIX, 10 => Num::Internal::LapackHelper::ARG_INTOUT},
          "geqp3" => {3 => Num::Internal::LapackHelper::ARG_MATRIX, 5 => Num::Internal::LapackHelper::ARG_MATRIX, 6 => Num::Internal::LapackHelper::ARG_MATRIX},
          "geqrf" => {3 => Num::Internal::LapackHelper::ARG_MATRIX, 5 => Num::Internal::LapackHelper::ARG_MATRIX},
          "gerqf" => {3 => Num::Internal::LapackHelper::ARG_MATRIX, 5 => Num::Internal::LapackHelper::ARG_MATRIX},
          "gelqf" => {3 => Num::Internal::LapackHelper::ARG_MATRIX, 5 => Num::Internal::LapackHelper::ARG_MATRIX},
          "geqlf" => {3 => Num::Internal::LapackHelper::ARG_MATRIX, 5 => Num::Internal::LapackHelper::ARG_MATRIX},
          "gesdd" => {4 => Num::Internal::LapackHelper::ARG_MATRIX, 6 => Num::Internal::LapackHelper::ARG_MATRIX, 7 => Num::Internal::LapackHelper::ARG_MATRIX, 9 => Num::Internal::LapackHelper::ARG_MATRIX},
          "gesv"  => {3 => Num::Internal::LapackHelper::ARG_MATRIX, 5 => Num::Internal::LapackHelper::ARG_MATRIX, 6 => Num::Internal::LapackHelper::ARG_MATRIX},
          "getrf" => {3 => Num::Internal::LapackHelper::ARG_MATRIX, 5 => Num::Internal::LapackHelper::ARG_MATRIX},
          "getri" => {2 => Num::Internal::LapackHelper::ARG_MATRIX, 4 => Num::Internal::LapackHelper::ARG_MATRIX},
          "getrs" => {4 => Num::Internal::LapackHelper::ARG_MATRIX, 6 => Num::Internal::LapackHelper::ARG_MATRIX, 7 => Num::Internal::LapackHelper::ARG_MATRIX},
          "gges"  => {4 => Num::Internal::LapackHelper::ARG_MATRIX, 6 => Num::Internal::LapackHelper::ARG_MATRIX, 8 => Num::Internal::LapackHelper::ARG_MATRIX, 10 => Num::Internal::LapackHelper::ARG_INTOUT, 11 => Num::Internal::LapackHelper::ARG_MATRIX, 12 => Num::Internal::LapackHelper::ARG_MATRIX, 13 => Num::Internal::LapackHelper::ARG_MATRIX, 14 => Num::Internal::LapackHelper::ARG_MATRIX, 16 => Num::Internal::LapackHelper::ARG_MATRIX},
          "ggev"  => {4 => Num::Internal::LapackHelper::ARG_MATRIX, 6 => Num::Internal::LapackHelper::ARG_MATRIX, 8 => Num::Internal::LapackHelper::ARG_MATRIX, 9 => Num::Internal::LapackHelper::ARG_MATRIX, 10 => Num::Internal::LapackHelper::ARG_MATRIX, 11 => Num::Internal::LapackHelper::ARG_MATRIX, 13 => Num::Internal::LapackHelper::ARG_MATRIX},
          "heevr" => {5 => Num::Internal::LapackHelper::ARG_MATRIX, 12 => Num::Internal::LapackHelper::ARG_INTOUT, 13 => Num::Internal::LapackHelper::ARG_MATRIX, 14 => Num::Internal::LapackHelper::ARG_MATRIX, 16 => Num::Internal::LapackHelper::ARG_MATRIX},
          "hegvd" => {5 => Num::Internal::LapackHelper::ARG_MATRIX, 7 => Num::Internal::LapackHelper::ARG_MATRIX, 9 => Num::Internal::LapackHelper::ARG_MATRIX},
          "hesv"  => {4 => Num::Internal::LapackHelper::ARG_MATRIX, 6 => Num::Internal::LapackHelper::ARG_MATRIX, 7 => Num::Internal::LapackHelper::ARG_MATRIX},
          "hetrf" => {3 => Num::Internal::LapackHelper::ARG_MATRIX, 5 => Num::Internal::LapackHelper::ARG_MATRIX},
          "hetri" => {3 => Num::Internal::LapackHelper::ARG_MATRIX, 5 => Num::Internal::LapackHelper::ARG_MATRIX},
          "orghr" => {4 => Num::Internal::LapackHelper::ARG_MATRIX, 6 => Num::Internal::LapackHelper::ARG_MATRIX},
          "orgqr" => {4 => Num::Internal::LapackHelper::ARG_MATRIX, 6 => Num::Internal::LapackHelper::ARG_MATRIX},
          "orgrq" => {4 => Num::Internal::LapackHelper::ARG_MATRIX, 6 => Num::Internal::LapackHelper::ARG_MATRIX},
          "orglq" => {4 => Num::Internal::LapackHelper::ARG_MATRIX, 6 => Num::Internal::LapackHelper::ARG_MATRIX},
          "orgql" => {4 => Num::Internal::LapackHelper::ARG_MATRIX, 6 => Num::Internal::LapackHelper::ARG_MATRIX},
          "posv"  => {4 => Num::Internal::LapackHelper::ARG_MATRIX, 6 => Num::Internal::LapackHelper::ARG_MATRIX},
          "potrf" => {3 => Num::Internal::LapackHelper::ARG_MATRIX},
          "potri" => {3 => Num::Internal::LapackHelper::ARG_MATRIX},
          "potrs" => {4 => Num::Internal::LapackHelper::ARG_MATRIX, 6 => Num::Internal::LapackHelper::ARG_MATRIX},
          "syevr" => {5 => Num::Internal::LapackHelper::ARG_MATRIX, 12 => Num::Internal::LapackHelper::ARG_INTOUT, 13 => Num::Internal::LapackHelper::ARG_MATRIX, 14 => Num::Internal::LapackHelper::ARG_MATRIX, 16 => Num::Internal::LapackHelper::ARG_MATRIX},
          "sygvd" => {5 => Num::Internal::LapackHelper::ARG_MATRIX, 7 => Num::Internal::LapackHelper::ARG_MATRIX, 9 => Num::Internal::LapackHelper::ARG_MATRIX},
          "sysv"  => {4 => Num::Internal::LapackHelper::ARG_MATRIX, 6 => Num::Internal::LapackHelper::ARG_MATRIX, 7 => Num::Internal::LapackHelper::ARG_MATRIX},
          "sytrf" => {3 => Num::Internal::LapackHelper::ARG_MATRIX, 5 => Num::Internal::LapackHelper::ARG_MATRIX},
          "sytri" => {3 => Num::Internal::LapackHelper::ARG_MATRIX, 5 => Num::Internal::LapackHelper::ARG_MATRIX},
          "trtri" => {4 => Num::Internal::LapackHelper::ARG_MATRIX},
          "trtrs" => {6 => Num::Internal::LapackHelper::ARG_MATRIX, 8 => Num::Internal::LapackHelper::ARG_MATRIX},
          "syev"  => {4 => Num::Internal::LapackHelper::ARG_MATRIX, 6 => Num::Internal::LapackHelper::ARG_MATRIX},
          "gecon" => {3 => Num::Internal::LapackHelper::ARG_MATRIX},
        }

        lapack_args_complex = {
          "gees" => {3 => Num::Internal::LapackHelper::ARG_MATRIX, 5 => Num::Internal::LapackHelper::ARG_MATRIX, 7 => Num::Internal::LapackHelper::ARG_INTOUT, 8 => Num::Internal::LapackHelper::ARG_MATRIX, 9 => Num::Internal::LapackHelper::ARG_MATRIX},
          "geev" => {4 => Num::Internal::LapackHelper::ARG_MATRIX, 6 => Num::Internal::LapackHelper::ARG_MATRIX, 7 => Num::Internal::LapackHelper::ARG_MATRIX, 9 => Num::Internal::LapackHelper::ARG_MATRIX},
          "gges" => {4 => Num::Internal::LapackHelper::ARG_MATRIX, 6 => Num::Internal::LapackHelper::ARG_MATRIX, 8 => Num::Internal::LapackHelper::ARG_MATRIX, 10 => Num::Internal::LapackHelper::ARG_INTOUT, 11 => Num::Internal::LapackHelper::ARG_MATRIX, 12 => Num::Internal::LapackHelper::ARG_MATRIX, 13 => Num::Internal::LapackHelper::ARG_MATRIX, 15 => Num::Internal::LapackHelper::ARG_MATRIX},
          "ggev" => {4 => Num::Internal::LapackHelper::ARG_MATRIX, 6 => Num::Internal::LapackHelper::ARG_MATRIX, 8 => Num::Internal::LapackHelper::ARG_MATRIX, 9 => Num::Internal::LapackHelper::ARG_MATRIX, 10 => Num::Internal::LapackHelper::ARG_MATRIX, 12 => Num::Internal::LapackHelper::ARG_MATRIX},
        }

        lapack_worksize = {
          "gees"  => {"cwork" => Num::Internal::LapackHelper::WORK_DETECT, "rwork" => Num::Internal::LapackHelper::WORK_PARAM1, "bwork" => Num::Internal::LapackHelper::WORK_EMPTY},
          "geev"  => {"cwork" => Num::Internal::LapackHelper::WORK_DETECT, "rwork" => Num::Internal::LapackHelper::WORK_PARAM1},
          "gehrd" => {"cwork" => Num::Internal::LapackHelper::WORK_DETECT},
          "gels"  => {"cwork" => Num::Internal::LapackHelper::WORK_DETECT},
          "gelsd" => {"cwork" => Num::Internal::LapackHelper::WORK_DETECT, "rwork" => Num::Internal::LapackHelper::WORK_DETECT_SPECIAL, "iwork" => Num::Internal::LapackHelper::WORK_DETECT_SPECIAL},
          "gelsy" => {"cwork" => Num::Internal::LapackHelper::WORK_DETECT, "rwork" => Num::Internal::LapackHelper::WORK_PARAM1},
          "geqp3" => {"cwork" => Num::Internal::LapackHelper::WORK_DETECT, "rwork" => Num::Internal::LapackHelper::WORK_PARAM1},
          "geqrf" => {"cwork" => Num::Internal::LapackHelper::WORK_DETECT},
          "gerqf" => {"cwork" => Num::Internal::LapackHelper::WORK_DETECT},
          "gelqf" => {"cwork" => Num::Internal::LapackHelper::WORK_DETECT},
          "geqlf" => {"cwork" => Num::Internal::LapackHelper::WORK_DETECT},
          "gesdd" => {"cwork" => Num::Internal::LapackHelper::WORK_DETECT, "rwork" => Num::Internal::LapackHelper::WORK_PARAM1, "iwork" => Num::Internal::LapackHelper::WORK_PARAM2},
          "getri" => {"cwork" => Num::Internal::LapackHelper::WORK_DETECT},
          "gges"  => {"cwork" => Num::Internal::LapackHelper::WORK_DETECT, "rwork" => Num::Internal::LapackHelper::WORK_PARAM1, "bwork" => Num::Internal::LapackHelper::WORK_EMPTY},
          "ggev"  => {"cwork" => Num::Internal::LapackHelper::WORK_DETECT, "rwork" => Num::Internal::LapackHelper::WORK_PARAM1},
          "heevr" => {"cwork" => Num::Internal::LapackHelper::WORK_DETECT, "rwork" => Num::Internal::LapackHelper::WORK_DETECT, "iwork" => Num::Internal::LapackHelper::WORK_DETECT},
          "hegvd" => {"cwork" => Num::Internal::LapackHelper::WORK_DETECT, "rwork" => Num::Internal::LapackHelper::WORK_DETECT, "iwork" => Num::Internal::LapackHelper::WORK_DETECT},
          "hesv"  => {"cwork" => Num::Internal::LapackHelper::WORK_DETECT},
          "hetrf" => {"cwork" => Num::Internal::LapackHelper::WORK_DETECT},
          "hetri" => {"cwork" => Num::Internal::LapackHelper::WORK_PARAM1},
          "orghr" => {"cwork" => Num::Internal::LapackHelper::WORK_DETECT},
          "orgqr" => {"cwork" => Num::Internal::LapackHelper::WORK_DETECT},
          "orgrq" => {"cwork" => Num::Internal::LapackHelper::WORK_DETECT},
          "orglq" => {"cwork" => Num::Internal::LapackHelper::WORK_DETECT},
          "orgql" => {"cwork" => Num::Internal::LapackHelper::WORK_DETECT},
          "syevr" => {"cwork" => Num::Internal::LapackHelper::WORK_DETECT, "iwork" => Num::Internal::LapackHelper::WORK_DETECT},
          "sygvd" => {"cwork" => Num::Internal::LapackHelper::WORK_DETECT, "iwork" => Num::Internal::LapackHelper::WORK_DETECT},
          "sysv"  => {"cwork" => Num::Internal::LapackHelper::WORK_DETECT},
          "sytrf" => {"cwork" => Num::Internal::LapackHelper::WORK_DETECT},
          "sytri" => {"cwork" => Num::Internal::LapackHelper::WORK_PARAM1},
          "syev"  => {"cwork" => Num::Internal::LapackHelper::WORK_DETECT, "rwork" => Num::Internal::LapackHelper::WORK_PARAM1},
          "gecon" => {"cwork" => Num::Internal::LapackHelper::WORK_DETECT, "rwork" => Num::Internal::LapackHelper::WORK_PARAM1},
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
        {% if argtype == Num::Internal::LapackHelper::ARG_MATRIX %}
        {% elsif argtype == Num::Internal::LapackHelper::ARG_INTOUT %}
          {{arg}} = 0
        {% else %}
        %var{index} = {{arg}}
        {% end %}
      {% end %}

      {% if func_worksize && (func_worksize.values.includes?(Num::Internal::LapackHelper::WORK_DETECT) || func_worksize.values.includes?(Num::Internal::LapackHelper::WORK_DETECT_SPECIAL)) %}
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
          {% if argtype == Num::Internal::LapackHelper::ARG_MATRIX %}
            {{arg}},
          {% elsif argtype == Num::Internal::LapackHelper::ARG_INTOUT %}
            pointerof({{arg}}),
          {% else %}
           pointerof(%var{index}),
          {% end %}
          {% end %}

          {% if func_worksize %}
            {% if func_worksize["cwork"] %}
              {% if T == Complex %} pointerof(%cresult).as(LibCblas::ComplexDouble*) {% else %}pointerof(%cresult) {% end %},
               {% if func_worksize["cwork"] == Num::Internal::LapackHelper::WORK_DETECT %}
                 pointerof(%csize),
               {% end %}
            {% end %}
            {% if T == Complex && func_worksize["rwork"] %}
              pointerof(%rresult),
               {% if func_worksize["rwork"] == Num::Internal::LapackHelper::WORK_DETECT %}
                 pointerof(%rsize),
               {% end %}
            {% end %}
            {% if func_worksize["iwork"] %}
              pointerof(%iresult),
               {% if func_worksize["iwork"] == Num::Internal::LapackHelper::WORK_DETECT %}
                 pointerof(%isize),
               {% end %}
            {% end %}
            {% if func_worksize["bwork"] %}
               nil,
            {% end %}
          {% end %}

          pointerof(%info))
         #3. set sizes
         {% if func_worksize["cwork"] == Num::Internal::LapackHelper::WORK_DETECT %}
           %csize = {% if T == Complex %} %cresult.real.to_i {% else %}%cresult.to_i {% end %}
         {% end %}
         {% if T == Complex && func_worksize["rwork"] == Num::Internal::LapackHelper::WORK_DETECT || func_worksize["rwork"] == Num::Internal::LapackHelper::WORK_DETECT_SPECIAL %}
           %rsize = %rresult.to_i
         {% end %}
         {% if func_worksize["iwork"] == Num::Internal::LapackHelper::WORK_DETECT || func_worksize["iwork"] == Num::Internal::LapackHelper::WORK_DETECT_SPECIAL %}
           %isize = %iresult
         {% end %}
      {% end %}


      {% if func_worksize %}
        %asize = 0
        {% if func_worksize["cwork"] %}
          {% if func_worksize["cwork"] == Num::Internal::LapackHelper::WORK_PARAM1 %}
            %csize = {{worksize[0]}}
          {% elsif func_worksize["cwork"] == Num::Internal::LapackHelper::WORK_PARAM2 %}
            %csize = {{worksize[1]}}
          {% end %}
          %asize += %csize*sizeof(T)
        {% end %}

        {% if T == Complex && func_worksize["rwork"] %}
          {% if func_worksize["rwork"] == Num::Internal::LapackHelper::WORK_PARAM1 %}
            %rsize = {{worksize[0]}}
          {% elsif func_worksize["rwork"] == Num::Internal::LapackHelper::WORK_PARAM2 %}
            %rsize = {{worksize[1]}}
          {% end %}
          %asize += %rsize*{% if T == Complex %} sizeof(Float64) {% else %} sizeof(T) {% end %}
        {% end %}

        {% if func_worksize["iwork"] %}
          {% if func_worksize["iwork"] == Num::Internal::LapackHelper::WORK_PARAM1 %}
            %isize = {{worksize[0]}}
          {% elsif func_worksize["iwork"] == Num::Internal::LapackHelper::WORK_PARAM2 %}
            %isize = {{worksize[1]}}
          {% end %}
          %asize += %isize*sizeof(Int32)
        {% end %}

        Num::WORK_POOL.reallocate(%asize)

        {% if func_worksize["cwork"] %}
          %cbuf = alloc_type(%csize)
        {% end %}

        {% if T == Complex && func_worksize["rwork"] %}
          %rbuf = alloc_real_type(%rsize)
        {% end %}

        {% if func_worksize["iwork"] %}
          %ibuf = Num::WORK_POOL.get_i32(%isize)
        {% end %}

      {% end %}

       %info = 0
       LibLapack.{{typ}}{{name}}(
         {% for arg, index in args %}
         {% argtype = func_args[index + 1] %}
         {% if argtype == Num::Internal::LapackHelper::ARG_MATRIX %}
           {{arg}},
         {% elsif argtype == Num::Internal::LapackHelper::ARG_INTOUT %}
           pointerof({{arg}}),
         {% else %}
          pointerof(%var{index}),
         {% end %}
         {% end %}

         {% if func_worksize %}
           {% if func_worksize["cwork"] %}
              %cbuf,
              {% if func_worksize["cwork"] == Num::Internal::LapackHelper::WORK_DETECT %}
                pointerof(%csize),
              {% end %}
           {% end %}
           {% if T == Complex && func_worksize["rwork"] %}
              %rbuf,
              {% if func_worksize["rwork"] == Num::Internal::LapackHelper::WORK_DETECT %}
                pointerof(%rsize),
              {% end %}
           {% end %}
           {% if func_worksize["iwork"] %}
              %ibuf,
              {% if func_worksize["iwork"] == Num::Internal::LapackHelper::WORK_DETECT %}
                pointerof(%isize),
              {% end %}
           {% end %}
           {% if func_worksize["bwork"] %}
              nil,
           {% end %}
         {% end %}

         pointerof(%info))

         {% if func_worksize %}
           Num::WORK_POOL.release
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

  macro blas_call(fn, *args)
    {%
      if T == Float32
        typ = :s.id
      elsif T == Float64
        typ = :d.id
      elsif T == Complex
        typ = :z.id
      end
    %}
    LibCblas.{{typ}}{{fn}}({{*args}})
  end

  private macro blas_const(x)
    {% if T == Complex %}
      pointerof({{x}}).as(LibCblas::ComplexDouble*)
    {% else %}
      {{x}}
    {% end %}
  end

  macro blast(name, *args, prefix = "")
    {%
      if T == Float32
        typ = :S.id
      elsif T == Float64
        typ = :D.id
      end
    %}
    event = Pointer(Void).malloc(1).unsafe_as(LibCL::ClEvent)
    queue = Num::ClContext.instance.queue
    LibBlast.clblast_{{prefix.id}}{{typ}}{{name}}({{*args}}, pointerof(queue), pointerof(event))
    Cl.check LibCL.cl_wait_for_events(1, pointerof(event))
    Cl.check LibCL.cl_release_event(event)
  end
end
