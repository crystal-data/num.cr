require "./cltensor"

module Num
  extend self

  private def gen_cl_apply3(kern_name : String, ctype : String, op : String) : String
    "
    __kernel void #{kern_name}(__global const #{ctype} *a, __global const #{ctype} *b, __global #{ctype} *c) {
		    int gid = get_global_id(0);
		    c[gid] = a[gid] #{op} b[gid];
	  }
    "
  end

  macro gen_cl_infix_op(dtype, ctype, fn, cname, op)
    def {{fn.id}}(a : ClTensor({{dtype}}), b : ClTensor({{dtype}}))
      result = ClTensor({{dtype}}).new(a.shape)

      cl_kernel = gen_cl_apply3({{cname}}, {{ctype}}, {{op}})
      program = Cl.create_and_build(NumInternal::ClContext.instance.context, cl_kernel, NumInternal::ClContext.instance.device)

      cl_proc = Cl.create_kernel(program, {{cname}})

      Cl.args(cl_proc, a.buffer, b.buffer, result.buffer)
      Cl.run(NumInternal::ClContext.instance.queue, cl_proc, result.size)
      result
    end
  end

  gen_cl_infix_op(Float64, "double", "add", "add_vector", "+")
  gen_cl_infix_op(Float32, "float", "add", "add_vector", "+")
  gen_cl_infix_op(Int32, "int", "add", "add_vector", "+")
  gen_cl_infix_op(UInt8, "uint8", "add", "add_vector", "+")

  gen_cl_infix_op(Float64, "double", "subtract", "subtract_vector", "-")
  gen_cl_infix_op(Float32, "float", "subtract", "subtract_vector", "-")
  gen_cl_infix_op(Int32, "int", "subtract", "subtract_vector", "-")

  private def gen_custom_kernel(kern_name : String, def_string : String, op_string : String) : String
    "
    __kernel void #{kern_name}(#{def_string}) {
		    int gid = get_global_id(0);
		    #{op_string};
	  }
    "
  end

  private def ctypes(dtype : U.class) : String forall U
    case dtype.to_s
    when "Float64"
      "double"
    when "Float32"
      "float"
    when "Int32"
      "int"
    else
      ""
    end
  end

  def runcl(*args, dtype : U.class = Float64) forall U
    ret_shape = [] of Int32
    kernel_op = "ret[gid] = "
    def_string = "__global #{ctypes(dtype)} *ret,"
    args.each_with_index do |arg, index|
      if arg.is_a?(ClTensor(U))
        ret_shape = arg.shape
        kernel_op += " arg_#{index}[gid] "
        def_string += " __global const #{ctypes(dtype)} *arg_#{index},"
      else
        kernel_op += " #{arg} "
      end
    end
    cl_kernel = gen_custom_kernel("custom", def_string[...-1], kernel_op)

    program = Cl.create_and_build(NumInternal::ClContext.instance.context, cl_kernel, NumInternal::ClContext.instance.device)
    cl_proc = Cl.create_kernel(program, "custom")
    ret = ClTensor(U).new(ret_shape)

    index = 0_u32

    Cl.set_arg(cl_proc, ret.buffer, index)
    index += 1_u32

    args.each do |arg|
      if arg.is_a?(ClTensor(U))
        Cl.set_arg(cl_proc, arg.buffer, index)
        index += 1_u32
      end
    end

    Cl.run(NumInternal::ClContext.instance.queue, cl_proc, ret.size)
    ret
  end
end
