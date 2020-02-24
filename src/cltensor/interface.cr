require "./cltensor"

module Num
  extend self

  def gen_cl_apply3(kern_name : String, ctype : String, op : String) : String
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
end
