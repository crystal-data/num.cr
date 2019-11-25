require "./extension"
require "./tensor"
require "./creation"

struct Bottle::Tensor(T)

  macro linalg_nd_self(op)
    def {{op}}(*args, **kwargs)
      if ndims == 2
        {{op}}_(*args, **kwargs)
      else
        a = dup('F')
        a.matrix_iter.each do |subm|
          subm[...] = subm.{{op}}_(*args, **kwargs)
        end
        a
      end
    end
  end

  linalg_nd_self inv

  def cholesky(*, lower = false)
    ret = dup('F')
    char = lower ? 'L' : 'U'
    lapack(potrf, char.ord.to_u8, shape[0], ret.buffer, shape[0])
    lower ? Creation.triu(ret) : Creation.tril(ret)
  end

  private def qr_setup(a, m, n, k)
    tau = Tensor(T).new([k])
    jpvt = Tensor(Int32).new([1])
    lapack(geqrf, m, n, a.buffer, m, tau.buffer)
    {tau, jpvt}
  end

  def qr
    m, n = shape
    k = {m, n}.min
    a = dup('F')
    tau, pvt = qr_setup(a, m, n, k)
    r = Creation.triu(a)
    lapack(orgqr, m, n, k, a.buffer, m, tau.buffer)
    {a, r}
  end

  def svd
    a = dup('F')
    m, n = a.shape
    mn = {m, n}.min
    mx = {m, n}.max
    s = of_real_type(mn)
    u = Tensor(T).new([m, m])
    vt = Tensor(T).new([n, n])
    lapack(gesdd, 'A'.ord.to_u8, m, n, a.buffer, m, s.buffer, u.buffer, m,
      vt.buffer, n, worksize: [{5*mn*mn + 5*mn, 2*mx*mn + 2*mn*mn + mn}.max, 8*mn])
    {u, s, vt}
  end

  def eigh
    a = dup('F')
    n = a.shape[0]
    w = Tensor(T).new([n])
    lapack(syev, 'V'.ord.to_u8, 'L'.ord.to_u8, n, a.buffer, n, w.buffer, worksize: 3 * n - 1)
    {w, a}
  end

  def eig
    a = dup('F')
    n = a.shape[0]
    wr = Tensor(T).new([n])
    wl = wr.dup
    vl = Tensor(T).new([n, n], ArrayFlags::Fortran)
    vr = wr.dup
    lapack(geev, 'V'.ord.to_u8, 'V'.ord.to_u8, n, a.buffer, n, wr.buffer,
      wl.buffer, vl.buffer, n, vr.buffer, n, worksize: 3 * n)
    {wr, vl}
  end

  def eigvalsh
    a = dup('F')
    n = a.shape[0]
    w = Tensor(T).new([n])
    lapack(syev, 'N'.ord.to_u8, 'L'.ord.to_u8, n, a.buffer, n, w.buffer, worksize: 3 * n - 1)
    w
  end

  def eigvals
    a = dup('F')
    n = a.shape[0]
    wr = Tensor(T).new([n])
    wl = wr.dup
    vl = Tensor(T).new([n, n])
    vr = wr.dup
    lapack(geev, 'N'.ord.to_u8, 'N'.ord.to_u8, n, a.buffer, n, wr.buffer,
      wl.buffer, vl.buffer, n, vr.buffer, n, worksize: 3 * n)
    wr
  end

  def norm(*, order = 'F')
    a = dup('F')
    m, n = a.shape
    worksize = order == 'I' ? m : 0
    lapack_util(lange, worksize, order.ord.to_u8, m, m, tensor(a.buffer), m)
  end

  def det
    a = dup('F')
    m, n = a.shape
    ipiv = Pointer(Int32).malloc(n)

    lapack(getrf, m, n, a.buffer, n, ipiv)
    ldet = Statistics.prod(a.diag_view)
    detp = 1
    n.times do |j|
      if j+1 != ipiv[j]
        detp = -detp
      end
    end
    ldet * detp
  end

  protected def inv_
    a = dup('F')
    n = a.shape[0]
    ipiv = Pointer(Int32).malloc(n)
    lapack(getrf, n, n, a.buffer, n, ipiv)
    lapack(getri, n, a.buffer, n, ipiv, worksize: n * n)
    a
  end

  def solve(x : Tensor(T))
    a = dup('F')
    x = x.dup('F')
    n = a.shape[0]
    m = x.ndims > 1 ? x.shape[1] : x.shape[0]
    ipiv = Pointer(Int32).malloc(n)
    lapack(gesv, n, m, a.buffer, n, ipiv, x.buffer, m)
    x
  end

  def hessenberg
    a = dup('F')

    if a.shape[0] < 2
      return a
    end

    m, n = a.shape
    s = of_real_type(n)
    ilo = 0
    ihi = 0
    lapack(gebal, 'B'.ord.to_u8, n, a.buffer, n, ilo, ihi, s.buffer)
    tau = Tensor(T).new([n])
    lapack(gehrd, n, ilo, ihi, a.buffer, n, tau.buffer)
    Creation.triu(a, -1)
  end

  def matmul(other : Tensor(T))
    dest = Tensor(T).new([shape[0], other.shape[1]])
    no = LibCblas::CblasTranspose::CblasNoTrans
    blas(ge, mm, no, no, shape[0], other.shape[1], shape[1], 1.0, buffer, shape[0], other.buffer, other.shape[0], 1.0, dest.buffer, dest.shape[0])
    dest
  end
end
