require "opencl"

@[Link("clblast")]
lib LibBlast
  struct ComplexFloat
    re : LibC::Float
    im : LibC::Float
  end

  struct ComplexDouble
    re : LibC::Double
    im : LibC::Double
  end

  enum CLBlastStatusCode
    CLBlastInsufficientMemoryTemp     = -2050
    CLBlastInvalidBatchCount          = -2049
    CLBlastInvalidOverrideKernel      = -2048
    CLBlastMissingOverrideParameter   = -2047
    CLBlastInvalidLocalMemUsage       = -2046
    CLBlastNoHalfPrecision            = -2045
    CLBlastNoDoublePrecision          = -2044
    CLBlastInvalidVectorScalar        = -2043
    CLBlastInsufficientMemoryScalar   = -2042
    CLBlastDatabaseError              = -2041
    CLBlastUnknownError               = -2040
    CLBlastUnexpectedError            = -2039
    CLBlastNotImplemented             = -1024
    CLBlastInvalidMatrixA             = -1022
    CLBlastInvalidMatrixB             = -1021
    CLBlastInvalidMatrixC             = -1020
    CLBlastInvalidVectorX             = -1019
    CLBlastInvalidVectorY             = -1018
    CLBlastInvalidDimension           = -1017
    CLBlastInvalidLeadDimA            = -1016
    CLBlastInvalidLeadDimB            = -1015
    CLBlastInvalidLeadDimC            = -1014
    CLBlastInvalidIncrementX          = -1013
    CLBlastInvalidIncrementY          = -1012
    CLBlastInsufficientMemoryA        = -1011
    CLBlastInsufficientMemoryB        = -1010
    CLBlastInsufficientMemoryC        = -1009
    CLBlastInsufficientMemoryX        = -1008
    CLBlastInsufficientMemoryY        = -1007
    CLBlastInvalidGlobalWorkSize      =   -63
    CLBlastInvalidBufferSize          =   -61
    CLBlastInvalidOperation           =   -59
    CLBlastInvalidEvent               =   -58
    CLBlastInvalidEventWaitList       =   -57
    CLBlastInvalidGlobalOffset        =   -56
    CLBlastInvalidLocalThreadsDim     =   -55
    CLBlastInvalidLocalThreadsTotal   =   -54
    CLBlastInvalidLocalNumDimensions  =   -53
    CLBlastInvalidKernelArgs          =   -52
    CLBlastInvalidArgSize             =   -51
    CLBlastInvalidArgValue            =   -50
    CLBlastInvalidArgIndex            =   -49
    CLBlastInvalidKernel              =   -48
    CLBlastInvalidKernelDefinition    =   -47
    CLBlastInvalidKernelName          =   -46
    CLBlastInvalidProgramExecutable   =   -45
    CLBlastInvalidProgram             =   -44
    CLBlastInvalidBuildOptions        =   -43
    CLBlastInvalidBinary              =   -42
    CLBlastInvalidMemObject           =   -38
    CLBlastInvalidCommandQueue        =   -36
    CLBlastInvalidValue               =   -30
    CLBlastOpenCLBuildProgramFailure  =   -11
    CLBlastOpenCLOutOfHostMemory      =    -6
    CLBlastOpenCLOutOfResources       =    -5
    CLBlastTempBufferAllocFailure     =    -4
    CLBlastOpenCLCompilerNotAvailable =    -3
    CLBlastSuccess                    =     0
  end

  enum CLBlastLayout
    CLBlastLayoutRowMajor = 101
    CLBlastLayoutColMajor = 102
  end

  enum CLBlastTranspose
    CLBlastTransposeNo        = 111
    CLBlastTransposeYes       = 112
    CLBlastTransposeConjugate = 113
  end

  enum CLBlastTriangle
    CLBlastTriangleUpper = 121
    CLBlastTriangleLower = 122
  end

  enum CLBlastDiagonal
    CLBlastDiagonalNonUnit = 131
    CLBlastDiagonalUnit    = 132
  end

  enum CLBlastSide
    CLBlastSideLeft  = 141
    CLBlastSideRight = 142
  end

  enum CLBlastKernelMode
    CLBlastKernelModeCrossCorrelation = 151
    CLBlastKernelModeConvolution      = 152
  end

  enum CLBlastPrecision
    CLBlastPrecisionHalf          =   16
    CLBlastPrecisionSingle        =   32
    CLBlastPrecisionDouble        =   64
    CLBlastPrecisionComplexSingle = 3232
    CLBlastPrecisionComplexDouble = 6464
  end

  fun clblast_Srotg = CLBlastSrotg(sa_buffer : LibCL::ClMem, sa_offset : LibC::SizeT, sb_buffer : LibCL::ClMem,
                                   sb_offset : LibC::SizeT, sc_buffer : LibCL::ClMem, sc_offset : LibC::SizeT,
                                   ss_buffer : LibCL::ClMem, ss_offset : LibC::SizeT, queue : LibCL::ClCommandQueue*,
                                   event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Drotg = CLBlastDrotg(sa_buffer : LibCL::ClMem, sa_offset : LibC::SizeT, sb_buffer : LibCL::ClMem,
                                   sb_offset : LibC::SizeT, sc_buffer : LibCL::ClMem, sc_offset : LibC::SizeT,
                                   ss_buffer : LibCL::ClMem, ss_offset : LibC::SizeT, queue : LibCL::ClCommandQueue*,
                                   event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Srotmg = CLBlastSrotmg(sd1_buffer : LibCL::ClMem, sd1_offset : LibC::SizeT, sd2_buffer : LibCL::ClMem,
                                     sd2_offset : LibC::SizeT, sx1_buffer : LibCL::ClMem, sx1_offset : LibC::SizeT,
                                     sy1_buffer : LibCL::ClMem, sy1_offset : LibC::SizeT, sparam_buffer : LibCL::ClMem,
                                     sparam_offset : LibC::SizeT, queue : LibCL::ClCommandQueue*,
                                     event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Drotmg = CLBlastDrotmg(sd1_buffer : LibCL::ClMem, sd1_offset : LibC::SizeT, sd2_buffer : LibCL::ClMem,
                                     sd2_offset : LibC::SizeT, sx1_buffer : LibCL::ClMem, sx1_offset : LibC::SizeT,
                                     sy1_buffer : LibCL::ClMem, sy1_offset : LibC::SizeT, sparam_buffer : LibCL::ClMem,
                                     sparam_offset : LibC::SizeT, queue : LibCL::ClCommandQueue*,
                                     event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Srot = CLBlastSrot(n : LibC::SizeT, x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT,
                                 y_buffer : LibCL::ClMem, y_offset : LibC::SizeT, y_inc : LibC::SizeT, cos : LibC::Float,
                                 sin : LibC::Float, queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Drot = CLBlastDrot(n : LibC::SizeT, x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT,
                                 y_buffer : LibCL::ClMem, y_offset : LibC::SizeT, y_inc : LibC::SizeT, cos : LibC::Double,
                                 sin : LibC::Double, queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Srotm = CLBlastSrotm(n : LibC::SizeT, x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT,
                                   y_buffer : LibCL::ClMem, y_offset : LibC::SizeT, y_inc : LibC::SizeT,
                                   sparam_buffer : LibCL::ClMem, sparam_offset : LibC::SizeT,
                                   queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Drotm = CLBlastDrotm(n : LibC::SizeT, x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT,
                                   y_buffer : LibCL::ClMem, y_offset : LibC::SizeT, y_inc : LibC::SizeT,
                                   sparam_buffer : LibCL::ClMem, sparam_offset : LibC::SizeT,
                                   queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Sswap = CLBlastSswap(n : LibC::SizeT, x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT,
                                   y_buffer : LibCL::ClMem, y_offset : LibC::SizeT, y_inc : LibC::SizeT,
                                   queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Dswap = CLBlastDswap(n : LibC::SizeT, x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT,
                                   y_buffer : LibCL::ClMem, y_offset : LibC::SizeT, y_inc : LibC::SizeT,
                                   queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Cswap = CLBlastCswap(n : LibC::SizeT, x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT,
                                   y_buffer : LibCL::ClMem, y_offset : LibC::SizeT, y_inc : LibC::SizeT,
                                   queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Zswap = CLBlastZswap(n : LibC::SizeT, x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT,
                                   y_buffer : LibCL::ClMem, y_offset : LibC::SizeT, y_inc : LibC::SizeT,
                                   queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Hswap = CLBlastHswap(n : LibC::SizeT, x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT,
                                   y_buffer : LibCL::ClMem, y_offset : LibC::SizeT, y_inc : LibC::SizeT,
                                   queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Sscal = CLBlastSscal(n : LibC::SizeT, alpha : LibC::Float, x_buffer : LibCL::ClMem, x_offset : LibC::SizeT,
                                   x_inc : LibC::SizeT, queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Dscal = CLBlastDscal(n : LibC::SizeT, alpha : LibC::Double, x_buffer : LibCL::ClMem, x_offset : LibC::SizeT,
                                   x_inc : LibC::SizeT, queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Cscal = CLBlastCscal(n : LibC::SizeT, alpha : ComplexFloat, x_buffer : LibCL::ClMem, x_offset : LibC::SizeT,
                                   x_inc : LibC::SizeT, queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Zscal = CLBlastZscal(n : LibC::SizeT, alpha : ComplexDouble, x_buffer : LibCL::ClMem, x_offset : LibC::SizeT,
                                   x_inc : LibC::SizeT, queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  # fun clblast_Hscal = CLBlastHscal(n: LibC::SizeT, alpha: cl_half, x_buffer: LibCL::ClMem, x_offset: LibC::SizeT,
  #                   x_inc: LibC::SizeT, queue: LibCL::ClCommandQueue*, event: LibCL::ClEvent*): CLBlastStatusCode
  fun clblast_Scopy = CLBlastScopy(n : LibC::SizeT, x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT,
                                   y_buffer : LibCL::ClMem, y_offset : LibC::SizeT, y_inc : LibC::SizeT,
                                   queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Dcopy = CLBlastDcopy(n : LibC::SizeT, x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT,
                                   y_buffer : LibCL::ClMem, y_offset : LibC::SizeT, y_inc : LibC::SizeT,
                                   queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Ccopy = CLBlastCcopy(n : LibC::SizeT, x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT,
                                   y_buffer : LibCL::ClMem, y_offset : LibC::SizeT, y_inc : LibC::SizeT,
                                   queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Zcopy = CLBlastZcopy(n : LibC::SizeT, x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT,
                                   y_buffer : LibCL::ClMem, y_offset : LibC::SizeT, y_inc : LibC::SizeT,
                                   queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Hcopy = CLBlastHcopy(n : LibC::SizeT, x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT,
                                   y_buffer : LibCL::ClMem, y_offset : LibC::SizeT, y_inc : LibC::SizeT,
                                   queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Saxpy = CLBlastSaxpy(n : LibC::SizeT, alpha : LibC::Float, x_buffer : LibCL::ClMem, x_offset : LibC::SizeT,
                                   x_inc : LibC::SizeT, y_buffer : LibCL::ClMem, y_offset : LibC::SizeT, y_inc : LibC::SizeT,
                                   queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Daxpy = CLBlastDaxpy(n : LibC::SizeT, alpha : LibC::Double, x_buffer : LibCL::ClMem, x_offset : LibC::SizeT,
                                   x_inc : LibC::SizeT, y_buffer : LibCL::ClMem, y_offset : LibC::SizeT, y_inc : LibC::SizeT,
                                   queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Caxpy = CLBlastCaxpy(n : LibC::SizeT, alpha : ComplexFloat, x_buffer : LibCL::ClMem, x_offset : LibC::SizeT,
                                   x_inc : LibC::SizeT, y_buffer : LibCL::ClMem, y_offset : LibC::SizeT, y_inc : LibC::SizeT,
                                   queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Zaxpy = CLBlastZaxpy(n : LibC::SizeT, alpha : ComplexDouble, x_buffer : LibCL::ClMem, x_offset : LibC::SizeT,
                                   x_inc : LibC::SizeT, y_buffer : LibCL::ClMem, y_offset : LibC::SizeT, y_inc : LibC::SizeT,
                                   queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  # fun clblast_Haxpy = CLBlastHaxpy(n: LibC::SizeT, alpha: cl_half, x_buffer: LibCL::ClMem, x_offset: LibC::SizeT,
  #                   x_inc: LibC::SizeT, y_buffer: LibCL::ClMem, y_offset: LibC::SizeT, y_inc: LibC::SizeT,
  #                   queue: LibCL::ClCommandQueue*, event: LibCL::ClEvent*): CLBlastStatusCode
  fun clblast_Sdot = CLBlastSdot(n : LibC::SizeT, dot_buffer : LibCL::ClMem, dot_offset : LibC::SizeT, x_buffer : LibCL::ClMem,
                                 x_offset : LibC::SizeT, x_inc : LibC::SizeT, y_buffer : LibCL::ClMem, y_offset : LibC::SizeT,
                                 y_inc : LibC::SizeT, queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Ddot = CLBlastDdot(n : LibC::SizeT, dot_buffer : LibCL::ClMem, dot_offset : LibC::SizeT, x_buffer : LibCL::ClMem,
                                 x_offset : LibC::SizeT, x_inc : LibC::SizeT, y_buffer : LibCL::ClMem, y_offset : LibC::SizeT,
                                 y_inc : LibC::SizeT, queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  # fun clblast_Hdot = CLBlastHdot(n: LibC::SizeT, dot_buffer: LibCL::ClMem, dot_offset: LibC::SizeT, x_buffer: LibCL::ClMem,
  #                  x_offset: LibC::SizeT, x_inc: LibC::SizeT, y_buffer: LibCL::ClMem, y_offset: LibC::SizeT,
  #                  y_inc: LibC::SizeT, queue: LibCL::ClCommandQueue*, event: LibCL::ClEvent*): CLBlastStatusCode
  fun clblast_Cdotu = CLBlastCdotu(n : LibC::SizeT, dot_buffer : LibCL::ClMem, dot_offset : LibC::SizeT, x_buffer : LibCL::ClMem,
                                   x_offset : LibC::SizeT, x_inc : LibC::SizeT, y_buffer : LibCL::ClMem, y_offset : LibC::SizeT,
                                   y_inc : LibC::SizeT, queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Zdotu = CLBlastZdotu(n : LibC::SizeT, dot_buffer : LibCL::ClMem, dot_offset : LibC::SizeT, x_buffer : LibCL::ClMem,
                                   x_offset : LibC::SizeT, x_inc : LibC::SizeT, y_buffer : LibCL::ClMem, y_offset : LibC::SizeT,
                                   y_inc : LibC::SizeT, queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Cdotc = CLBlastCdotc(n : LibC::SizeT, dot_buffer : LibCL::ClMem, dot_offset : LibC::SizeT, x_buffer : LibCL::ClMem,
                                   x_offset : LibC::SizeT, x_inc : LibC::SizeT, y_buffer : LibCL::ClMem, y_offset : LibC::SizeT,
                                   y_inc : LibC::SizeT, queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Zdotc = CLBlastZdotc(n : LibC::SizeT, dot_buffer : LibCL::ClMem, dot_offset : LibC::SizeT, x_buffer : LibCL::ClMem,
                                   x_offset : LibC::SizeT, x_inc : LibC::SizeT, y_buffer : LibCL::ClMem, y_offset : LibC::SizeT,
                                   y_inc : LibC::SizeT, queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Snrm2 = CLBlastSnrm2(n : LibC::SizeT, nrm2_buffer : LibCL::ClMem, nrm2_offset : LibC::SizeT, x_buffer : LibCL::ClMem,
                                   x_offset : LibC::SizeT, x_inc : LibC::SizeT, queue : LibCL::ClCommandQueue*,
                                   event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Dnrm2 = CLBlastDnrm2(n : LibC::SizeT, nrm2_buffer : LibCL::ClMem, nrm2_offset : LibC::SizeT, x_buffer : LibCL::ClMem,
                                   x_offset : LibC::SizeT, x_inc : LibC::SizeT, queue : LibCL::ClCommandQueue*,
                                   event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Scnrm2 = CLBlastScnrm2(n : LibC::SizeT, nrm2_buffer : LibCL::ClMem, nrm2_offset : LibC::SizeT,
                                     x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT,
                                     queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Dznrm2 = CLBlastDznrm2(n : LibC::SizeT, nrm2_buffer : LibCL::ClMem, nrm2_offset : LibC::SizeT,
                                     x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT,
                                     queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  # fun clblast_Hnrm2 = CLBlastHnrm2(n: LibC::SizeT, nrm2_buffer: LibCL::ClMem, nrm2_offset: LibC::SizeT, x_buffer: LibCL::ClMem,
  #                   x_offset: LibC::SizeT, x_inc: LibC::SizeT, queue: LibCL::ClCommandQueue*,
  #                   event: LibCL::ClEvent*): CLBlastStatusCode
  fun clblast_Sasum = CLBlastSasum(n : LibC::SizeT, asum_buffer : LibCL::ClMem, asum_offset : LibC::SizeT, x_buffer : LibCL::ClMem,
                                   x_offset : LibC::SizeT, x_inc : LibC::SizeT, queue : LibCL::ClCommandQueue*,
                                   event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Dasum = CLBlastDasum(n : LibC::SizeT, asum_buffer : LibCL::ClMem, asum_offset : LibC::SizeT, x_buffer : LibCL::ClMem,
                                   x_offset : LibC::SizeT, x_inc : LibC::SizeT, queue : LibCL::ClCommandQueue*,
                                   event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Scasum = CLBlastScasum(n : LibC::SizeT, asum_buffer : LibCL::ClMem, asum_offset : LibC::SizeT,
                                     x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT,
                                     queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Dzasum = CLBlastDzasum(n : LibC::SizeT, asum_buffer : LibCL::ClMem, asum_offset : LibC::SizeT,
                                     x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT,
                                     queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Hasum = CLBlastHasum(n : LibC::SizeT, asum_buffer : LibCL::ClMem, asum_offset : LibC::SizeT, x_buffer : LibCL::ClMem,
                                   x_offset : LibC::SizeT, x_inc : LibC::SizeT, queue : LibCL::ClCommandQueue*,
                                   event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Ssum = CLBlastSsum(n : LibC::SizeT, sum_buffer : LibCL::ClMem, sum_offset : LibC::SizeT, x_buffer : LibCL::ClMem,
                                 x_offset : LibC::SizeT, x_inc : LibC::SizeT, queue : LibCL::ClCommandQueue*,
                                 event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Dsum = CLBlastDsum(n : LibC::SizeT, sum_buffer : LibCL::ClMem, sum_offset : LibC::SizeT, x_buffer : LibCL::ClMem,
                                 x_offset : LibC::SizeT, x_inc : LibC::SizeT, queue : LibCL::ClCommandQueue*,
                                 event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Scsum = CLBlastScsum(n : LibC::SizeT, sum_buffer : LibCL::ClMem, sum_offset : LibC::SizeT, x_buffer : LibCL::ClMem,
                                   x_offset : LibC::SizeT, x_inc : LibC::SizeT, queue : LibCL::ClCommandQueue*,
                                   event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Dzsum = CLBlastDzsum(n : LibC::SizeT, sum_buffer : LibCL::ClMem, sum_offset : LibC::SizeT, x_buffer : LibCL::ClMem,
                                   x_offset : LibC::SizeT, x_inc : LibC::SizeT, queue : LibCL::ClCommandQueue*,
                                   event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Hsum = CLBlastHsum(n : LibC::SizeT, sum_buffer : LibCL::ClMem, sum_offset : LibC::SizeT, x_buffer : LibCL::ClMem,
                                 x_offset : LibC::SizeT, x_inc : LibC::SizeT, queue : LibCL::ClCommandQueue*,
                                 event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_iSamax = CLBlastiSamax(n : LibC::SizeT, imax_buffer : LibCL::ClMem, imax_offset : LibC::SizeT,
                                     x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT,
                                     queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_iDamax = CLBlastiDamax(n : LibC::SizeT, imax_buffer : LibCL::ClMem, imax_offset : LibC::SizeT,
                                     x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT,
                                     queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_iCamax = CLBlastiCamax(n : LibC::SizeT, imax_buffer : LibCL::ClMem, imax_offset : LibC::SizeT,
                                     x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT,
                                     queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_iZamax = CLBlastiZamax(n : LibC::SizeT, imax_buffer : LibCL::ClMem, imax_offset : LibC::SizeT,
                                     x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT,
                                     queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_iHamax = CLBlastiHamax(n : LibC::SizeT, imax_buffer : LibCL::ClMem, imax_offset : LibC::SizeT,
                                     x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT,
                                     queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_iSamin = CLBlastiSamin(n : LibC::SizeT, imin_buffer : LibCL::ClMem, imin_offset : LibC::SizeT,
                                     x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT,
                                     queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_iDamin = CLBlastiDamin(n : LibC::SizeT, imin_buffer : LibCL::ClMem, imin_offset : LibC::SizeT,
                                     x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT,
                                     queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_iCamin = CLBlastiCamin(n : LibC::SizeT, imin_buffer : LibCL::ClMem, imin_offset : LibC::SizeT,
                                     x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT,
                                     queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_iZamin = CLBlastiZamin(n : LibC::SizeT, imin_buffer : LibCL::ClMem, imin_offset : LibC::SizeT,
                                     x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT,
                                     queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_iHamin = CLBlastiHamin(n : LibC::SizeT, imin_buffer : LibCL::ClMem, imin_offset : LibC::SizeT,
                                     x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT,
                                     queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_iSmax = CLBlastiSmax(n : LibC::SizeT, imax_buffer : LibCL::ClMem, imax_offset : LibC::SizeT, x_buffer : LibCL::ClMem,
                                   x_offset : LibC::SizeT, x_inc : LibC::SizeT, queue : LibCL::ClCommandQueue*,
                                   event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_iDmax = CLBlastiDmax(n : LibC::SizeT, imax_buffer : LibCL::ClMem, imax_offset : LibC::SizeT, x_buffer : LibCL::ClMem,
                                   x_offset : LibC::SizeT, x_inc : LibC::SizeT, queue : LibCL::ClCommandQueue*,
                                   event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_iCmax = CLBlastiCmax(n : LibC::SizeT, imax_buffer : LibCL::ClMem, imax_offset : LibC::SizeT, x_buffer : LibCL::ClMem,
                                   x_offset : LibC::SizeT, x_inc : LibC::SizeT, queue : LibCL::ClCommandQueue*,
                                   event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_iZmax = CLBlastiZmax(n : LibC::SizeT, imax_buffer : LibCL::ClMem, imax_offset : LibC::SizeT, x_buffer : LibCL::ClMem,
                                   x_offset : LibC::SizeT, x_inc : LibC::SizeT, queue : LibCL::ClCommandQueue*,
                                   event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_iHmax = CLBlastiHmax(n : LibC::SizeT, imax_buffer : LibCL::ClMem, imax_offset : LibC::SizeT, x_buffer : LibCL::ClMem,
                                   x_offset : LibC::SizeT, x_inc : LibC::SizeT, queue : LibCL::ClCommandQueue*,
                                   event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_iSmin = CLBlastiSmin(n : LibC::SizeT, imin_buffer : LibCL::ClMem, imin_offset : LibC::SizeT, x_buffer : LibCL::ClMem,
                                   x_offset : LibC::SizeT, x_inc : LibC::SizeT, queue : LibCL::ClCommandQueue*,
                                   event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_iDmin = CLBlastiDmin(n : LibC::SizeT, imin_buffer : LibCL::ClMem, imin_offset : LibC::SizeT, x_buffer : LibCL::ClMem,
                                   x_offset : LibC::SizeT, x_inc : LibC::SizeT, queue : LibCL::ClCommandQueue*,
                                   event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_iCmin = CLBlastiCmin(n : LibC::SizeT, imin_buffer : LibCL::ClMem, imin_offset : LibC::SizeT, x_buffer : LibCL::ClMem,
                                   x_offset : LibC::SizeT, x_inc : LibC::SizeT, queue : LibCL::ClCommandQueue*,
                                   event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_iZmin = CLBlastiZmin(n : LibC::SizeT, imin_buffer : LibCL::ClMem, imin_offset : LibC::SizeT, x_buffer : LibCL::ClMem,
                                   x_offset : LibC::SizeT, x_inc : LibC::SizeT, queue : LibCL::ClCommandQueue*,
                                   event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_iHmin = CLBlastiHmin(n : LibC::SizeT, imin_buffer : LibCL::ClMem, imin_offset : LibC::SizeT, x_buffer : LibCL::ClMem,
                                   x_offset : LibC::SizeT, x_inc : LibC::SizeT, queue : LibCL::ClCommandQueue*,
                                   event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Sgemv = CLBlastSgemv(layout : CLBlastLayout, a_transpose : CLBlastTranspose, m : LibC::SizeT,
                                   n : LibC::SizeT, alpha : LibC::Float, a_buffer : LibCL::ClMem, a_offset : LibC::SizeT,
                                   a_ld : LibC::SizeT, x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT,
                                   beta : LibC::Float, y_buffer : LibCL::ClMem, y_offset : LibC::SizeT, y_inc : LibC::SizeT,
                                   queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Dgemv = CLBlastDgemv(layout : CLBlastLayout, a_transpose : CLBlastTranspose, m : LibC::SizeT,
                                   n : LibC::SizeT, alpha : LibC::Double, a_buffer : LibCL::ClMem, a_offset : LibC::SizeT,
                                   a_ld : LibC::SizeT, x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT,
                                   beta : LibC::Double, y_buffer : LibCL::ClMem, y_offset : LibC::SizeT, y_inc : LibC::SizeT,
                                   queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Cgemv = CLBlastCgemv(layout : CLBlastLayout, a_transpose : CLBlastTranspose, m : LibC::SizeT,
                                   n : LibC::SizeT, alpha : ComplexFloat, a_buffer : LibCL::ClMem, a_offset : LibC::SizeT,
                                   a_ld : LibC::SizeT, x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT,
                                   beta : ComplexFloat, y_buffer : LibCL::ClMem, y_offset : LibC::SizeT, y_inc : LibC::SizeT,
                                   queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Zgemv = CLBlastZgemv(layout : CLBlastLayout, a_transpose : CLBlastTranspose, m : LibC::SizeT,
                                   n : LibC::SizeT, alpha : ComplexDouble, a_buffer : LibCL::ClMem, a_offset : LibC::SizeT,
                                   a_ld : LibC::SizeT, x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT,
                                   beta : ComplexDouble, y_buffer : LibCL::ClMem, y_offset : LibC::SizeT, y_inc : LibC::SizeT,
                                   queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  # fun clblast_Hgemv = CLBlastHgemv(layout: CLBlastLayout, a_transpose: CLBlastTranspose, m: LibC::SizeT,
  #                   n: LibC::SizeT, alpha: cl_half, a_buffer: LibCL::ClMem, a_offset: LibC::SizeT,
  #                   a_ld: LibC::SizeT, x_buffer: LibCL::ClMem, x_offset: LibC::SizeT, x_inc: LibC::SizeT,
  #                   beta: cl_half, y_buffer: LibCL::ClMem, y_offset: LibC::SizeT, y_inc: LibC::SizeT,
  #                   queue: LibCL::ClCommandQueue*, event: LibCL::ClEvent*): CLBlastStatusCode
  fun clblast_Sgbmv = CLBlastSgbmv(layout : CLBlastLayout, a_transpose : CLBlastTranspose, m : LibC::SizeT,
                                   n : LibC::SizeT, kl : LibC::SizeT, ku : LibC::SizeT, alpha : LibC::Float, a_buffer : LibCL::ClMem,
                                   a_offset : LibC::SizeT, a_ld : LibC::SizeT, x_buffer : LibCL::ClMem, x_offset : LibC::SizeT,
                                   x_inc : LibC::SizeT, beta : LibC::Float, y_buffer : LibCL::ClMem, y_offset : LibC::SizeT,
                                   y_inc : LibC::SizeT, queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Dgbmv = CLBlastDgbmv(layout : CLBlastLayout, a_transpose : CLBlastTranspose, m : LibC::SizeT,
                                   n : LibC::SizeT, kl : LibC::SizeT, ku : LibC::SizeT, alpha : LibC::Double, a_buffer : LibCL::ClMem,
                                   a_offset : LibC::SizeT, a_ld : LibC::SizeT, x_buffer : LibCL::ClMem, x_offset : LibC::SizeT,
                                   x_inc : LibC::SizeT, beta : LibC::Double, y_buffer : LibCL::ClMem, y_offset : LibC::SizeT,
                                   y_inc : LibC::SizeT, queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Cgbmv = CLBlastCgbmv(layout : CLBlastLayout, a_transpose : CLBlastTranspose, m : LibC::SizeT,
                                   n : LibC::SizeT, kl : LibC::SizeT, ku : LibC::SizeT, alpha : ComplexFloat, a_buffer : LibCL::ClMem,
                                   a_offset : LibC::SizeT, a_ld : LibC::SizeT, x_buffer : LibCL::ClMem, x_offset : LibC::SizeT,
                                   x_inc : LibC::SizeT, beta : ComplexFloat, y_buffer : LibCL::ClMem, y_offset : LibC::SizeT,
                                   y_inc : LibC::SizeT, queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Zgbmv = CLBlastZgbmv(layout : CLBlastLayout, a_transpose : CLBlastTranspose, m : LibC::SizeT,
                                   n : LibC::SizeT, kl : LibC::SizeT, ku : LibC::SizeT, alpha : ComplexDouble, a_buffer : LibCL::ClMem,
                                   a_offset : LibC::SizeT, a_ld : LibC::SizeT, x_buffer : LibCL::ClMem, x_offset : LibC::SizeT,
                                   x_inc : LibC::SizeT, beta : ComplexDouble, y_buffer : LibCL::ClMem, y_offset : LibC::SizeT,
                                   y_inc : LibC::SizeT, queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  # fun clblast_Hgbmv = CLBlastHgbmv(layout: CLBlastLayout, a_transpose: CLBlastTranspose, m: LibC::SizeT,
  #                   n: LibC::SizeT, kl: LibC::SizeT, ku: LibC::SizeT, alpha: cl_half, a_buffer: LibCL::ClMem,
  #                   a_offset: LibC::SizeT, a_ld: LibC::SizeT, x_buffer: LibCL::ClMem, x_offset: LibC::SizeT,
  #                   x_inc: LibC::SizeT, beta: cl_half, y_buffer: LibCL::ClMem, y_offset: LibC::SizeT,
  #                   y_inc: LibC::SizeT, queue: LibCL::ClCommandQueue*, event: LibCL::ClEvent*): CLBlastStatusCode
  fun clblast_Chemv = CLBlastChemv(layout : CLBlastLayout, triangle : CLBlastTriangle, n : LibC::SizeT,
                                   alpha : ComplexFloat, a_buffer : LibCL::ClMem, a_offset : LibC::SizeT, a_ld : LibC::SizeT,
                                   x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT, beta : ComplexFloat,
                                   y_buffer : LibCL::ClMem, y_offset : LibC::SizeT, y_inc : LibC::SizeT,
                                   queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Zhemv = CLBlastZhemv(layout : CLBlastLayout, triangle : CLBlastTriangle, n : LibC::SizeT,
                                   alpha : ComplexDouble, a_buffer : LibCL::ClMem, a_offset : LibC::SizeT, a_ld : LibC::SizeT,
                                   x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT, beta : ComplexDouble,
                                   y_buffer : LibCL::ClMem, y_offset : LibC::SizeT, y_inc : LibC::SizeT,
                                   queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Chbmv = CLBlastChbmv(layout : CLBlastLayout, triangle : CLBlastTriangle, n : LibC::SizeT,
                                   k : LibC::SizeT, alpha : ComplexFloat, a_buffer : LibCL::ClMem, a_offset : LibC::SizeT,
                                   a_ld : LibC::SizeT, x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT,
                                   beta : ComplexFloat, y_buffer : LibCL::ClMem, y_offset : LibC::SizeT, y_inc : LibC::SizeT,
                                   queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Zhbmv = CLBlastZhbmv(layout : CLBlastLayout, triangle : CLBlastTriangle, n : LibC::SizeT,
                                   k : LibC::SizeT, alpha : ComplexDouble, a_buffer : LibCL::ClMem, a_offset : LibC::SizeT,
                                   a_ld : LibC::SizeT, x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT,
                                   beta : ComplexDouble, y_buffer : LibCL::ClMem, y_offset : LibC::SizeT, y_inc : LibC::SizeT,
                                   queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Chpmv = CLBlastChpmv(layout : CLBlastLayout, triangle : CLBlastTriangle, n : LibC::SizeT,
                                   alpha : ComplexFloat, ap_buffer : LibCL::ClMem, ap_offset : LibC::SizeT,
                                   x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT, beta : ComplexFloat,
                                   y_buffer : LibCL::ClMem, y_offset : LibC::SizeT, y_inc : LibC::SizeT,
                                   queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Zhpmv = CLBlastZhpmv(layout : CLBlastLayout, triangle : CLBlastTriangle, n : LibC::SizeT,
                                   alpha : ComplexDouble, ap_buffer : LibCL::ClMem, ap_offset : LibC::SizeT,
                                   x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT, beta : ComplexDouble,
                                   y_buffer : LibCL::ClMem, y_offset : LibC::SizeT, y_inc : LibC::SizeT,
                                   queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Ssymv = CLBlastSsymv(layout : CLBlastLayout, triangle : CLBlastTriangle, n : LibC::SizeT,
                                   alpha : LibC::Float, a_buffer : LibCL::ClMem, a_offset : LibC::SizeT, a_ld : LibC::SizeT,
                                   x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT, beta : LibC::Float,
                                   y_buffer : LibCL::ClMem, y_offset : LibC::SizeT, y_inc : LibC::SizeT,
                                   queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Dsymv = CLBlastDsymv(layout : CLBlastLayout, triangle : CLBlastTriangle, n : LibC::SizeT,
                                   alpha : LibC::Double, a_buffer : LibCL::ClMem, a_offset : LibC::SizeT, a_ld : LibC::SizeT,
                                   x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT, beta : LibC::Double,
                                   y_buffer : LibCL::ClMem, y_offset : LibC::SizeT, y_inc : LibC::SizeT,
                                   queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  # fun clblast_Hsymv = CLBlastHsymv(layout: CLBlastLayout, triangle: CLBlastTriangle, n: LibC::SizeT,
  #                   alpha: cl_half, a_buffer: LibCL::ClMem, a_offset: LibC::SizeT, a_ld: LibC::SizeT,
  #                   x_buffer: LibCL::ClMem, x_offset: LibC::SizeT, x_inc: LibC::SizeT, beta: cl_half,
  #                   y_buffer: LibCL::ClMem, y_offset: LibC::SizeT, y_inc: LibC::SizeT,
  #                   queue: LibCL::ClCommandQueue*, event: LibCL::ClEvent*): CLBlastStatusCode
  fun clblast_Ssbmv = CLBlastSsbmv(layout : CLBlastLayout, triangle : CLBlastTriangle, n : LibC::SizeT,
                                   k : LibC::SizeT, alpha : LibC::Float, a_buffer : LibCL::ClMem, a_offset : LibC::SizeT,
                                   a_ld : LibC::SizeT, x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT,
                                   beta : LibC::Float, y_buffer : LibCL::ClMem, y_offset : LibC::SizeT, y_inc : LibC::SizeT,
                                   queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Dsbmv = CLBlastDsbmv(layout : CLBlastLayout, triangle : CLBlastTriangle, n : LibC::SizeT,
                                   k : LibC::SizeT, alpha : LibC::Double, a_buffer : LibCL::ClMem, a_offset : LibC::SizeT,
                                   a_ld : LibC::SizeT, x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT,
                                   beta : LibC::Double, y_buffer : LibCL::ClMem, y_offset : LibC::SizeT, y_inc : LibC::SizeT,
                                   queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  # fun clblast_Hsbmv = CLBlastHsbmv(layout: CLBlastLayout, triangle: CLBlastTriangle, n: LibC::SizeT,
  #                   k: LibC::SizeT, alpha: cl_half, a_buffer: LibCL::ClMem, a_offset: LibC::SizeT,
  #                   a_ld: LibC::SizeT, x_buffer: LibCL::ClMem, x_offset: LibC::SizeT, x_inc: LibC::SizeT,
  #                   beta: cl_half, y_buffer: LibCL::ClMem, y_offset: LibC::SizeT, y_inc: LibC::SizeT,
  #                   queue: LibCL::ClCommandQueue*, event: LibCL::ClEvent*): CLBlastStatusCode
  fun clblast_Sspmv = CLBlastSspmv(layout : CLBlastLayout, triangle : CLBlastTriangle, n : LibC::SizeT,
                                   alpha : LibC::Float, ap_buffer : LibCL::ClMem, ap_offset : LibC::SizeT,
                                   x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT, beta : LibC::Float,
                                   y_buffer : LibCL::ClMem, y_offset : LibC::SizeT, y_inc : LibC::SizeT,
                                   queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Dspmv = CLBlastDspmv(layout : CLBlastLayout, triangle : CLBlastTriangle, n : LibC::SizeT,
                                   alpha : LibC::Double, ap_buffer : LibCL::ClMem, ap_offset : LibC::SizeT,
                                   x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT, beta : LibC::Double,
                                   y_buffer : LibCL::ClMem, y_offset : LibC::SizeT, y_inc : LibC::SizeT,
                                   queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  # fun clblast_Hspmv = CLBlastHspmv(layout: CLBlastLayout, triangle: CLBlastTriangle, n: LibC::SizeT,
  #                   alpha: cl_half, ap_buffer: LibCL::ClMem, ap_offset: LibC::SizeT,
  #                   x_buffer: LibCL::ClMem, x_offset: LibC::SizeT, x_inc: LibC::SizeT, beta: cl_half,
  #                   y_buffer: LibCL::ClMem, y_offset: LibC::SizeT, y_inc: LibC::SizeT,
  #                   queue: LibCL::ClCommandQueue*, event: LibCL::ClEvent*): CLBlastStatusCode
  fun clblast_Strmv = CLBlastStrmv(layout : CLBlastLayout, triangle : CLBlastTriangle,
                                   a_transpose : CLBlastTranspose, diagonal : CLBlastDiagonal,
                                   n : LibC::SizeT, a_buffer : LibCL::ClMem, a_offset : LibC::SizeT, a_ld : LibC::SizeT,
                                   x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT,
                                   queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Dtrmv = CLBlastDtrmv(layout : CLBlastLayout, triangle : CLBlastTriangle,
                                   a_transpose : CLBlastTranspose, diagonal : CLBlastDiagonal,
                                   n : LibC::SizeT, a_buffer : LibCL::ClMem, a_offset : LibC::SizeT, a_ld : LibC::SizeT,
                                   x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT,
                                   queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Ctrmv = CLBlastCtrmv(layout : CLBlastLayout, triangle : CLBlastTriangle,
                                   a_transpose : CLBlastTranspose, diagonal : CLBlastDiagonal,
                                   n : LibC::SizeT, a_buffer : LibCL::ClMem, a_offset : LibC::SizeT, a_ld : LibC::SizeT,
                                   x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT,
                                   queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Ztrmv = CLBlastZtrmv(layout : CLBlastLayout, triangle : CLBlastTriangle,
                                   a_transpose : CLBlastTranspose, diagonal : CLBlastDiagonal,
                                   n : LibC::SizeT, a_buffer : LibCL::ClMem, a_offset : LibC::SizeT, a_ld : LibC::SizeT,
                                   x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT,
                                   queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  # fun clblast_Htrmv = CLBlastHtrmv(layout: CLBlastLayout, triangle: CLBlastTriangle,
  #                   a_transpose: CLBlastTranspose, diagonal: CLBlastDiagonal,
  #                   n: LibC::SizeT, a_buffer: LibCL::ClMem, a_offset: LibC::SizeT, a_ld: LibC::SizeT,
  #                   x_buffer: LibCL::ClMem, x_offset: LibC::SizeT, x_inc: LibC::SizeT,
  #                   queue: LibCL::ClCommandQueue*, event: LibCL::ClEvent*): CLBlastStatusCode
  fun clblast_Stbmv = CLBlastStbmv(layout : CLBlastLayout, triangle : CLBlastTriangle,
                                   a_transpose : CLBlastTranspose, diagonal : CLBlastDiagonal,
                                   n : LibC::SizeT, k : LibC::SizeT, a_buffer : LibCL::ClMem, a_offset : LibC::SizeT, a_ld : LibC::SizeT,
                                   x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT,
                                   queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Dtbmv = CLBlastDtbmv(layout : CLBlastLayout, triangle : CLBlastTriangle,
                                   a_transpose : CLBlastTranspose, diagonal : CLBlastDiagonal,
                                   n : LibC::SizeT, k : LibC::SizeT, a_buffer : LibCL::ClMem, a_offset : LibC::SizeT, a_ld : LibC::SizeT,
                                   x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT,
                                   queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Ctbmv = CLBlastCtbmv(layout : CLBlastLayout, triangle : CLBlastTriangle,
                                   a_transpose : CLBlastTranspose, diagonal : CLBlastDiagonal,
                                   n : LibC::SizeT, k : LibC::SizeT, a_buffer : LibCL::ClMem, a_offset : LibC::SizeT, a_ld : LibC::SizeT,
                                   x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT,
                                   queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Ztbmv = CLBlastZtbmv(layout : CLBlastLayout, triangle : CLBlastTriangle,
                                   a_transpose : CLBlastTranspose, diagonal : CLBlastDiagonal,
                                   n : LibC::SizeT, k : LibC::SizeT, a_buffer : LibCL::ClMem, a_offset : LibC::SizeT, a_ld : LibC::SizeT,
                                   x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT,
                                   queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  # fun clblast_Htbmv = CLBlastHtbmv(layout: CLBlastLayout, triangle: CLBlastTriangle,
  #                   a_transpose: CLBlastTranspose, diagonal: CLBlastDiagonal,
  #                   n: LibC::SizeT, k: LibC::SizeT, a_buffer: LibCL::ClMem, a_offset: LibC::SizeT, a_ld: LibC::SizeT,
  #                   x_buffer: LibCL::ClMem, x_offset: LibC::SizeT, x_inc: LibC::SizeT,
  #                   queue: LibCL::ClCommandQueue*, event: LibCL::ClEvent*): CLBlastStatusCode
  fun clblast_Stpmv = CLBlastStpmv(layout : CLBlastLayout, triangle : CLBlastTriangle,
                                   a_transpose : CLBlastTranspose, diagonal : CLBlastDiagonal,
                                   n : LibC::SizeT, ap_buffer : LibCL::ClMem, ap_offset : LibC::SizeT, x_buffer : LibCL::ClMem,
                                   x_offset : LibC::SizeT, x_inc : LibC::SizeT, queue : LibCL::ClCommandQueue*,
                                   event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Dtpmv = CLBlastDtpmv(layout : CLBlastLayout, triangle : CLBlastTriangle,
                                   a_transpose : CLBlastTranspose, diagonal : CLBlastDiagonal,
                                   n : LibC::SizeT, ap_buffer : LibCL::ClMem, ap_offset : LibC::SizeT, x_buffer : LibCL::ClMem,
                                   x_offset : LibC::SizeT, x_inc : LibC::SizeT, queue : LibCL::ClCommandQueue*,
                                   event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Ctpmv = CLBlastCtpmv(layout : CLBlastLayout, triangle : CLBlastTriangle,
                                   a_transpose : CLBlastTranspose, diagonal : CLBlastDiagonal,
                                   n : LibC::SizeT, ap_buffer : LibCL::ClMem, ap_offset : LibC::SizeT, x_buffer : LibCL::ClMem,
                                   x_offset : LibC::SizeT, x_inc : LibC::SizeT, queue : LibCL::ClCommandQueue*,
                                   event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Ztpmv = CLBlastZtpmv(layout : CLBlastLayout, triangle : CLBlastTriangle,
                                   a_transpose : CLBlastTranspose, diagonal : CLBlastDiagonal,
                                   n : LibC::SizeT, ap_buffer : LibCL::ClMem, ap_offset : LibC::SizeT, x_buffer : LibCL::ClMem,
                                   x_offset : LibC::SizeT, x_inc : LibC::SizeT, queue : LibCL::ClCommandQueue*,
                                   event : LibCL::ClEvent*) : CLBlastStatusCode
  # fun clblast_Htpmv = CLBlastHtpmv(layout: CLBlastLayout, triangle: CLBlastTriangle,
  #                   a_transpose: CLBlastTranspose, diagonal: CLBlastDiagonal,
  #                   n: LibC::SizeT, ap_buffer: LibCL::ClMem, ap_offset: LibC::SizeT, x_buffer: LibCL::ClMem,
  #                   x_offset: LibC::SizeT, x_inc: LibC::SizeT, queue: LibCL::ClCommandQueue*,
  #                   event: LibCL::ClEvent*): CLBlastStatusCode
  fun clblast_Strsv = CLBlastStrsv(layout : CLBlastLayout, triangle : CLBlastTriangle,
                                   a_transpose : CLBlastTranspose, diagonal : CLBlastDiagonal,
                                   n : LibC::SizeT, a_buffer : LibCL::ClMem, a_offset : LibC::SizeT, a_ld : LibC::SizeT,
                                   x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT,
                                   queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Dtrsv = CLBlastDtrsv(layout : CLBlastLayout, triangle : CLBlastTriangle,
                                   a_transpose : CLBlastTranspose, diagonal : CLBlastDiagonal,
                                   n : LibC::SizeT, a_buffer : LibCL::ClMem, a_offset : LibC::SizeT, a_ld : LibC::SizeT,
                                   x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT,
                                   queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Ctrsv = CLBlastCtrsv(layout : CLBlastLayout, triangle : CLBlastTriangle,
                                   a_transpose : CLBlastTranspose, diagonal : CLBlastDiagonal,
                                   n : LibC::SizeT, a_buffer : LibCL::ClMem, a_offset : LibC::SizeT, a_ld : LibC::SizeT,
                                   x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT,
                                   queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Ztrsv = CLBlastZtrsv(layout : CLBlastLayout, triangle : CLBlastTriangle,
                                   a_transpose : CLBlastTranspose, diagonal : CLBlastDiagonal,
                                   n : LibC::SizeT, a_buffer : LibCL::ClMem, a_offset : LibC::SizeT, a_ld : LibC::SizeT,
                                   x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT,
                                   queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Stbsv = CLBlastStbsv(layout : CLBlastLayout, triangle : CLBlastTriangle,
                                   a_transpose : CLBlastTranspose, diagonal : CLBlastDiagonal,
                                   n : LibC::SizeT, k : LibC::SizeT, a_buffer : LibCL::ClMem, a_offset : LibC::SizeT, a_ld : LibC::SizeT,
                                   x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT,
                                   queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Dtbsv = CLBlastDtbsv(layout : CLBlastLayout, triangle : CLBlastTriangle,
                                   a_transpose : CLBlastTranspose, diagonal : CLBlastDiagonal,
                                   n : LibC::SizeT, k : LibC::SizeT, a_buffer : LibCL::ClMem, a_offset : LibC::SizeT, a_ld : LibC::SizeT,
                                   x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT,
                                   queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Ctbsv = CLBlastCtbsv(layout : CLBlastLayout, triangle : CLBlastTriangle,
                                   a_transpose : CLBlastTranspose, diagonal : CLBlastDiagonal,
                                   n : LibC::SizeT, k : LibC::SizeT, a_buffer : LibCL::ClMem, a_offset : LibC::SizeT, a_ld : LibC::SizeT,
                                   x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT,
                                   queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Ztbsv = CLBlastZtbsv(layout : CLBlastLayout, triangle : CLBlastTriangle,
                                   a_transpose : CLBlastTranspose, diagonal : CLBlastDiagonal,
                                   n : LibC::SizeT, k : LibC::SizeT, a_buffer : LibCL::ClMem, a_offset : LibC::SizeT, a_ld : LibC::SizeT,
                                   x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT,
                                   queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Stpsv = CLBlastStpsv(layout : CLBlastLayout, triangle : CLBlastTriangle,
                                   a_transpose : CLBlastTranspose, diagonal : CLBlastDiagonal,
                                   n : LibC::SizeT, ap_buffer : LibCL::ClMem, ap_offset : LibC::SizeT, x_buffer : LibCL::ClMem,
                                   x_offset : LibC::SizeT, x_inc : LibC::SizeT, queue : LibCL::ClCommandQueue*,
                                   event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Dtpsv = CLBlastDtpsv(layout : CLBlastLayout, triangle : CLBlastTriangle,
                                   a_transpose : CLBlastTranspose, diagonal : CLBlastDiagonal,
                                   n : LibC::SizeT, ap_buffer : LibCL::ClMem, ap_offset : LibC::SizeT, x_buffer : LibCL::ClMem,
                                   x_offset : LibC::SizeT, x_inc : LibC::SizeT, queue : LibCL::ClCommandQueue*,
                                   event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Ctpsv = CLBlastCtpsv(layout : CLBlastLayout, triangle : CLBlastTriangle,
                                   a_transpose : CLBlastTranspose, diagonal : CLBlastDiagonal,
                                   n : LibC::SizeT, ap_buffer : LibCL::ClMem, ap_offset : LibC::SizeT, x_buffer : LibCL::ClMem,
                                   x_offset : LibC::SizeT, x_inc : LibC::SizeT, queue : LibCL::ClCommandQueue*,
                                   event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Ztpsv = CLBlastZtpsv(layout : CLBlastLayout, triangle : CLBlastTriangle,
                                   a_transpose : CLBlastTranspose, diagonal : CLBlastDiagonal,
                                   n : LibC::SizeT, ap_buffer : LibCL::ClMem, ap_offset : LibC::SizeT, x_buffer : LibCL::ClMem,
                                   x_offset : LibC::SizeT, x_inc : LibC::SizeT, queue : LibCL::ClCommandQueue*,
                                   event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Sger = CLBlastSger(layout : CLBlastLayout, m : LibC::SizeT, n : LibC::SizeT, alpha : LibC::Float,
                                 x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT, y_buffer : LibCL::ClMem,
                                 y_offset : LibC::SizeT, y_inc : LibC::SizeT, a_buffer : LibCL::ClMem, a_offset : LibC::SizeT,
                                 a_ld : LibC::SizeT, queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Dger = CLBlastDger(layout : CLBlastLayout, m : LibC::SizeT, n : LibC::SizeT, alpha : LibC::Double,
                                 x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT, y_buffer : LibCL::ClMem,
                                 y_offset : LibC::SizeT, y_inc : LibC::SizeT, a_buffer : LibCL::ClMem, a_offset : LibC::SizeT,
                                 a_ld : LibC::SizeT, queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  # fun clblast_Hger = CLBlastHger(layout: CLBlastLayout, m: LibC::SizeT, n: LibC::SizeT, alpha: cl_half,
  #                  x_buffer: LibCL::ClMem, x_offset: LibC::SizeT, x_inc: LibC::SizeT, y_buffer: LibCL::ClMem,
  #                  y_offset: LibC::SizeT, y_inc: LibC::SizeT, a_buffer: LibCL::ClMem, a_offset: LibC::SizeT,
  #                  a_ld: LibC::SizeT, queue: LibCL::ClCommandQueue*, event: LibCL::ClEvent*): CLBlastStatusCode
  fun clblast_Cgeru = CLBlastCgeru(layout : CLBlastLayout, m : LibC::SizeT, n : LibC::SizeT, alpha : ComplexFloat,
                                   x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT, y_buffer : LibCL::ClMem,
                                   y_offset : LibC::SizeT, y_inc : LibC::SizeT, a_buffer : LibCL::ClMem, a_offset : LibC::SizeT,
                                   a_ld : LibC::SizeT, queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Zgeru = CLBlastZgeru(layout : CLBlastLayout, m : LibC::SizeT, n : LibC::SizeT, alpha : ComplexDouble,
                                   x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT, y_buffer : LibCL::ClMem,
                                   y_offset : LibC::SizeT, y_inc : LibC::SizeT, a_buffer : LibCL::ClMem, a_offset : LibC::SizeT,
                                   a_ld : LibC::SizeT, queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Cgerc = CLBlastCgerc(layout : CLBlastLayout, m : LibC::SizeT, n : LibC::SizeT, alpha : ComplexFloat,
                                   x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT, y_buffer : LibCL::ClMem,
                                   y_offset : LibC::SizeT, y_inc : LibC::SizeT, a_buffer : LibCL::ClMem, a_offset : LibC::SizeT,
                                   a_ld : LibC::SizeT, queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Zgerc = CLBlastZgerc(layout : CLBlastLayout, m : LibC::SizeT, n : LibC::SizeT, alpha : ComplexDouble,
                                   x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT, y_buffer : LibCL::ClMem,
                                   y_offset : LibC::SizeT, y_inc : LibC::SizeT, a_buffer : LibCL::ClMem, a_offset : LibC::SizeT,
                                   a_ld : LibC::SizeT, queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Cher = CLBlastCher(layout : CLBlastLayout, triangle : CLBlastTriangle, n : LibC::SizeT,
                                 alpha : LibC::Float, x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT,
                                 a_buffer : LibCL::ClMem, a_offset : LibC::SizeT, a_ld : LibC::SizeT,
                                 queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Zher = CLBlastZher(layout : CLBlastLayout, triangle : CLBlastTriangle, n : LibC::SizeT,
                                 alpha : LibC::Double, x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT,
                                 a_buffer : LibCL::ClMem, a_offset : LibC::SizeT, a_ld : LibC::SizeT,
                                 queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Chpr = CLBlastChpr(layout : CLBlastLayout, triangle : CLBlastTriangle, n : LibC::SizeT,
                                 alpha : LibC::Float, x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT,
                                 ap_buffer : LibCL::ClMem, ap_offset : LibC::SizeT, queue : LibCL::ClCommandQueue*,
                                 event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Zhpr = CLBlastZhpr(layout : CLBlastLayout, triangle : CLBlastTriangle, n : LibC::SizeT,
                                 alpha : LibC::Double, x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT,
                                 ap_buffer : LibCL::ClMem, ap_offset : LibC::SizeT, queue : LibCL::ClCommandQueue*,
                                 event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Cher2 = CLBlastCher2(layout : CLBlastLayout, triangle : CLBlastTriangle, n : LibC::SizeT,
                                   alpha : ComplexFloat, x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT,
                                   y_buffer : LibCL::ClMem, y_offset : LibC::SizeT, y_inc : LibC::SizeT, a_buffer : LibCL::ClMem,
                                   a_offset : LibC::SizeT, a_ld : LibC::SizeT, queue : LibCL::ClCommandQueue*,
                                   event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Zher2 = CLBlastZher2(layout : CLBlastLayout, triangle : CLBlastTriangle, n : LibC::SizeT,
                                   alpha : ComplexDouble, x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT,
                                   y_buffer : LibCL::ClMem, y_offset : LibC::SizeT, y_inc : LibC::SizeT, a_buffer : LibCL::ClMem,
                                   a_offset : LibC::SizeT, a_ld : LibC::SizeT, queue : LibCL::ClCommandQueue*,
                                   event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Chpr2 = CLBlastChpr2(layout : CLBlastLayout, triangle : CLBlastTriangle, n : LibC::SizeT,
                                   alpha : ComplexFloat, x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT,
                                   y_buffer : LibCL::ClMem, y_offset : LibC::SizeT, y_inc : LibC::SizeT, ap_buffer : LibCL::ClMem,
                                   ap_offset : LibC::SizeT, queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Zhpr2 = CLBlastZhpr2(layout : CLBlastLayout, triangle : CLBlastTriangle, n : LibC::SizeT,
                                   alpha : ComplexDouble, x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT,
                                   y_buffer : LibCL::ClMem, y_offset : LibC::SizeT, y_inc : LibC::SizeT, ap_buffer : LibCL::ClMem,
                                   ap_offset : LibC::SizeT, queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Ssyr = CLBlastSsyr(layout : CLBlastLayout, triangle : CLBlastTriangle, n : LibC::SizeT,
                                 alpha : LibC::Float, x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT,
                                 a_buffer : LibCL::ClMem, a_offset : LibC::SizeT, a_ld : LibC::SizeT,
                                 queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Dsyr = CLBlastDsyr(layout : CLBlastLayout, triangle : CLBlastTriangle, n : LibC::SizeT,
                                 alpha : LibC::Double, x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT,
                                 a_buffer : LibCL::ClMem, a_offset : LibC::SizeT, a_ld : LibC::SizeT,
                                 queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  # fun clblast_Hsyr = CLBlastHsyr(layout: CLBlastLayout, triangle: CLBlastTriangle, n: LibC::SizeT,
  #                  alpha: cl_half, x_buffer: LibCL::ClMem, x_offset: LibC::SizeT, x_inc: LibC::SizeT,
  #                  a_buffer: LibCL::ClMem, a_offset: LibC::SizeT, a_ld: LibC::SizeT,
  #                  queue: LibCL::ClCommandQueue*, event: LibCL::ClEvent*): CLBlastStatusCode
  fun clblast_Sspr = CLBlastSspr(layout : CLBlastLayout, triangle : CLBlastTriangle, n : LibC::SizeT,
                                 alpha : LibC::Float, x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT,
                                 ap_buffer : LibCL::ClMem, ap_offset : LibC::SizeT, queue : LibCL::ClCommandQueue*,
                                 event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Dspr = CLBlastDspr(layout : CLBlastLayout, triangle : CLBlastTriangle, n : LibC::SizeT,
                                 alpha : LibC::Double, x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT,
                                 ap_buffer : LibCL::ClMem, ap_offset : LibC::SizeT, queue : LibCL::ClCommandQueue*,
                                 event : LibCL::ClEvent*) : CLBlastStatusCode
  # fun clblast_Hspr = CLBlastHspr(layout: CLBlastLayout, triangle: CLBlastTriangle, n: LibC::SizeT,
  #                  alpha: cl_half, x_buffer: LibCL::ClMem, x_offset: LibC::SizeT, x_inc: LibC::SizeT,
  #                  ap_buffer: LibCL::ClMem, ap_offset: LibC::SizeT, queue: LibCL::ClCommandQueue*,
  #                  event: LibCL::ClEvent*): CLBlastStatusCode
  fun clblast_Ssyr2 = CLBlastSsyr2(layout : CLBlastLayout, triangle : CLBlastTriangle, n : LibC::SizeT,
                                   alpha : LibC::Float, x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT,
                                   y_buffer : LibCL::ClMem, y_offset : LibC::SizeT, y_inc : LibC::SizeT, a_buffer : LibCL::ClMem,
                                   a_offset : LibC::SizeT, a_ld : LibC::SizeT, queue : LibCL::ClCommandQueue*,
                                   event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Dsyr2 = CLBlastDsyr2(layout : CLBlastLayout, triangle : CLBlastTriangle, n : LibC::SizeT,
                                   alpha : LibC::Double, x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT,
                                   y_buffer : LibCL::ClMem, y_offset : LibC::SizeT, y_inc : LibC::SizeT, a_buffer : LibCL::ClMem,
                                   a_offset : LibC::SizeT, a_ld : LibC::SizeT, queue : LibCL::ClCommandQueue*,
                                   event : LibCL::ClEvent*) : CLBlastStatusCode
  # fun clblast_Hsyr2 = CLBlastHsyr2(layout: CLBlastLayout, triangle: CLBlastTriangle, n: LibC::SizeT,
  #                   alpha: cl_half, x_buffer: LibCL::ClMem, x_offset: LibC::SizeT, x_inc: LibC::SizeT,
  #                   y_buffer: LibCL::ClMem, y_offset: LibC::SizeT, y_inc: LibC::SizeT, a_buffer: LibCL::ClMem,
  #                   a_offset: LibC::SizeT, a_ld: LibC::SizeT, queue: LibCL::ClCommandQueue*,
  #                   event: LibCL::ClEvent*): CLBlastStatusCode
  fun clblast_Sspr2 = CLBlastSspr2(layout : CLBlastLayout, triangle : CLBlastTriangle, n : LibC::SizeT,
                                   alpha : LibC::Float, x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT,
                                   y_buffer : LibCL::ClMem, y_offset : LibC::SizeT, y_inc : LibC::SizeT, ap_buffer : LibCL::ClMem,
                                   ap_offset : LibC::SizeT, queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Dspr2 = CLBlastDspr2(layout : CLBlastLayout, triangle : CLBlastTriangle, n : LibC::SizeT,
                                   alpha : LibC::Double, x_buffer : LibCL::ClMem, x_offset : LibC::SizeT, x_inc : LibC::SizeT,
                                   y_buffer : LibCL::ClMem, y_offset : LibC::SizeT, y_inc : LibC::SizeT, ap_buffer : LibCL::ClMem,
                                   ap_offset : LibC::SizeT, queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  # fun clblast_Hspr2 = CLBlastHspr2(layout: CLBlastLayout, triangle: CLBlastTriangle, n: LibC::SizeT,
  #                   alpha: cl_half, x_buffer: LibCL::ClMem, x_offset: LibC::SizeT, x_inc: LibC::SizeT,
  #                   y_buffer: LibCL::ClMem, y_offset: LibC::SizeT, y_inc: LibC::SizeT, ap_buffer: LibCL::ClMem,
  #                   ap_offset: LibC::SizeT, queue: LibCL::ClCommandQueue*, event: LibCL::ClEvent*): CLBlastStatusCode
  fun clblast_Sgemm = CLBlastSgemm(layout : CLBlastLayout, a_transpose : CLBlastTranspose,
                                   b_transpose : CLBlastTranspose, m : LibC::SizeT, n : LibC::SizeT, k : LibC::SizeT,
                                   alpha : LibC::Float, a_buffer : LibCL::ClMem, a_offset : LibC::SizeT, a_ld : LibC::SizeT,
                                   b_buffer : LibCL::ClMem, b_offset : LibC::SizeT, b_ld : LibC::SizeT, beta : LibC::Float,
                                   c_buffer : LibCL::ClMem, c_offset : LibC::SizeT, c_ld : LibC::SizeT,
                                   queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Dgemm = CLBlastDgemm(layout : CLBlastLayout, a_transpose : CLBlastTranspose,
                                   b_transpose : CLBlastTranspose, m : LibC::SizeT, n : LibC::SizeT, k : LibC::SizeT,
                                   alpha : LibC::Double, a_buffer : LibCL::ClMem, a_offset : LibC::SizeT, a_ld : LibC::SizeT,
                                   b_buffer : LibCL::ClMem, b_offset : LibC::SizeT, b_ld : LibC::SizeT, beta : LibC::Double,
                                   c_buffer : LibCL::ClMem, c_offset : LibC::SizeT, c_ld : LibC::SizeT,
                                   queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Cgemm = CLBlastCgemm(layout : CLBlastLayout, a_transpose : CLBlastTranspose,
                                   b_transpose : CLBlastTranspose, m : LibC::SizeT, n : LibC::SizeT, k : LibC::SizeT,
                                   alpha : ComplexFloat, a_buffer : LibCL::ClMem, a_offset : LibC::SizeT, a_ld : LibC::SizeT,
                                   b_buffer : LibCL::ClMem, b_offset : LibC::SizeT, b_ld : LibC::SizeT, beta : ComplexFloat,
                                   c_buffer : LibCL::ClMem, c_offset : LibC::SizeT, c_ld : LibC::SizeT,
                                   queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Zgemm = CLBlastZgemm(layout : CLBlastLayout, a_transpose : CLBlastTranspose,
                                   b_transpose : CLBlastTranspose, m : LibC::SizeT, n : LibC::SizeT, k : LibC::SizeT,
                                   alpha : ComplexDouble, a_buffer : LibCL::ClMem, a_offset : LibC::SizeT, a_ld : LibC::SizeT,
                                   b_buffer : LibCL::ClMem, b_offset : LibC::SizeT, b_ld : LibC::SizeT, beta : ComplexDouble,
                                   c_buffer : LibCL::ClMem, c_offset : LibC::SizeT, c_ld : LibC::SizeT,
                                   queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  # fun clblast_Hgemm = CLBlastHgemm(layout: CLBlastLayout, a_transpose: CLBlastTranspose,
  #                   b_transpose: CLBlastTranspose, m: LibC::SizeT, n: LibC::SizeT, k: LibC::SizeT,
  #                   alpha: cl_half, a_buffer: LibCL::ClMem, a_offset: LibC::SizeT, a_ld: LibC::SizeT,
  #                   b_buffer: LibCL::ClMem, b_offset: LibC::SizeT, b_ld: LibC::SizeT, beta: cl_half,
  #                   c_buffer: LibCL::ClMem, c_offset: LibC::SizeT, c_ld: LibC::SizeT,
  #                   queue: LibCL::ClCommandQueue*, event: LibCL::ClEvent*): CLBlastStatusCode
  fun clblast_Ssymm = CLBlastSsymm(layout : CLBlastLayout, side : CLBlastSide,
                                   triangle : CLBlastTriangle, m : LibC::SizeT, n : LibC::SizeT, alpha : LibC::Float,
                                   a_buffer : LibCL::ClMem, a_offset : LibC::SizeT, a_ld : LibC::SizeT, b_buffer : LibCL::ClMem,
                                   b_offset : LibC::SizeT, b_ld : LibC::SizeT, beta : LibC::Float, c_buffer : LibCL::ClMem,
                                   c_offset : LibC::SizeT, c_ld : LibC::SizeT, queue : LibCL::ClCommandQueue*,
                                   event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Dsymm = CLBlastDsymm(layout : CLBlastLayout, side : CLBlastSide,
                                   triangle : CLBlastTriangle, m : LibC::SizeT, n : LibC::SizeT, alpha : LibC::Double,
                                   a_buffer : LibCL::ClMem, a_offset : LibC::SizeT, a_ld : LibC::SizeT, b_buffer : LibCL::ClMem,
                                   b_offset : LibC::SizeT, b_ld : LibC::SizeT, beta : LibC::Double, c_buffer : LibCL::ClMem,
                                   c_offset : LibC::SizeT, c_ld : LibC::SizeT, queue : LibCL::ClCommandQueue*,
                                   event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Csymm = CLBlastCsymm(layout : CLBlastLayout, side : CLBlastSide,
                                   triangle : CLBlastTriangle, m : LibC::SizeT, n : LibC::SizeT, alpha : ComplexFloat,
                                   a_buffer : LibCL::ClMem, a_offset : LibC::SizeT, a_ld : LibC::SizeT, b_buffer : LibCL::ClMem,
                                   b_offset : LibC::SizeT, b_ld : LibC::SizeT, beta : ComplexFloat, c_buffer : LibCL::ClMem,
                                   c_offset : LibC::SizeT, c_ld : LibC::SizeT, queue : LibCL::ClCommandQueue*,
                                   event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Zsymm = CLBlastZsymm(layout : CLBlastLayout, side : CLBlastSide,
                                   triangle : CLBlastTriangle, m : LibC::SizeT, n : LibC::SizeT, alpha : ComplexDouble,
                                   a_buffer : LibCL::ClMem, a_offset : LibC::SizeT, a_ld : LibC::SizeT, b_buffer : LibCL::ClMem,
                                   b_offset : LibC::SizeT, b_ld : LibC::SizeT, beta : ComplexDouble, c_buffer : LibCL::ClMem,
                                   c_offset : LibC::SizeT, c_ld : LibC::SizeT, queue : LibCL::ClCommandQueue*,
                                   event : LibCL::ClEvent*) : CLBlastStatusCode
  # fun clblast_Hsymm = CLBlastHsymm(layout: CLBlastLayout, side: CLBlastSide,
  #                   triangle: CLBlastTriangle, m: LibC::SizeT, n: LibC::SizeT, alpha: cl_half,
  #                   a_buffer: LibCL::ClMem, a_offset: LibC::SizeT, a_ld: LibC::SizeT, b_buffer: LibCL::ClMem,
  #                   b_offset: LibC::SizeT, b_ld: LibC::SizeT, beta: cl_half, c_buffer: LibCL::ClMem,
  #                   c_offset: LibC::SizeT, c_ld: LibC::SizeT, queue: LibCL::ClCommandQueue*,
  #                   event: LibCL::ClEvent*): CLBlastStatusCode
  fun clblast_Chemm = CLBlastChemm(layout : CLBlastLayout, side : CLBlastSide,
                                   triangle : CLBlastTriangle, m : LibC::SizeT, n : LibC::SizeT, alpha : ComplexFloat,
                                   a_buffer : LibCL::ClMem, a_offset : LibC::SizeT, a_ld : LibC::SizeT, b_buffer : LibCL::ClMem,
                                   b_offset : LibC::SizeT, b_ld : LibC::SizeT, beta : ComplexFloat, c_buffer : LibCL::ClMem,
                                   c_offset : LibC::SizeT, c_ld : LibC::SizeT, queue : LibCL::ClCommandQueue*,
                                   event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Zhemm = CLBlastZhemm(layout : CLBlastLayout, side : CLBlastSide,
                                   triangle : CLBlastTriangle, m : LibC::SizeT, n : LibC::SizeT, alpha : ComplexDouble,
                                   a_buffer : LibCL::ClMem, a_offset : LibC::SizeT, a_ld : LibC::SizeT, b_buffer : LibCL::ClMem,
                                   b_offset : LibC::SizeT, b_ld : LibC::SizeT, beta : ComplexDouble, c_buffer : LibCL::ClMem,
                                   c_offset : LibC::SizeT, c_ld : LibC::SizeT, queue : LibCL::ClCommandQueue*,
                                   event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Ssyrk = CLBlastSsyrk(layout : CLBlastLayout, triangle : CLBlastTriangle,
                                   a_transpose : CLBlastTranspose, n : LibC::SizeT, k : LibC::SizeT, alpha : LibC::Float,
                                   a_buffer : LibCL::ClMem, a_offset : LibC::SizeT, a_ld : LibC::SizeT, beta : LibC::Float,
                                   c_buffer : LibCL::ClMem, c_offset : LibC::SizeT, c_ld : LibC::SizeT,
                                   queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Dsyrk = CLBlastDsyrk(layout : CLBlastLayout, triangle : CLBlastTriangle,
                                   a_transpose : CLBlastTranspose, n : LibC::SizeT, k : LibC::SizeT, alpha : LibC::Double,
                                   a_buffer : LibCL::ClMem, a_offset : LibC::SizeT, a_ld : LibC::SizeT, beta : LibC::Double,
                                   c_buffer : LibCL::ClMem, c_offset : LibC::SizeT, c_ld : LibC::SizeT,
                                   queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Csyrk = CLBlastCsyrk(layout : CLBlastLayout, triangle : CLBlastTriangle,
                                   a_transpose : CLBlastTranspose, n : LibC::SizeT, k : LibC::SizeT, alpha : ComplexFloat,
                                   a_buffer : LibCL::ClMem, a_offset : LibC::SizeT, a_ld : LibC::SizeT, beta : ComplexFloat,
                                   c_buffer : LibCL::ClMem, c_offset : LibC::SizeT, c_ld : LibC::SizeT,
                                   queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Zsyrk = CLBlastZsyrk(layout : CLBlastLayout, triangle : CLBlastTriangle,
                                   a_transpose : CLBlastTranspose, n : LibC::SizeT, k : LibC::SizeT,
                                   alpha : ComplexDouble, a_buffer : LibCL::ClMem, a_offset : LibC::SizeT, a_ld : LibC::SizeT,
                                   beta : ComplexDouble, c_buffer : LibCL::ClMem, c_offset : LibC::SizeT, c_ld : LibC::SizeT,
                                   queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  # fun clblast_Hsyrk = CLBlastHsyrk(layout: CLBlastLayout, triangle: CLBlastTriangle,
  #                   a_transpose: CLBlastTranspose, n: LibC::SizeT, k: LibC::SizeT, alpha: cl_half,
  #                   a_buffer: LibCL::ClMem, a_offset: LibC::SizeT, a_ld: LibC::SizeT, beta: cl_half,
  #                   c_buffer: LibCL::ClMem, c_offset: LibC::SizeT, c_ld: LibC::SizeT,
  #                   queue: LibCL::ClCommandQueue*, event: LibCL::ClEvent*): CLBlastStatusCode
  fun clblast_Cherk = CLBlastCherk(layout : CLBlastLayout, triangle : CLBlastTriangle,
                                   a_transpose : CLBlastTranspose, n : LibC::SizeT, k : LibC::SizeT, alpha : LibC::Float,
                                   a_buffer : LibCL::ClMem, a_offset : LibC::SizeT, a_ld : LibC::SizeT, beta : LibC::Float,
                                   c_buffer : LibCL::ClMem, c_offset : LibC::SizeT, c_ld : LibC::SizeT,
                                   queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Zherk = CLBlastZherk(layout : CLBlastLayout, triangle : CLBlastTriangle,
                                   a_transpose : CLBlastTranspose, n : LibC::SizeT, k : LibC::SizeT, alpha : LibC::Double,
                                   a_buffer : LibCL::ClMem, a_offset : LibC::SizeT, a_ld : LibC::SizeT, beta : LibC::Double,
                                   c_buffer : LibCL::ClMem, c_offset : LibC::SizeT, c_ld : LibC::SizeT,
                                   queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Ssyr2k = CLBlastSsyr2k(layout : CLBlastLayout, triangle : CLBlastTriangle,
                                     ab_transpose : CLBlastTranspose, n : LibC::SizeT, k : LibC::SizeT, alpha : LibC::Float,
                                     a_buffer : LibCL::ClMem, a_offset : LibC::SizeT, a_ld : LibC::SizeT, b_buffer : LibCL::ClMem,
                                     b_offset : LibC::SizeT, b_ld : LibC::SizeT, beta : LibC::Float, c_buffer : LibCL::ClMem,
                                     c_offset : LibC::SizeT, c_ld : LibC::SizeT, queue : LibCL::ClCommandQueue*,
                                     event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Dsyr2k = CLBlastDsyr2k(layout : CLBlastLayout, triangle : CLBlastTriangle,
                                     ab_transpose : CLBlastTranspose, n : LibC::SizeT, k : LibC::SizeT, alpha : LibC::Double,
                                     a_buffer : LibCL::ClMem, a_offset : LibC::SizeT, a_ld : LibC::SizeT, b_buffer : LibCL::ClMem,
                                     b_offset : LibC::SizeT, b_ld : LibC::SizeT, beta : LibC::Double, c_buffer : LibCL::ClMem,
                                     c_offset : LibC::SizeT, c_ld : LibC::SizeT, queue : LibCL::ClCommandQueue*,
                                     event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Csyr2k = CLBlastCsyr2k(layout : CLBlastLayout, triangle : CLBlastTriangle,
                                     ab_transpose : CLBlastTranspose, n : LibC::SizeT, k : LibC::SizeT,
                                     alpha : ComplexFloat, a_buffer : LibCL::ClMem, a_offset : LibC::SizeT, a_ld : LibC::SizeT,
                                     b_buffer : LibCL::ClMem, b_offset : LibC::SizeT, b_ld : LibC::SizeT, beta : ComplexFloat,
                                     c_buffer : LibCL::ClMem, c_offset : LibC::SizeT, c_ld : LibC::SizeT,
                                     queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Zsyr2k = CLBlastZsyr2k(layout : CLBlastLayout, triangle : CLBlastTriangle,
                                     ab_transpose : CLBlastTranspose, n : LibC::SizeT, k : LibC::SizeT,
                                     alpha : ComplexDouble, a_buffer : LibCL::ClMem, a_offset : LibC::SizeT, a_ld : LibC::SizeT,
                                     b_buffer : LibCL::ClMem, b_offset : LibC::SizeT, b_ld : LibC::SizeT, beta : ComplexDouble,
                                     c_buffer : LibCL::ClMem, c_offset : LibC::SizeT, c_ld : LibC::SizeT,
                                     queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  # fun clblast_Hsyr2k = CLBlastHsyr2k(layout: CLBlastLayout, triangle: CLBlastTriangle,
  #                    ab_transpose: CLBlastTranspose, n: LibC::SizeT, k: LibC::SizeT, alpha: cl_half,
  #                    a_buffer: LibCL::ClMem, a_offset: LibC::SizeT, a_ld: LibC::SizeT, b_buffer: LibCL::ClMem,
  #                    b_offset: LibC::SizeT, b_ld: LibC::SizeT, beta: cl_half, c_buffer: LibCL::ClMem,
  #                    c_offset: LibC::SizeT, c_ld: LibC::SizeT, queue: LibCL::ClCommandQueue*,
  #                    event: LibCL::ClEvent*): CLBlastStatusCode
  fun clblast_Cher2k = CLBlastCher2k(layout : CLBlastLayout, triangle : CLBlastTriangle,
                                     ab_transpose : CLBlastTranspose, n : LibC::SizeT, k : LibC::SizeT,
                                     alpha : ComplexFloat, a_buffer : LibCL::ClMem, a_offset : LibC::SizeT, a_ld : LibC::SizeT,
                                     b_buffer : LibCL::ClMem, b_offset : LibC::SizeT, b_ld : LibC::SizeT, beta : LibC::Float,
                                     c_buffer : LibCL::ClMem, c_offset : LibC::SizeT, c_ld : LibC::SizeT,
                                     queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Zher2k = CLBlastZher2k(layout : CLBlastLayout, triangle : CLBlastTriangle,
                                     ab_transpose : CLBlastTranspose, n : LibC::SizeT, k : LibC::SizeT,
                                     alpha : ComplexDouble, a_buffer : LibCL::ClMem, a_offset : LibC::SizeT, a_ld : LibC::SizeT,
                                     b_buffer : LibCL::ClMem, b_offset : LibC::SizeT, b_ld : LibC::SizeT, beta : LibC::Double,
                                     c_buffer : LibCL::ClMem, c_offset : LibC::SizeT, c_ld : LibC::SizeT,
                                     queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Strmm = CLBlastStrmm(layout : CLBlastLayout, side : CLBlastSide,
                                   triangle : CLBlastTriangle, a_transpose : CLBlastTranspose,
                                   diagonal : CLBlastDiagonal, m : LibC::SizeT, n : LibC::SizeT, alpha : LibC::Float,
                                   a_buffer : LibCL::ClMem, a_offset : LibC::SizeT, a_ld : LibC::SizeT, b_buffer : LibCL::ClMem,
                                   b_offset : LibC::SizeT, b_ld : LibC::SizeT, queue : LibCL::ClCommandQueue*,
                                   event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Dtrmm = CLBlastDtrmm(layout : CLBlastLayout, side : CLBlastSide,
                                   triangle : CLBlastTriangle, a_transpose : CLBlastTranspose,
                                   diagonal : CLBlastDiagonal, m : LibC::SizeT, n : LibC::SizeT, alpha : LibC::Double,
                                   a_buffer : LibCL::ClMem, a_offset : LibC::SizeT, a_ld : LibC::SizeT, b_buffer : LibCL::ClMem,
                                   b_offset : LibC::SizeT, b_ld : LibC::SizeT, queue : LibCL::ClCommandQueue*,
                                   event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Ctrmm = CLBlastCtrmm(layout : CLBlastLayout, side : CLBlastSide,
                                   triangle : CLBlastTriangle, a_transpose : CLBlastTranspose,
                                   diagonal : CLBlastDiagonal, m : LibC::SizeT, n : LibC::SizeT, alpha : ComplexFloat,
                                   a_buffer : LibCL::ClMem, a_offset : LibC::SizeT, a_ld : LibC::SizeT, b_buffer : LibCL::ClMem,
                                   b_offset : LibC::SizeT, b_ld : LibC::SizeT, queue : LibCL::ClCommandQueue*,
                                   event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Ztrmm = CLBlastZtrmm(layout : CLBlastLayout, side : CLBlastSide,
                                   triangle : CLBlastTriangle, a_transpose : CLBlastTranspose,
                                   diagonal : CLBlastDiagonal, m : LibC::SizeT, n : LibC::SizeT, alpha : ComplexDouble,
                                   a_buffer : LibCL::ClMem, a_offset : LibC::SizeT, a_ld : LibC::SizeT, b_buffer : LibCL::ClMem,
                                   b_offset : LibC::SizeT, b_ld : LibC::SizeT, queue : LibCL::ClCommandQueue*,
                                   event : LibCL::ClEvent*) : CLBlastStatusCode
  # fun clblast_Htrmm = CLBlastHtrmm(layout: CLBlastLayout, side: CLBlastSide,
  #                   triangle: CLBlastTriangle, a_transpose: CLBlastTranspose,
  #                   diagonal: CLBlastDiagonal, m: LibC::SizeT, n: LibC::SizeT, alpha: cl_half,
  #                   a_buffer: LibCL::ClMem, a_offset: LibC::SizeT, a_ld: LibC::SizeT, b_buffer: LibCL::ClMem,
  #                   b_offset: LibC::SizeT, b_ld: LibC::SizeT, queue: LibCL::ClCommandQueue*,
  #                   event: LibCL::ClEvent*): CLBlastStatusCode
  fun clblast_Strsm = CLBlastStrsm(layout : CLBlastLayout, side : CLBlastSide,
                                   triangle : CLBlastTriangle, a_transpose : CLBlastTranspose,
                                   diagonal : CLBlastDiagonal, m : LibC::SizeT, n : LibC::SizeT, alpha : LibC::Float,
                                   a_buffer : LibCL::ClMem, a_offset : LibC::SizeT, a_ld : LibC::SizeT, b_buffer : LibCL::ClMem,
                                   b_offset : LibC::SizeT, b_ld : LibC::SizeT, queue : LibCL::ClCommandQueue*,
                                   event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Dtrsm = CLBlastDtrsm(layout : CLBlastLayout, side : CLBlastSide,
                                   triangle : CLBlastTriangle, a_transpose : CLBlastTranspose,
                                   diagonal : CLBlastDiagonal, m : LibC::SizeT, n : LibC::SizeT, alpha : LibC::Double,
                                   a_buffer : LibCL::ClMem, a_offset : LibC::SizeT, a_ld : LibC::SizeT, b_buffer : LibCL::ClMem,
                                   b_offset : LibC::SizeT, b_ld : LibC::SizeT, queue : LibCL::ClCommandQueue*,
                                   event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Ctrsm = CLBlastCtrsm(layout : CLBlastLayout, side : CLBlastSide,
                                   triangle : CLBlastTriangle, a_transpose : CLBlastTranspose,
                                   diagonal : CLBlastDiagonal, m : LibC::SizeT, n : LibC::SizeT, alpha : ComplexFloat,
                                   a_buffer : LibCL::ClMem, a_offset : LibC::SizeT, a_ld : LibC::SizeT, b_buffer : LibCL::ClMem,
                                   b_offset : LibC::SizeT, b_ld : LibC::SizeT, queue : LibCL::ClCommandQueue*,
                                   event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Ztrsm = CLBlastZtrsm(layout : CLBlastLayout, side : CLBlastSide,
                                   triangle : CLBlastTriangle, a_transpose : CLBlastTranspose,
                                   diagonal : CLBlastDiagonal, m : LibC::SizeT, n : LibC::SizeT, alpha : ComplexDouble,
                                   a_buffer : LibCL::ClMem, a_offset : LibC::SizeT, a_ld : LibC::SizeT, b_buffer : LibCL::ClMem,
                                   b_offset : LibC::SizeT, b_ld : LibC::SizeT, queue : LibCL::ClCommandQueue*,
                                   event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Somatcopy = CLBlastSomatcopy(layout : CLBlastLayout, a_transpose : CLBlastTranspose,
                                           m : LibC::SizeT, n : LibC::SizeT, alpha : LibC::Float, a_buffer : LibCL::ClMem,
                                           a_offset : LibC::SizeT, a_ld : LibC::SizeT, b_buffer : LibCL::ClMem, b_offset : LibC::SizeT,
                                           b_ld : LibC::SizeT, queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Domatcopy = CLBlastDomatcopy(layout : CLBlastLayout, a_transpose : CLBlastTranspose,
                                           m : LibC::SizeT, n : LibC::SizeT, alpha : LibC::Double, a_buffer : LibCL::ClMem,
                                           a_offset : LibC::SizeT, a_ld : LibC::SizeT, b_buffer : LibCL::ClMem, b_offset : LibC::SizeT,
                                           b_ld : LibC::SizeT, queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Comatcopy = CLBlastComatcopy(layout : CLBlastLayout, a_transpose : CLBlastTranspose,
                                           m : LibC::SizeT, n : LibC::SizeT, alpha : ComplexFloat, a_buffer : LibCL::ClMem,
                                           a_offset : LibC::SizeT, a_ld : LibC::SizeT, b_buffer : LibCL::ClMem, b_offset : LibC::SizeT,
                                           b_ld : LibC::SizeT, queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Zomatcopy = CLBlastZomatcopy(layout : CLBlastLayout, a_transpose : CLBlastTranspose,
                                           m : LibC::SizeT, n : LibC::SizeT, alpha : ComplexDouble, a_buffer : LibCL::ClMem,
                                           a_offset : LibC::SizeT, a_ld : LibC::SizeT, b_buffer : LibCL::ClMem, b_offset : LibC::SizeT,
                                           b_ld : LibC::SizeT, queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  # fun clblast_Homatcopy = CLBlastHomatcopy(layout: CLBlastLayout, a_transpose: CLBlastTranspose,
  #                       m: LibC::SizeT, n: LibC::SizeT, alpha: cl_half, a_buffer: LibCL::ClMem,
  #                       a_offset: LibC::SizeT, a_ld: LibC::SizeT, b_buffer: LibCL::ClMem, b_offset: LibC::SizeT,
  #                       b_ld: LibC::SizeT, queue: LibCL::ClCommandQueue*, event: LibCL::ClEvent*): CLBlastStatusCode
  fun clblast_Sim2col = CLBlastSim2col(kernel_mode : CLBlastKernelMode, channels : LibC::SizeT, height : LibC::SizeT,
                                       width : LibC::SizeT, kernel_h : LibC::SizeT, kernel_w : LibC::SizeT, pad_h : LibC::SizeT,
                                       pad_w : LibC::SizeT, stride_h : LibC::SizeT, stride_w : LibC::SizeT, dilation_h : LibC::SizeT,
                                       dilation_w : LibC::SizeT, im_buffer : LibCL::ClMem, im_offset : LibC::SizeT,
                                       col_buffer : LibCL::ClMem, col_offset : LibC::SizeT,
                                       queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Dim2col = CLBlastDim2col(kernel_mode : CLBlastKernelMode, channels : LibC::SizeT, height : LibC::SizeT,
                                       width : LibC::SizeT, kernel_h : LibC::SizeT, kernel_w : LibC::SizeT, pad_h : LibC::SizeT,
                                       pad_w : LibC::SizeT, stride_h : LibC::SizeT, stride_w : LibC::SizeT, dilation_h : LibC::SizeT,
                                       dilation_w : LibC::SizeT, im_buffer : LibCL::ClMem, im_offset : LibC::SizeT,
                                       col_buffer : LibCL::ClMem, col_offset : LibC::SizeT,
                                       queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Cim2col = CLBlastCim2col(kernel_mode : CLBlastKernelMode, channels : LibC::SizeT, height : LibC::SizeT,
                                       width : LibC::SizeT, kernel_h : LibC::SizeT, kernel_w : LibC::SizeT, pad_h : LibC::SizeT,
                                       pad_w : LibC::SizeT, stride_h : LibC::SizeT, stride_w : LibC::SizeT, dilation_h : LibC::SizeT,
                                       dilation_w : LibC::SizeT, im_buffer : LibCL::ClMem, im_offset : LibC::SizeT,
                                       col_buffer : LibCL::ClMem, col_offset : LibC::SizeT,
                                       queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Zim2col = CLBlastZim2col(kernel_mode : CLBlastKernelMode, channels : LibC::SizeT, height : LibC::SizeT,
                                       width : LibC::SizeT, kernel_h : LibC::SizeT, kernel_w : LibC::SizeT, pad_h : LibC::SizeT,
                                       pad_w : LibC::SizeT, stride_h : LibC::SizeT, stride_w : LibC::SizeT, dilation_h : LibC::SizeT,
                                       dilation_w : LibC::SizeT, im_buffer : LibCL::ClMem, im_offset : LibC::SizeT,
                                       col_buffer : LibCL::ClMem, col_offset : LibC::SizeT,
                                       queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  # fun clblast_Him2col = CLBlastHim2col(kernel_mode: CLBlastKernelMode, channels: LibC::SizeT, height: LibC::SizeT,
  #                     width: LibC::SizeT, kernel_h: LibC::SizeT, kernel_w: LibC::SizeT, pad_h: LibC::SizeT,
  #                     pad_w: LibC::SizeT, stride_h: LibC::SizeT, stride_w: LibC::SizeT, dilation_h: LibC::SizeT,
  #                     dilation_w: LibC::SizeT, im_buffer: LibCL::ClMem, im_offset: LibC::SizeT,
  #                     col_buffer: LibCL::ClMem, col_offset: LibC::SizeT,
  #                     queue: LibCL::ClCommandQueue*, event: LibCL::ClEvent*): CLBlastStatusCode
  fun clblast_Scol2im = CLBlastScol2im(kernel_mode : CLBlastKernelMode, channels : LibC::SizeT, height : LibC::SizeT,
                                       width : LibC::SizeT, kernel_h : LibC::SizeT, kernel_w : LibC::SizeT, pad_h : LibC::SizeT,
                                       pad_w : LibC::SizeT, stride_h : LibC::SizeT, stride_w : LibC::SizeT, dilation_h : LibC::SizeT,
                                       dilation_w : LibC::SizeT, col_buffer : LibCL::ClMem, col_offset : LibC::SizeT,
                                       im_buffer : LibCL::ClMem, im_offset : LibC::SizeT,
                                       queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Dcol2im = CLBlastDcol2im(kernel_mode : CLBlastKernelMode, channels : LibC::SizeT, height : LibC::SizeT,
                                       width : LibC::SizeT, kernel_h : LibC::SizeT, kernel_w : LibC::SizeT, pad_h : LibC::SizeT,
                                       pad_w : LibC::SizeT, stride_h : LibC::SizeT, stride_w : LibC::SizeT, dilation_h : LibC::SizeT,
                                       dilation_w : LibC::SizeT, col_buffer : LibCL::ClMem, col_offset : LibC::SizeT,
                                       im_buffer : LibCL::ClMem, im_offset : LibC::SizeT,
                                       queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Ccol2im = CLBlastCcol2im(kernel_mode : CLBlastKernelMode, channels : LibC::SizeT, height : LibC::SizeT,
                                       width : LibC::SizeT, kernel_h : LibC::SizeT, kernel_w : LibC::SizeT, pad_h : LibC::SizeT,
                                       pad_w : LibC::SizeT, stride_h : LibC::SizeT, stride_w : LibC::SizeT, dilation_h : LibC::SizeT,
                                       dilation_w : LibC::SizeT, col_buffer : LibCL::ClMem, col_offset : LibC::SizeT,
                                       im_buffer : LibCL::ClMem, im_offset : LibC::SizeT,
                                       queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Zcol2im = CLBlastZcol2im(kernel_mode : CLBlastKernelMode, channels : LibC::SizeT, height : LibC::SizeT,
                                       width : LibC::SizeT, kernel_h : LibC::SizeT, kernel_w : LibC::SizeT, pad_h : LibC::SizeT,
                                       pad_w : LibC::SizeT, stride_h : LibC::SizeT, stride_w : LibC::SizeT, dilation_h : LibC::SizeT,
                                       dilation_w : LibC::SizeT, col_buffer : LibCL::ClMem, col_offset : LibC::SizeT,
                                       im_buffer : LibCL::ClMem, im_offset : LibC::SizeT,
                                       queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  # fun clblast_Hcol2im = CLBlastHcol2im(kernel_mode: CLBlastKernelMode, channels: LibC::SizeT, height: LibC::SizeT,
  #                     width: LibC::SizeT, kernel_h: LibC::SizeT, kernel_w: LibC::SizeT, pad_h: LibC::SizeT,
  #                     pad_w: LibC::SizeT, stride_h: LibC::SizeT, stride_w: LibC::SizeT, dilation_h: LibC::SizeT,
  #                     dilation_w: LibC::SizeT, col_buffer: LibCL::ClMem, col_offset: LibC::SizeT,
  #                     im_buffer: LibCL::ClMem, im_offset: LibC::SizeT,
  #                     queue: LibCL::ClCommandQueue*, event: LibCL::ClEvent*): CLBlastStatusCode
  fun clblast_Sconvgemm = CLBlastSconvgemm(kernel_mode : CLBlastKernelMode, channels : LibC::SizeT,
                                           height : LibC::SizeT, width : LibC::SizeT, kernel_h : LibC::SizeT, kernel_w : LibC::SizeT,
                                           pad_h : LibC::SizeT, pad_w : LibC::SizeT, stride_h : LibC::SizeT, stride_w : LibC::SizeT,
                                           dilation_h : LibC::SizeT, dilation_w : LibC::SizeT, num_kernels : LibC::SizeT,
                                           batch_count : LibC::SizeT, im_buffer : LibCL::ClMem, im_offset : LibC::SizeT,
                                           kernel_buffer : LibCL::ClMem, kernel_offset : LibC::SizeT,
                                           result_buffer : LibCL::ClMem, result_offset : LibC::SizeT,
                                           queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_Dconvgemm = CLBlastDconvgemm(kernel_mode : CLBlastKernelMode, channels : LibC::SizeT,
                                           height : LibC::SizeT, width : LibC::SizeT, kernel_h : LibC::SizeT, kernel_w : LibC::SizeT,
                                           pad_h : LibC::SizeT, pad_w : LibC::SizeT, stride_h : LibC::SizeT, stride_w : LibC::SizeT,
                                           dilation_h : LibC::SizeT, dilation_w : LibC::SizeT, num_kernels : LibC::SizeT,
                                           batch_count : LibC::SizeT, im_buffer : LibCL::ClMem, im_offset : LibC::SizeT,
                                           kernel_buffer : LibCL::ClMem, kernel_offset : LibC::SizeT,
                                           result_buffer : LibCL::ClMem, result_offset : LibC::SizeT,
                                           queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  # fun clblast_Hconvgemm = CLBlastHconvgemm(kernel_mode: CLBlastKernelMode, channels: LibC::SizeT,
  #                       height: LibC::SizeT, width: LibC::SizeT, kernel_h: LibC::SizeT, kernel_w: LibC::SizeT,
  #                       pad_h: LibC::SizeT, pad_w: LibC::SizeT, stride_h: LibC::SizeT, stride_w: LibC::SizeT,
  #                       dilation_h: LibC::SizeT, dilation_w: LibC::SizeT, num_kernels: LibC::SizeT,
  #                       batch_count: LibC::SizeT, im_buffer: LibCL::ClMem, im_offset: LibC::SizeT,
  #                       kernel_buffer: LibCL::ClMem, kernel_offset: LibC::SizeT,
  #                       result_buffer: LibCL::ClMem, result_offset: LibC::SizeT,
  #                       queue: LibCL::ClCommandQueue*, event: LibCL::ClEvent*): CLBlastStatusCode
  fun clblast_SaxpyBatched = CLBlastSaxpyBatched(n : LibC::SizeT, alphas : LibC::Float*, x_buffer : LibCL::ClMem,
                                                 x_offsets : LibC::SizeT*, x_inc : LibC::SizeT, y_buffer : LibCL::ClMem,
                                                 y_offsets : LibC::SizeT*, y_inc : LibC::SizeT, batch_count : LibC::SizeT,
                                                 queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_DaxpyBatched = CLBlastDaxpyBatched(n : LibC::SizeT, alphas : LibC::Double*, x_buffer : LibCL::ClMem,
                                                 x_offsets : LibC::SizeT*, x_inc : LibC::SizeT, y_buffer : LibCL::ClMem,
                                                 y_offsets : LibC::SizeT*, y_inc : LibC::SizeT, batch_count : LibC::SizeT,
                                                 queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_CaxpyBatched = CLBlastCaxpyBatched(n : LibC::SizeT, alphas : ComplexFloat*, x_buffer : LibCL::ClMem,
                                                 x_offsets : LibC::SizeT*, x_inc : LibC::SizeT, y_buffer : LibCL::ClMem,
                                                 y_offsets : LibC::SizeT*, y_inc : LibC::SizeT, batch_count : LibC::SizeT,
                                                 queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_ZaxpyBatched = CLBlastZaxpyBatched(n : LibC::SizeT, alphas : ComplexDouble*, x_buffer : LibCL::ClMem,
                                                 x_offsets : LibC::SizeT*, x_inc : LibC::SizeT, y_buffer : LibCL::ClMem,
                                                 y_offsets : LibC::SizeT*, y_inc : LibC::SizeT, batch_count : LibC::SizeT,
                                                 queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  # fun clblast_HaxpyBatched = CLBlastHaxpyBatched(n: LibC::SizeT, alphas: cl_half*, x_buffer: LibCL::ClMem,
  #                          x_offsets: LibC::SizeT*, x_inc: LibC::SizeT, y_buffer: LibCL::ClMem,
  #                          y_offsets: LibC::SizeT*, y_inc: LibC::SizeT, batch_count: LibC::SizeT,
  #                          queue: LibCL::ClCommandQueue*, event: LibCL::ClEvent*): CLBlastStatusCode
  fun clblast_SgemmBatched = CLBlastSgemmBatched(layout : CLBlastLayout, a_transpose : CLBlastTranspose,
                                                 b_transpose : CLBlastTranspose, m : LibC::SizeT, n : LibC::SizeT, k : LibC::SizeT,
                                                 alphas : LibC::Float*, a_buffer : LibCL::ClMem, a_offsets : LibC::SizeT*,
                                                 a_ld : LibC::SizeT, b_buffer : LibCL::ClMem, b_offsets : LibC::SizeT*,
                                                 b_ld : LibC::SizeT, betas : LibC::Float*, c_buffer : LibCL::ClMem,
                                                 c_offsets : LibC::SizeT*, c_ld : LibC::SizeT, batch_count : LibC::SizeT,
                                                 queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_DgemmBatched = CLBlastDgemmBatched(layout : CLBlastLayout, a_transpose : CLBlastTranspose,
                                                 b_transpose : CLBlastTranspose, m : LibC::SizeT, n : LibC::SizeT, k : LibC::SizeT,
                                                 alphas : LibC::Double*, a_buffer : LibCL::ClMem, a_offsets : LibC::SizeT*,
                                                 a_ld : LibC::SizeT, b_buffer : LibCL::ClMem, b_offsets : LibC::SizeT*,
                                                 b_ld : LibC::SizeT, betas : LibC::Double*, c_buffer : LibCL::ClMem,
                                                 c_offsets : LibC::SizeT*, c_ld : LibC::SizeT, batch_count : LibC::SizeT,
                                                 queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_CgemmBatched = CLBlastCgemmBatched(layout : CLBlastLayout, a_transpose : CLBlastTranspose,
                                                 b_transpose : CLBlastTranspose, m : LibC::SizeT, n : LibC::SizeT, k : LibC::SizeT,
                                                 alphas : ComplexFloat*, a_buffer : LibCL::ClMem,
                                                 a_offsets : LibC::SizeT*, a_ld : LibC::SizeT, b_buffer : LibCL::ClMem,
                                                 b_offsets : LibC::SizeT*, b_ld : LibC::SizeT, betas : ComplexFloat*,
                                                 c_buffer : LibCL::ClMem, c_offsets : LibC::SizeT*, c_ld : LibC::SizeT,
                                                 batch_count : LibC::SizeT, queue : LibCL::ClCommandQueue*,
                                                 event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_ZgemmBatched = CLBlastZgemmBatched(layout : CLBlastLayout, a_transpose : CLBlastTranspose,
                                                 b_transpose : CLBlastTranspose, m : LibC::SizeT, n : LibC::SizeT, k : LibC::SizeT,
                                                 alphas : ComplexDouble*, a_buffer : LibCL::ClMem,
                                                 a_offsets : LibC::SizeT*, a_ld : LibC::SizeT, b_buffer : LibCL::ClMem,
                                                 b_offsets : LibC::SizeT*, b_ld : LibC::SizeT, betas : ComplexDouble*,
                                                 c_buffer : LibCL::ClMem, c_offsets : LibC::SizeT*, c_ld : LibC::SizeT,
                                                 batch_count : LibC::SizeT, queue : LibCL::ClCommandQueue*,
                                                 event : LibCL::ClEvent*) : CLBlastStatusCode
  # fun clblast_HgemmBatched = CLBlastHgemmBatched(layout: CLBlastLayout, a_transpose: CLBlastTranspose,
  #                          b_transpose: CLBlastTranspose, m: LibC::SizeT, n: LibC::SizeT, k: LibC::SizeT,
  #                          alphas: cl_half*, a_buffer: LibCL::ClMem, a_offsets: LibC::SizeT*,
  #                          a_ld: LibC::SizeT, b_buffer: LibCL::ClMem, b_offsets: LibC::SizeT*,
  #                          b_ld: LibC::SizeT, betas: cl_half*, c_buffer: LibCL::ClMem,
  #                          c_offsets: LibC::SizeT*, c_ld: LibC::SizeT, batch_count: LibC::SizeT,
  #                          queue: LibCL::ClCommandQueue*, event: LibCL::ClEvent*): CLBlastStatusCode
  fun clblast_SgemmStridedBatched = CLBlastSgemmStridedBatched(layout : CLBlastLayout,
                                                               a_transpose : CLBlastTranspose,
                                                               b_transpose : CLBlastTranspose, m : LibC::SizeT, n : LibC::SizeT,
                                                               k : LibC::SizeT, alpha : LibC::Float, a_buffer : LibCL::ClMem,
                                                               a_offset : LibC::SizeT, a_ld : LibC::SizeT, a_stride : LibC::SizeT,
                                                               b_buffer : LibCL::ClMem, b_offset : LibC::SizeT, b_ld : LibC::SizeT,
                                                               b_stride : LibC::SizeT, beta : LibC::Float, c_buffer : LibCL::ClMem,
                                                               c_offset : LibC::SizeT, c_ld : LibC::SizeT, c_stride : LibC::SizeT,
                                                               batch_count : LibC::SizeT, queue : LibCL::ClCommandQueue*,
                                                               event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_DgemmStridedBatched = CLBlastDgemmStridedBatched(layout : CLBlastLayout,
                                                               a_transpose : CLBlastTranspose,
                                                               b_transpose : CLBlastTranspose, m : LibC::SizeT, n : LibC::SizeT,
                                                               k : LibC::SizeT, alpha : LibC::Double, a_buffer : LibCL::ClMem,
                                                               a_offset : LibC::SizeT, a_ld : LibC::SizeT, a_stride : LibC::SizeT,
                                                               b_buffer : LibCL::ClMem, b_offset : LibC::SizeT, b_ld : LibC::SizeT,
                                                               b_stride : LibC::SizeT, beta : LibC::Double, c_buffer : LibCL::ClMem,
                                                               c_offset : LibC::SizeT, c_ld : LibC::SizeT, c_stride : LibC::SizeT,
                                                               batch_count : LibC::SizeT, queue : LibCL::ClCommandQueue*,
                                                               event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_CgemmStridedBatched = CLBlastCgemmStridedBatched(layout : CLBlastLayout,
                                                               a_transpose : CLBlastTranspose,
                                                               b_transpose : CLBlastTranspose, m : LibC::SizeT, n : LibC::SizeT,
                                                               k : LibC::SizeT, alpha : ComplexFloat, a_buffer : LibCL::ClMem,
                                                               a_offset : LibC::SizeT, a_ld : LibC::SizeT, a_stride : LibC::SizeT,
                                                               b_buffer : LibCL::ClMem, b_offset : LibC::SizeT, b_ld : LibC::SizeT,
                                                               b_stride : LibC::SizeT, beta : ComplexFloat, c_buffer : LibCL::ClMem,
                                                               c_offset : LibC::SizeT, c_ld : LibC::SizeT, c_stride : LibC::SizeT,
                                                               batch_count : LibC::SizeT, queue : LibCL::ClCommandQueue*,
                                                               event : LibCL::ClEvent*) : CLBlastStatusCode
  fun clblast_ZgemmStridedBatched = CLBlastZgemmStridedBatched(layout : CLBlastLayout,
                                                               a_transpose : CLBlastTranspose,
                                                               b_transpose : CLBlastTranspose, m : LibC::SizeT, n : LibC::SizeT,
                                                               k : LibC::SizeT, alpha : ComplexDouble, a_buffer : LibCL::ClMem,
                                                               a_offset : LibC::SizeT, a_ld : LibC::SizeT, a_stride : LibC::SizeT,
                                                               b_buffer : LibCL::ClMem, b_offset : LibC::SizeT, b_ld : LibC::SizeT,
                                                               b_stride : LibC::SizeT, beta : ComplexDouble, c_buffer : LibCL::ClMem,
                                                               c_offset : LibC::SizeT, c_ld : LibC::SizeT, c_stride : LibC::SizeT,
                                                               batch_count : LibC::SizeT, queue : LibCL::ClCommandQueue*,
                                                               event : LibCL::ClEvent*) : CLBlastStatusCode
  # fun clblast_HgemmStridedBatched = CLBlastHgemmStridedBatched(layout: CLBlastLayout,
  #                                 a_transpose: CLBlastTranspose,
  #                                 b_transpose: CLBlastTranspose, m: LibC::SizeT, n: LibC::SizeT,
  #                                 k: LibC::SizeT, alpha: cl_half, a_buffer: LibCL::ClMem,
  #                                 a_offset: LibC::SizeT, a_ld: LibC::SizeT, a_stride: LibC::SizeT,
  #                                 b_buffer: LibCL::ClMem, b_offset: LibC::SizeT, b_ld: LibC::SizeT,
  #                                 b_stride: LibC::SizeT, beta: cl_half, c_buffer: LibCL::ClMem,
  #                                 c_offset: LibC::SizeT, c_ld: LibC::SizeT, c_stride: LibC::SizeT,
  #                                 batch_count: LibC::SizeT, queue: LibCL::ClCommandQueue*,
  #                                 event: LibCL::ClEvent*): CLBlastStatusCode
  fun clblast_SgemmWithTempBuffer = CLBlastSgemmWithTempBuffer(layout : CLBlastLayout,
                                                               a_transpose : CLBlastTranspose,
                                                               b_transpose : CLBlastTranspose, m : LibC::SizeT, n : LibC::SizeT,
                                                               k : LibC::SizeT, alpha : LibC::Float, a_buffer : LibCL::ClMem,
                                                               a_offset : LibC::SizeT, a_ld : LibC::SizeT, b_buffer : LibCL::ClMem,
                                                               b_offset : LibC::SizeT, b_ld : LibC::SizeT, beta : LibC::Float,
                                                               c_buffer : LibCL::ClMem, c_offset : LibC::SizeT, c_ld : LibC::SizeT,
                                                               queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*,
                                                               temp_buffer : LibCL::ClMem) : CLBlastStatusCode
  fun clblast_DgemmWithTempBuffer = CLBlastDgemmWithTempBuffer(layout : CLBlastLayout,
                                                               a_transpose : CLBlastTranspose,
                                                               b_transpose : CLBlastTranspose, m : LibC::SizeT, n : LibC::SizeT,
                                                               k : LibC::SizeT, alpha : LibC::Double, a_buffer : LibCL::ClMem,
                                                               a_offset : LibC::SizeT, a_ld : LibC::SizeT, b_buffer : LibCL::ClMem,
                                                               b_offset : LibC::SizeT, b_ld : LibC::SizeT, beta : LibC::Double,
                                                               c_buffer : LibCL::ClMem, c_offset : LibC::SizeT, c_ld : LibC::SizeT,
                                                               queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*,
                                                               temp_buffer : LibCL::ClMem) : CLBlastStatusCode
  fun clblast_CgemmWithTempBuffer = CLBlastCgemmWithTempBuffer(layout : CLBlastLayout,
                                                               a_transpose : CLBlastTranspose,
                                                               b_transpose : CLBlastTranspose, m : LibC::SizeT, n : LibC::SizeT,
                                                               k : LibC::SizeT, alpha : ComplexFloat, a_buffer : LibCL::ClMem,
                                                               a_offset : LibC::SizeT, a_ld : LibC::SizeT, b_buffer : LibCL::ClMem,
                                                               b_offset : LibC::SizeT, b_ld : LibC::SizeT, beta : ComplexFloat,
                                                               c_buffer : LibCL::ClMem, c_offset : LibC::SizeT, c_ld : LibC::SizeT,
                                                               queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*,
                                                               temp_buffer : LibCL::ClMem) : CLBlastStatusCode
  fun clblast_ZgemmWithTempBuffer = CLBlastZgemmWithTempBuffer(layout : CLBlastLayout,
                                                               a_transpose : CLBlastTranspose,
                                                               b_transpose : CLBlastTranspose, m : LibC::SizeT, n : LibC::SizeT,
                                                               k : LibC::SizeT, alpha : ComplexDouble, a_buffer : LibCL::ClMem,
                                                               a_offset : LibC::SizeT, a_ld : LibC::SizeT, b_buffer : LibCL::ClMem,
                                                               b_offset : LibC::SizeT, b_ld : LibC::SizeT, beta : ComplexDouble,
                                                               c_buffer : LibCL::ClMem, c_offset : LibC::SizeT, c_ld : LibC::SizeT,
                                                               queue : LibCL::ClCommandQueue*, event : LibCL::ClEvent*,
                                                               temp_buffer : LibCL::ClMem) : CLBlastStatusCode
  # fun clblast_HgemmWithTempBuffer = CLBlastHgemmWithTempBuffer(layout: CLBlastLayout,
  #                                 a_transpose: CLBlastTranspose,
  #                                 b_transpose: CLBlastTranspose, m: LibC::SizeT, n: LibC::SizeT,
  #                                 k: LibC::SizeT, alpha: cl_half, a_buffer: LibCL::ClMem,
  #                                 a_offset: LibC::SizeT, a_ld: LibC::SizeT, b_buffer: LibCL::ClMem,
  #                                 b_offset: LibC::SizeT, b_ld: LibC::SizeT, beta: cl_half,
  #                                 c_buffer: LibCL::ClMem, c_offset: LibC::SizeT, c_ld: LibC::SizeT,
  #                                 queue: LibCL::ClCommandQueue*, event: LibCL::ClEvent*,
  #                                 temp_buffer: LibCL::ClMem): CLBlastStatusCode
  fun clblast_SGemmTempBufferSize = CLBlastSGemmTempBufferSize(layout : CLBlastLayout,
                                                               a_transpose : CLBlastTranspose,
                                                               b_transpose : CLBlastTranspose, m : LibC::SizeT, n : LibC::SizeT,
                                                               k : LibC::SizeT, a_offset : LibC::SizeT, a_ld : LibC::SizeT,
                                                               b_offset : LibC::SizeT, b_ld : LibC::SizeT, c_offset : LibC::SizeT,
                                                               c_ld : LibC::SizeT, queue : LibCL::ClCommandQueue*,
                                                               temp_buffer_size : LibC::SizeT*) : CLBlastStatusCode
  fun clblast_DGemmTempBufferSize = CLBlastDGemmTempBufferSize(layout : CLBlastLayout,
                                                               a_transpose : CLBlastTranspose,
                                                               b_transpose : CLBlastTranspose, m : LibC::SizeT, n : LibC::SizeT,
                                                               k : LibC::SizeT, a_offset : LibC::SizeT, a_ld : LibC::SizeT,
                                                               b_offset : LibC::SizeT, b_ld : LibC::SizeT, c_offset : LibC::SizeT,
                                                               c_ld : LibC::SizeT, queue : LibCL::ClCommandQueue*,
                                                               temp_buffer_size : LibC::SizeT*) : CLBlastStatusCode
  fun clblast_CGemmTempBufferSize = CLBlastCGemmTempBufferSize(layout : CLBlastLayout,
                                                               a_transpose : CLBlastTranspose,
                                                               b_transpose : CLBlastTranspose, m : LibC::SizeT, n : LibC::SizeT,
                                                               k : LibC::SizeT, a_offset : LibC::SizeT, a_ld : LibC::SizeT,
                                                               b_offset : LibC::SizeT, b_ld : LibC::SizeT, c_offset : LibC::SizeT,
                                                               c_ld : LibC::SizeT, queue : LibCL::ClCommandQueue*,
                                                               temp_buffer_size : LibC::SizeT*) : CLBlastStatusCode
  fun clblast_ZGemmTempBufferSize = CLBlastZGemmTempBufferSize(layout : CLBlastLayout,
                                                               a_transpose : CLBlastTranspose,
                                                               b_transpose : CLBlastTranspose, m : LibC::SizeT, n : LibC::SizeT,
                                                               k : LibC::SizeT, a_offset : LibC::SizeT, a_ld : LibC::SizeT,
                                                               b_offset : LibC::SizeT, b_ld : LibC::SizeT, c_offset : LibC::SizeT,
                                                               c_ld : LibC::SizeT, queue : LibCL::ClCommandQueue*,
                                                               temp_buffer_size : LibC::SizeT*) : CLBlastStatusCode
  # fun clblast_HGemmTempBufferSize = CLBlastHGemmTempBufferSize(layout: CLBlastLayout,
  #                                 a_transpose: CLBlastTranspose,
  #                                 b_transpose: CLBlastTranspose, m: LibC::SizeT, n: LibC::SizeT,
  #                                 k: LibC::SizeT, a_offset: LibC::SizeT, a_ld: LibC::SizeT,
  #                                 b_offset: LibC::SizeT, b_ld: LibC::SizeT, c_offset: LibC::SizeT,
  #                                 c_ld: LibC::SizeT, queue: LibCL::ClCommandQueue*,
  #                                 temp_buffer_size: LibC::SizeT*): CLBlastStatusCode
  fun clblast_ClearCache = CLBlastClearCache : CLBlastStatusCode
  fun clblast_FillCache = CLBlastFillCache(device : LibCL::ClDeviceId) : CLBlastStatusCode
  fun clblast_OverrideParameters = CLBlastOverrideParameters(device : LibCL::ClDeviceId, kernel_name : UInt8*,
                                                             precision : CLBlastPrecision, num_parameters : LibC::SizeT,
                                                             parameters_names : UInt8**,
                                                             parameters_values : LibC::SizeT*) : CLBlastStatusCode
end
