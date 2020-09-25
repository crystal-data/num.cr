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

@[Link("plplot")]
lib LibPlplot
  fun pl_setcontlabelformat = c_pl_setcontlabelformat(lexp : Plint, sigdig : Plint)
  alias X__Int32T = LibC::Int
  alias Int32T = X__Int32T
  alias Plint = Int32T
  fun pl_setcontlabelparam = c_pl_setcontlabelparam(offset : Plflt, size : Plflt, spacing : Plflt, active : Plint)
  alias Plflt = LibC::Double
  fun pladv = c_pladv(page : Plint)
  fun plarc = c_plarc(x : Plflt, y : Plflt, a : Plflt, b : Plflt, angle1 : Plflt, angle2 : Plflt, rotate : Plflt, fill : Plbool)
  alias Plbool = Plint
  fun plaxes = c_plaxes(x0 : Plflt, y0 : Plflt, xopt : PlcharVector, xtick : Plflt, nxsub : Plint, yopt : PlcharVector, ytick : Plflt, nysub : Plint)
  alias PlcharVector = LibC::Char*
  fun plbin = c_plbin(nbin : Plint, x : PlfltVector, y : PlfltVector, opt : Plint)
  alias PlfltVector = Plflt*
  fun plbtime = c_plbtime(year : PlintNcScalar, month : PlintNcScalar, day : PlintNcScalar, hour : PlintNcScalar, min : PlintNcScalar, sec : PlfltNcScalar, ctime : Plflt)
  alias PlintNcScalar = Plint*
  alias PlfltNcScalar = Plflt*
  fun plbop = c_plbop
  fun plbox = c_plbox(xopt : PlcharVector, xtick : Plflt, nxsub : Plint, yopt : PlcharVector, ytick : Plflt, nysub : Plint)
  fun plbox3 = c_plbox3(xopt : PlcharVector, xlabel : PlcharVector, xtick : Plflt, nxsub : Plint, yopt : PlcharVector, ylabel : PlcharVector, ytick : Plflt, nysub : Plint, zopt : PlcharVector, zlabel : PlcharVector, ztick : Plflt, nzsub : Plint)
  fun plcalc_world = c_plcalc_world(rx : Plflt, ry : Plflt, wx : PlfltNcScalar, wy : PlfltNcScalar, window : PlintNcScalar)
  fun plclear = c_plclear
  fun plcol0 = c_plcol0(icol0 : Plint)
  fun plcol1 = c_plcol1(col1 : Plflt)
  fun plconfigtime = c_plconfigtime(scale : Plflt, offset1 : Plflt, offset2 : Plflt, ccontrol : Plint, ifbtime_offset : Plbool, year : Plint, month : Plint, day : Plint, hour : Plint, min : Plint, sec : Plflt)
  fun plcont = c_plcont(f : PlfltMatrix, nx : Plint, ny : Plint, kx : Plint, lx : Plint, ky : Plint, ly : Plint, clevel : PlfltVector, nlevel : Plint, pltr : PltransformCallback, pltr_data : PlPointer)
  alias PlfltMatrix = Plflt**
  alias PlPointer = Void*
  alias PltransformCallback = (Plflt, Plflt, PlfltNcScalar, PlfltNcScalar, PlPointer -> Void)
  fun plcpstrm = c_plcpstrm(iplsr : Plint, flags : Plbool)
  fun plctime = c_plctime(year : Plint, month : Plint, day : Plint, hour : Plint, min : Plint, sec : Plflt, ctime : PlfltNcScalar)
  fun plend = c_plend
  fun plend1 = c_plend1
  fun plenv = c_plenv(xmin : Plflt, xmax : Plflt, ymin : Plflt, ymax : Plflt, just : Plint, axis : Plint)
  fun plenv0 = c_plenv0(xmin : Plflt, xmax : Plflt, ymin : Plflt, ymax : Plflt, just : Plint, axis : Plint)
  fun pleop = c_pleop
  fun plerrx = c_plerrx(n : Plint, xmin : PlfltVector, xmax : PlfltVector, y : PlfltVector)
  fun plerry = c_plerry(n : Plint, x : PlfltVector, ymin : PlfltVector, ymax : PlfltVector)
  fun plfamadv = c_plfamadv
  fun plfill = c_plfill(n : Plint, x : PlfltVector, y : PlfltVector)
  fun plfill3 = c_plfill3(n : Plint, x : PlfltVector, y : PlfltVector, z : PlfltVector)
  fun plflush = c_plflush
  fun plfont = c_plfont(ifont : Plint)
  fun plfontld = c_plfontld(fnt : Plint)
  fun plgchr = c_plgchr(p_def : PlfltNcScalar, p_ht : PlfltNcScalar)
  fun plgcmap1_range = c_plgcmap1_range(min_color : PlfltNcScalar, max_color : PlfltNcScalar)
  fun plgcol0 = c_plgcol0(icol0 : Plint, r : PlintNcScalar, g : PlintNcScalar, b : PlintNcScalar)
  fun plgcol0a = c_plgcol0a(icol0 : Plint, r : PlintNcScalar, g : PlintNcScalar, b : PlintNcScalar, alpha : PlfltNcScalar)
  fun plgcolbg = c_plgcolbg(r : PlintNcScalar, g : PlintNcScalar, b : PlintNcScalar)
  fun plgcolbga = c_plgcolbga(r : PlintNcScalar, g : PlintNcScalar, b : PlintNcScalar, alpha : PlfltNcScalar)
  fun plgcompression = c_plgcompression(compression : PlintNcScalar)
  fun plgdev = c_plgdev(p_dev : PlcharNcVector)
  alias PlcharNcVector = LibC::Char*
  fun plgdidev = c_plgdidev(p_mar : PlfltNcScalar, p_aspect : PlfltNcScalar, p_jx : PlfltNcScalar, p_jy : PlfltNcScalar)
  fun plgdiori = c_plgdiori(p_rot : PlfltNcScalar)
  fun plgdiplt = c_plgdiplt(p_xmin : PlfltNcScalar, p_ymin : PlfltNcScalar, p_xmax : PlfltNcScalar, p_ymax : PlfltNcScalar)
  fun plgdrawmode = c_plgdrawmode : Plint
  fun plgfci = c_plgfci(p_fci : PlunicodeNcScalar)
  alias X__Uint32T = LibC::UInt
  alias Uint32T = X__Uint32T
  alias Pluint = Uint32T
  alias Plunicode = Pluint
  alias PlunicodeNcScalar = Plunicode*
  fun plgfam = c_plgfam(p_fam : PlintNcScalar, p_num : PlintNcScalar, p_bmax : PlintNcScalar)
  fun plgfnam = c_plgfnam(fnam : PlcharNcVector)
  fun plgfont = c_plgfont(p_family : PlintNcScalar, p_style : PlintNcScalar, p_weight : PlintNcScalar)
  fun plglevel = c_plglevel(p_level : PlintNcScalar)
  fun plgpage = c_plgpage(p_xp : PlfltNcScalar, p_yp : PlfltNcScalar, p_xleng : PlintNcScalar, p_yleng : PlintNcScalar, p_xoff : PlintNcScalar, p_yoff : PlintNcScalar)
  fun plgra = c_plgra
  fun plgradient = c_plgradient(n : Plint, x : PlfltVector, y : PlfltVector, angle : Plflt)
  fun plgriddata = c_plgriddata(x : PlfltVector, y : PlfltVector, z : PlfltVector, npts : Plint, xg : PlfltVector, nptsx : Plint, yg : PlfltVector, nptsy : Plint, zg : PlfltNcMatrix, type : Plint, data : Plflt)
  alias PlfltNcMatrix = Plflt**
  fun plgspa = c_plgspa(xmin : PlfltNcScalar, xmax : PlfltNcScalar, ymin : PlfltNcScalar, ymax : PlfltNcScalar)
  fun plgstrm = c_plgstrm(p_strm : PlintNcScalar)
  fun plgver = c_plgver(p_ver : PlcharNcVector)
  fun plgvpd = c_plgvpd(p_xmin : PlfltNcScalar, p_xmax : PlfltNcScalar, p_ymin : PlfltNcScalar, p_ymax : PlfltNcScalar)
  fun plgvpw = c_plgvpw(p_xmin : PlfltNcScalar, p_xmax : PlfltNcScalar, p_ymin : PlfltNcScalar, p_ymax : PlfltNcScalar)
  fun plgxax = c_plgxax(p_digmax : PlintNcScalar, p_digits : PlintNcScalar)
  fun plgyax = c_plgyax(p_digmax : PlintNcScalar, p_digits : PlintNcScalar)
  fun plgzax = c_plgzax(p_digmax : PlintNcScalar, p_digits : PlintNcScalar)
  fun plhist = c_plhist(n : Plint, data : PlfltVector, datmin : Plflt, datmax : Plflt, nbin : Plint, opt : Plint)
  fun plhlsrgb = c_plhlsrgb(h : Plflt, l : Plflt, s : Plflt, p_r : PlfltNcScalar, p_g : PlfltNcScalar, p_b : PlfltNcScalar)
  fun plinit = c_plinit
  fun pljoin = c_pljoin(x1 : Plflt, y1 : Plflt, x2 : Plflt, y2 : Plflt)
  fun pllab = c_pllab(xlabel : PlcharVector, ylabel : PlcharVector, tlabel : PlcharVector)
  fun pllegend = c_pllegend(p_legend_width : PlfltNcScalar, p_legend_height : PlfltNcScalar, opt : Plint, position : Plint, x : Plflt, y : Plflt, plot_width : Plflt, bg_color : Plint, bb_color : Plint, bb_style : Plint, nrow : Plint, ncolumn : Plint, nlegend : Plint, opt_array : PlintVector, text_offset : Plflt, text_scale : Plflt, text_spacing : Plflt, text_justification : Plflt, text_colors : PlintVector, text : PlcharMatrix, box_colors : PlintVector, box_patterns : PlintVector, box_scales : PlfltVector, box_line_widths : PlfltVector, line_colors : PlintVector, line_styles : PlintVector, line_widths : PlfltVector, symbol_colors : PlintVector, symbol_scales : PlfltVector, symbol_numbers : PlintVector, symbols : PlcharMatrix)
  alias PlintVector = Plint*
  alias PlcharMatrix = LibC::Char**
  fun plcolorbar = c_plcolorbar(p_colorbar_width : PlfltNcScalar, p_colorbar_height : PlfltNcScalar, opt : Plint, position : Plint, x : Plflt, y : Plflt, x_length : Plflt, y_length : Plflt, bg_color : Plint, bb_color : Plint, bb_style : Plint, low_cap_color : Plflt, high_cap_color : Plflt, cont_color : Plint, cont_width : Plflt, n_labels : Plint, label_opts : PlintVector, labels : PlcharMatrix, n_axes : Plint, axis_opts : PlcharMatrix, ticks : PlfltVector, sub_ticks : PlintVector, n_values : PlintVector, values : PlfltMatrix)
  fun pllightsource = c_pllightsource(x : Plflt, y : Plflt, z : Plflt)
  fun plline = c_plline(n : Plint, x : PlfltVector, y : PlfltVector)
  fun plline3 = c_plline3(n : Plint, x : PlfltVector, y : PlfltVector, z : PlfltVector)
  fun pllsty = c_pllsty(lin : Plint)
  fun plmap = c_plmap(mapform : PlmapformCallback, name : PlcharVector, minx : Plflt, maxx : Plflt, miny : Plflt, maxy : Plflt)
  alias PlfltNcVector = Plflt*
  alias PlmapformCallback = (Plint, PlfltNcVector, PlfltNcVector -> Void)
  fun plmapline = c_plmapline(mapform : PlmapformCallback, name : PlcharVector, minx : Plflt, maxx : Plflt, miny : Plflt, maxy : Plflt, plotentries : PlintVector, nplotentries : Plint)
  fun plmapstring = c_plmapstring(mapform : PlmapformCallback, name : PlcharVector, string : PlcharVector, minx : Plflt, maxx : Plflt, miny : Plflt, maxy : Plflt, plotentries : PlintVector, nplotentries : Plint)
  fun plmaptex = c_plmaptex(mapform : PlmapformCallback, name : PlcharVector, dx : Plflt, dy : Plflt, just : Plflt, text : PlcharVector, minx : Plflt, maxx : Plflt, miny : Plflt, maxy : Plflt, plotentry : Plint)
  fun plmapfill = c_plmapfill(mapform : PlmapformCallback, name : PlcharVector, minx : Plflt, maxx : Plflt, miny : Plflt, maxy : Plflt, plotentries : PlintVector, nplotentries : Plint)
  fun plmeridians = c_plmeridians(mapform : PlmapformCallback, dlong : Plflt, dlat : Plflt, minlong : Plflt, maxlong : Plflt, minlat : Plflt, maxlat : Plflt)
  fun plmesh = c_plmesh(x : PlfltVector, y : PlfltVector, z : PlfltMatrix, nx : Plint, ny : Plint, opt : Plint)
  fun plmeshc = c_plmeshc(x : PlfltVector, y : PlfltVector, z : PlfltMatrix, nx : Plint, ny : Plint, opt : Plint, clevel : PlfltVector, nlevel : Plint)
  fun plmkstrm = c_plmkstrm(p_strm : PlintNcScalar)
  fun plmtex = c_plmtex(side : PlcharVector, disp : Plflt, pos : Plflt, just : Plflt, text : PlcharVector)
  fun plmtex3 = c_plmtex3(side : PlcharVector, disp : Plflt, pos : Plflt, just : Plflt, text : PlcharVector)
  fun plot3d = c_plot3d(x : PlfltVector, y : PlfltVector, z : PlfltMatrix, nx : Plint, ny : Plint, opt : Plint, side : Plbool)
  fun plot3dc = c_plot3dc(x : PlfltVector, y : PlfltVector, z : PlfltMatrix, nx : Plint, ny : Plint, opt : Plint, clevel : PlfltVector, nlevel : Plint)
  fun plot3dcl = c_plot3dcl(x : PlfltVector, y : PlfltVector, z : PlfltMatrix, nx : Plint, ny : Plint, opt : Plint, clevel : PlfltVector, nlevel : Plint, indexxmin : Plint, indexxmax : Plint, indexymin : PlintVector, indexymax : PlintVector)
  fun plpat = c_plpat(nlin : Plint, inc : PlintVector, del : PlintVector)
  fun plpath = c_plpath(n : Plint, x1 : Plflt, y1 : Plflt, x2 : Plflt, y2 : Plflt)
  fun plpoin = c_plpoin(n : Plint, x : PlfltVector, y : PlfltVector, code : Plint)
  fun plpoin3 = c_plpoin3(n : Plint, x : PlfltVector, y : PlfltVector, z : PlfltVector, code : Plint)
  fun plpoly3 = c_plpoly3(n : Plint, x : PlfltVector, y : PlfltVector, z : PlfltVector, draw : PlboolVector, ifcc : Plbool)
  alias PlboolVector = Plbool*
  fun plprec = c_plprec(setp : Plint, prec : Plint)
  fun plpsty = c_plpsty(patt : Plint)
  fun plptex = c_plptex(x : Plflt, y : Plflt, dx : Plflt, dy : Plflt, just : Plflt, text : PlcharVector)
  fun plptex3 = c_plptex3(wx : Plflt, wy : Plflt, wz : Plflt, dx : Plflt, dy : Plflt, dz : Plflt, sx : Plflt, sy : Plflt, sz : Plflt, just : Plflt, text : PlcharVector)
  fun plrandd = c_plrandd : Plflt
  fun plreplot = c_plreplot
  fun plrgbhls = c_plrgbhls(r : Plflt, g : Plflt, b : Plflt, p_h : PlfltNcScalar, p_l : PlfltNcScalar, p_s : PlfltNcScalar)
  fun plschr = c_plschr(def : Plflt, scale : Plflt)
  fun plscmap0 = c_plscmap0(r : PlintVector, g : PlintVector, b : PlintVector, ncol0 : Plint)
  fun plscmap0a = c_plscmap0a(r : PlintVector, g : PlintVector, b : PlintVector, alpha : PlfltVector, ncol0 : Plint)
  fun plscmap0n = c_plscmap0n(ncol0 : Plint)
  fun plscmap1 = c_plscmap1(r : PlintVector, g : PlintVector, b : PlintVector, ncol1 : Plint)
  fun plscmap1a = c_plscmap1a(r : PlintVector, g : PlintVector, b : PlintVector, alpha : PlfltVector, ncol1 : Plint)
  fun plscmap1l = c_plscmap1l(itype : Plbool, npts : Plint, intensity : PlfltVector, coord1 : PlfltVector, coord2 : PlfltVector, coord3 : PlfltVector, alt_hue_path : PlboolVector)
  fun plscmap1la = c_plscmap1la(itype : Plbool, npts : Plint, intensity : PlfltVector, coord1 : PlfltVector, coord2 : PlfltVector, coord3 : PlfltVector, alpha : PlfltVector, alt_hue_path : PlboolVector)
  fun plscmap1n = c_plscmap1n(ncol1 : Plint)
  fun plscmap1_range = c_plscmap1_range(min_color : Plflt, max_color : Plflt)
  fun plscol0 = c_plscol0(icol0 : Plint, r : Plint, g : Plint, b : Plint)
  fun plscol0a = c_plscol0a(icol0 : Plint, r : Plint, g : Plint, b : Plint, alpha : Plflt)
  fun plscolbg = c_plscolbg(r : Plint, g : Plint, b : Plint)
  fun plscolbga = c_plscolbga(r : Plint, g : Plint, b : Plint, alpha : Plflt)
  fun plscolor = c_plscolor(color : Plint)
  fun plscompression = c_plscompression(compression : Plint)
  fun plsdev = c_plsdev(devname : PlcharVector)
  fun plsdidev = c_plsdidev(mar : Plflt, aspect : Plflt, jx : Plflt, jy : Plflt)
  fun plsdimap = c_plsdimap(dimxmin : Plint, dimxmax : Plint, dimymin : Plint, dimymax : Plint, dimxpmm : Plflt, dimypmm : Plflt)
  fun plsdiori = c_plsdiori(rot : Plflt)
  fun plsdiplt = c_plsdiplt(xmin : Plflt, ymin : Plflt, xmax : Plflt, ymax : Plflt)
  fun plsdiplz = c_plsdiplz(xmin : Plflt, ymin : Plflt, xmax : Plflt, ymax : Plflt)
  fun plsdrawmode = c_plsdrawmode(mode : Plint)
  fun plseed = c_plseed(seed : LibC::UInt)
  fun plsesc = c_plsesc(esc : LibC::Char)
  fun plsfam = c_plsfam(fam : Plint, num : Plint, bmax : Plint)
  fun plsfci = c_plsfci(fci : Plunicode)
  fun plsfnam = c_plsfnam(fnam : PlcharVector)
  fun plsfont = c_plsfont(family : Plint, style : Plint, weight : Plint)
  fun plshade = c_plshade(a : PlfltMatrix, nx : Plint, ny : Plint, defined : PldefinedCallback, xmin : Plflt, xmax : Plflt, ymin : Plflt, ymax : Plflt, shade_min : Plflt, shade_max : Plflt, sh_cmap : Plint, sh_color : Plflt, sh_width : Plflt, min_color : Plint, min_width : Plflt, max_color : Plint, max_width : Plflt, fill : PlfillCallback, rectangular : Plbool, pltr : PltransformCallback, pltr_data : PlPointer)
  alias PldefinedCallback = (Plflt, Plflt -> Plint)
  alias PlfillCallback = (Plint, PlfltVector, PlfltVector -> Void)
  fun plshades = c_plshades(a : PlfltMatrix, nx : Plint, ny : Plint, defined : PldefinedCallback, xmin : Plflt, xmax : Plflt, ymin : Plflt, ymax : Plflt, clevel : PlfltVector, nlevel : Plint, fill_width : Plflt, cont_color : Plint, cont_width : Plflt, fill : PlfillCallback, rectangular : Plbool, pltr : PltransformCallback, pltr_data : PlPointer)
  fun plslabelfunc = c_plslabelfunc(label_func : PllabelFuncCallback, label_data : PlPointer)
  alias PllabelFuncCallback = (Plint, Plflt, PlcharNcVector, Plint, PlPointer -> Void)
  fun plsmaj = c_plsmaj(def : Plflt, scale : Plflt)
  fun plsmem = c_plsmem(maxx : Plint, maxy : Plint, plotmem : PlPointer)
  fun plsmema = c_plsmema(maxx : Plint, maxy : Plint, plotmem : PlPointer)
  fun plsmin = c_plsmin(def : Plflt, scale : Plflt)
  fun plsori = c_plsori(ori : Plint)
  fun plspage = c_plspage(xp : Plflt, yp : Plflt, xleng : Plint, yleng : Plint, xoff : Plint, yoff : Plint)
  fun plspal0 = c_plspal0(filename : PlcharVector)
  fun plspal1 = c_plspal1(filename : PlcharVector, interpolate : Plbool)
  fun plspause = c_plspause(pause : Plbool)
  fun plsstrm = c_plsstrm(strm : Plint)
  fun plssub = c_plssub(nx : Plint, ny : Plint)
  fun plssym = c_plssym(def : Plflt, scale : Plflt)
  fun plstar = c_plstar(nx : Plint, ny : Plint)
  fun plstart = c_plstart(devname : PlcharVector, nx : Plint, ny : Plint)
  fun plstransform = c_plstransform(coordinate_transform : PltransformCallback, coordinate_transform_data : PlPointer)
  fun plstring = c_plstring(n : Plint, x : PlfltVector, y : PlfltVector, string : PlcharVector)
  fun plstring3 = c_plstring3(n : Plint, x : PlfltVector, y : PlfltVector, z : PlfltVector, string : PlcharVector)
  fun plstripa = c_plstripa(id : Plint, pen : Plint, x : Plflt, y : Plflt)
  fun plstripc = c_plstripc(id : PlintNcScalar, xspec : PlcharVector, yspec : PlcharVector, xmin : Plflt, xmax : Plflt, xjump : Plflt, ymin : Plflt, ymax : Plflt, xlpos : Plflt, ylpos : Plflt, y_ascl : Plbool, acc : Plbool, colbox : Plint, collab : Plint, colline : PlintVector, styline : PlintVector, legline : PlcharMatrix, labx : PlcharVector, laby : PlcharVector, labtop : PlcharVector)
  fun plstripd = c_plstripd(id : Plint)
  fun plimagefr = c_plimagefr(idata : PlfltMatrix, nx : Plint, ny : Plint, xmin : Plflt, xmax : Plflt, ymin : Plflt, ymax : Plflt, zmin : Plflt, zmax : Plflt, valuemin : Plflt, valuemax : Plflt, pltr : PltransformCallback, pltr_data : PlPointer)
  fun plimage = c_plimage(idata : PlfltMatrix, nx : Plint, ny : Plint, xmin : Plflt, xmax : Plflt, ymin : Plflt, ymax : Plflt, zmin : Plflt, zmax : Plflt, dxmin : Plflt, dxmax : Plflt, dymin : Plflt, dymax : Plflt)
  fun plstyl = c_plstyl(nms : Plint, mark : PlintVector, space : PlintVector)
  fun plsurf3d = c_plsurf3d(x : PlfltVector, y : PlfltVector, z : PlfltMatrix, nx : Plint, ny : Plint, opt : Plint, clevel : PlfltVector, nlevel : Plint)
  fun plsurf3dl = c_plsurf3dl(x : PlfltVector, y : PlfltVector, z : PlfltMatrix, nx : Plint, ny : Plint, opt : Plint, clevel : PlfltVector, nlevel : Plint, indexxmin : Plint, indexxmax : Plint, indexymin : PlintVector, indexymax : PlintVector)
  fun plsvect = c_plsvect(arrowx : PlfltVector, arrowy : PlfltVector, npts : Plint, fill : Plbool)
  fun plsvpa = c_plsvpa(xmin : Plflt, xmax : Plflt, ymin : Plflt, ymax : Plflt)
  fun plsxax = c_plsxax(digmax : Plint, digits : Plint)
  fun plsyax = c_plsyax(digmax : Plint, digits : Plint)
  fun plsym = c_plsym(n : Plint, x : PlfltVector, y : PlfltVector, code : Plint)
  fun plszax = c_plszax(digmax : Plint, digits : Plint)
  fun pltext = c_pltext
  fun pltimefmt = c_pltimefmt(fmt : PlcharVector)
  fun plvasp = c_plvasp(aspect : Plflt)
  fun plvect = c_plvect(u : PlfltMatrix, v : PlfltMatrix, nx : Plint, ny : Plint, scale : Plflt, pltr : PltransformCallback, pltr_data : PlPointer)
  fun plvpas = c_plvpas(xmin : Plflt, xmax : Plflt, ymin : Plflt, ymax : Plflt, aspect : Plflt)
  fun plvpor = c_plvpor(xmin : Plflt, xmax : Plflt, ymin : Plflt, ymax : Plflt)
  fun plvsta = c_plvsta
  fun plw3d = c_plw3d(basex : Plflt, basey : Plflt, height : Plflt, xmin : Plflt, xmax : Plflt, ymin : Plflt, ymax : Plflt, zmin : Plflt, zmax : Plflt, alt : Plflt, az : Plflt)
  fun plwidth = c_plwidth(width : Plflt)
  fun plwind = c_plwind(xmin : Plflt, xmax : Plflt, ymin : Plflt, ymax : Plflt)
  fun plxormod = c_plxormod(mode : Plbool, status : PlboolNcScalar)
  alias PlboolNcScalar = Plbool*
  fun plsetopt = c_plsetopt(opt : PlcharVector, optarg : PlcharVector) : Plint
  fun plparseopts = c_plparseopts(p_argc : LibC::Int*, argv : PlcharNcMatrix, mode : Plint) : Plint
  alias PlcharNcMatrix = LibC::Char**
end
