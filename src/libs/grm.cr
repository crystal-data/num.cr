@[Link("GRM")]
lib LibGRM
  fun args_new = grm_args_new : ArgsT
  type ArgsT = Void*
  fun args_delete = grm_args_delete(args : ArgsT)
  fun args_push = grm_args_push(args : ArgsT, key : LibC::Char*, value_format : LibC::Char*, ...) : LibC::Int
  fun args_push_buf = grm_args_push_buf(args : ArgsT, key : LibC::Char*, value_format : LibC::Char*, buffer : Void*, apply_padding : LibC::Int) : LibC::Int
  fun args_contains = grm_args_contains(args : ArgsT, keyword : LibC::Char*) : LibC::Int
  fun args_clear = grm_args_clear(args : ArgsT)
  fun args_remove = grm_args_remove(args : ArgsT, key : LibC::Char*)
  fun length = grm_length(value : LibC::Double, unit : LibC::Char*) : ArgsPtrT
  alias ArgsPtrT = ArgsT
  fun dump = grm_dump(args : ArgsT, f : File*)

  struct X_IoFile
    _flags : LibC::Int
    _io_read_ptr : LibC::Char*
    _io_read_end : LibC::Char*
    _io_read_base : LibC::Char*
    _io_write_base : LibC::Char*
    _io_write_ptr : LibC::Char*
    _io_write_end : LibC::Char*
    _io_buf_base : LibC::Char*
    _io_buf_end : LibC::Char*
    _io_save_base : LibC::Char*
    _io_backup_base : LibC::Char*
    _io_save_end : LibC::Char*
    _markers : X_IoMarker*
    _chain : X_IoFile*
    _fileno : LibC::Int
    _flags2 : LibC::Int
    _old_offset : X__OffT
    _cur_column : LibC::UShort
    _vtable_offset : LibC::Char
    _shortbuf : LibC::Char[1]
    _lock : X_IoLockT*
    _offset : X__Off64T
    _codecvt : X_IoCodecvt*
    _wide_data : X_IoWideData*
    _freeres_list : X_IoFile*
    _freeres_buf : Void*
    __pad5 : LibC::SizeT
    _mode : LibC::Int
    _unused2 : LibC::Char[20]
  end

  type File = X_IoFile
  alias X_IoMarker = Void
  alias X__OffT = LibC::Long
  alias X_IoLockT = Void
  alias X__Off64T = LibC::Long
  alias X_IoCodecvt = Void
  alias X_IoWideData = Void
  fun dump_json = grm_dump_json(args : ArgsT, f : File*)
  fun dump_json_str = grm_dump_json_str : LibC::Char*
  fun register = grm_register(type : EventTypeT, callback : EventCallbackT) : LibC::Int
  enum EventTypeT
    GrmEventNewPlot     = 0
    GrmEventUpdatePlot  = 1
    GrmEventSize        = 2
    GrmEventMergeEnd    = 3
    X_GrmEventTypeCount = 4
  end

  union EventT
    new_plot_event : NewPlotEventT
    size_event : SizeEventT
    update_plot_event : UpdatePlotEventT
    merge_end_event : MergeEndEventT
  end

  alias EventCallbackT = (EventT* -> Void)

  struct NewPlotEventT
    type : EventTypeT
    plot_id : LibC::Int
  end

  struct SizeEventT
    type : EventTypeT
    plot_id : LibC::Int
    width : LibC::Int
    height : LibC::Int
  end

  struct UpdatePlotEventT
    type : EventTypeT
    plot_id : LibC::Int
  end

  struct MergeEndEventT
    type : EventTypeT
    identificator : LibC::Char*
  end

  fun unregister = grm_unregister(type : EventTypeT) : LibC::Int
  fun input = grm_input(x0 : ArgsT) : LibC::Int
  fun get_box = grm_get_box(x0 : LibC::Int, x1 : LibC::Int, x2 : LibC::Int, x3 : LibC::Int, x4 : LibC::Int, x5 : LibC::Int*, x6 : LibC::Int*, x7 : LibC::Int*, x8 : LibC::Int*) : LibC::Int
  fun input = grm_input(input_args : ArgsT) : LibC::Int
  fun get_box = grm_get_box(x1 : LibC::Int, y1 : LibC::Int, x2 : LibC::Int, y2 : LibC::Int, keep_aspect_ratio : LibC::Int, x : LibC::Int*, y : LibC::Int*, w : LibC::Int*, h : LibC::Int*) : LibC::Int
  fun get_tooltip = grm_get_tooltip(x0 : LibC::Int, x1 : LibC::Int) : TooltipInfoT*

  struct TooltipInfoT
    x : LibC::Double
    y : LibC::Double
    x_px : LibC::Int
    y_px : LibC::Int
    xlabel : LibC::Char*
    ylabel : LibC::Char*
    label : LibC::Char*
  end

  fun open = grm_open(is_receiver : LibC::Int, name : LibC::Char*, id : LibC::UInt, custom_recv : (LibC::Char*, LibC::UInt -> LibC::Char*), custom_send : (LibC::Char*, LibC::UInt, LibC::Char* -> LibC::Int)) : Void*
  fun recv = grm_recv(p : Void*, args : ArgsT) : ArgsT
  fun send = grm_send(p : Void*, data_desc : LibC::Char*, ...) : LibC::Int
  fun send_buf = grm_send_buf(p : Void*, data_desc : LibC::Char*, buffer : Void*, apply_padding : LibC::Int) : LibC::Int
  fun send_ref = grm_send_ref(p : Void*, key : LibC::Char*, format : LibC::Char, ref : Void*, len : LibC::Int) : LibC::Int
  fun send_args = grm_send_args(p : Void*, args : ArgsT) : LibC::Int
  fun close = grm_close(p : Void*)
  fun finalize = grm_finalize
  fun clear = grm_clear : LibC::Int
  fun max_plotid = grm_max_plotid : LibC::UInt
  fun merge = grm_merge(args : ArgsT) : LibC::Int
  fun merge_extended = grm_merge_extended(args : ArgsT, hold : LibC::Int, identificator : LibC::Char*) : LibC::Int
  fun merge_hold = grm_merge_hold(args : ArgsT) : LibC::Int
  fun merge_named = grm_merge_named(args : ArgsT, identificator : LibC::Char*) : LibC::Int
  fun plot = grm_plot(args : ArgsT) : LibC::Int
  fun switch = grm_switch(id : LibC::UInt) : LibC::Int
end
