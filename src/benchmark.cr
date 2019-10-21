require "./jug/*"
require "./flask/*"
require "benchmark"

macro benchmark_helper(sizes, operations, min, max)
  {% for size in sizes %}
    arr_i32_{{size}} = Flask.random({{min.id}}...{{max.id}}, {{size.id}})
    arr_f32_{{size}} = Flask.random({{min.id}}_f32...{{max.id}}_f32, {{size.id}})
    arr_f64_{{size}} = Flask.random({{min.id}}_f64...{{max.id}}_f64, {{size.id}})
  {% end %}

  Benchmark.ips do |b|
    {% for op in operations %}
      {% for size in sizes %}
        b.report("i32_#{{{size}}}_{{op.id}}") { arr_i32_{{size}}.{{op.id}} }
        b.report("f32_#{{{size}}}_{{op.id}}") { arr_f32_{{size}}.{{op.id}} }
        b.report("f64_#{{{size}}}_{{op.id}}") { arr_f64_{{size}}.{{op.id}} }
      {% end %}
    {% end %}
  end
end

benchmark_helper [10, 100, 500, 1000, 10000], [cumsum, cumprod, sum, prod], 0, 2
