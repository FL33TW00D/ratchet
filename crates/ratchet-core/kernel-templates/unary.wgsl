{% if inplace %}

@group(0) @binding(0)
var<storage, read_write> X: array<{{ elem }}>;

{% else %}
@group(0) @binding(0)
var<storage, read> X: array<{{ elem }}>;

@group(0) @binding(1)
var<storage, read_write> Y: array<{{ elem }}>;
{% endif %}

struct Meta {
    numel: u32,
}

@group(1) @binding(0)
var<uniform> metadata: Meta;

{% if elem == "vec4<f32>" -%}
    const NORM_CONST: vec4<f32> = vec4<f32>(0.5f, 0.5f, 0.5f, 0.5f);
    const SQRT_2_OVER_PI: vec4<f32> = vec4<f32>(0.7978845608028654f, 0.7978845608028654f, 0.7978845608028654f, 0.7978845608028654f);
    const SCALED_SQRT_2_OVER_PI: vec4<f32> = vec4<f32>(0.035677408136300125f, 0.035677408136300125f, 0.035677408136300125f, 0.035677408136300125f);
    const TANH_LIMIT: vec4<f32> = vec4<f32>(10.0f, 10.0f, 10.0f, 10.0f);
    const RELU_CONST: vec4<f32> = vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f);
{%- elif elem == "vec2<f32>" -%}
    const NORM_CONST: vec2<f32> = vec2<f32>(0.5f, 0.5f);
    const SQRT_2_OVER_PI: vec2<f32> = vec2<f32>(0.7978845608028654f, 0.7978845608028654f);
    const SCALED_SQRT_2_OVER_PI: vec2<f32> = vec2<f32>(0.035677408136300125f, 0.035677408136300125f);
    const TANH_LIMIT: vec2<f32> = vec2<f32>(10.0f, 10.0f);
    const RELU_CONST: vec2<f32> = vec2<f32>(0.0f, 0.0f);
{%- else -%}
    const NORM_CONST: f32 = 0.5f; 
    const SQRT_2_OVER_PI: f32 = 0.7978845608028654f;
    const SCALED_SQRT_2_OVER_PI: f32 = 0.035677408136300125f;
    const TANH_LIMIT: f32 = 10.0f;
    const RELU_CONST: f32 = 0.0f;
{% endif %}


//Tanh is broken for large values on MSL
fn safe_tanh(x: {{ elem }}) -> {{ elem }} {
    return select(tanh(x), sign(x), abs(x) >= TANH_LIMIT);
}

fn gelu(val: {{ elem }}) -> {{ elem }} {
    let cdf = NORM_CONST + NORM_CONST * safe_tanh(val * (SCALED_SQRT_2_OVER_PI * (val * val) + SQRT_2_OVER_PI));
    return val * cdf;
}

fn relu(val: {{ elem }}) -> {{ elem }} {
    return max(val, RELU_CONST);
}

@compute @workgroup_size(8,8,1)
fn main( 
        @builtin(local_invocation_index) local_index: u32,
        @builtin(workgroup_id) group_id: vec3<u32>,
        @builtin(num_workgroups) num_groups: vec3<u32>
) {

    let x_offset = group_id.x * 64u;
    let index = (group_id.y * num_groups.x * 64u) + x_offset + local_index;
    if (index >= metadata.numel / {{ elem_size }}u) {
        return;
    }
    {% if inplace %}
        let val = X[index];
        X[index] = {{func}}(val);
    {% else %}
        Y[index] = {{func}}(X[index]);
    {% endif %}
}

