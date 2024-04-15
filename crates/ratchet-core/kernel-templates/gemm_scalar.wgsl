fn getAIndexFromCoords3D(coords : vec3<i32>) -> i32 {
    return dot(coords, metadata.aStrides);
}

fn getBIndexFromCoords3D(coords : vec3<i32>) -> i32 {
    return dot(coords, metadata.bStrides);
}

fn getOutputIndexFromCoords(coords : vec3<i32>) -> i32 {
  return dot(coords, metadata.outShapeStrides);
}
        
fn setOutputAtIndex(flatIndex: i32, value: f32) {
    result[flatIndex] = f32(value);
}

fn setOutputAtCoords(d0: i32, d1: i32, d2: i32, value: f32) {
    let flatIndex = getOutputIndexFromCoords(vec3<i32>(d0, d1, d2));
    setOutputAtIndex(flatIndex, value);
}

{% if QUANT %}
    fn getA(d0 : i32, d1 : i32, d2 : i32) -> vec4<f32> {
        return unpack4x8snorm(A[getAIndexFromCoords3D(vec3<i32>(d0,d1,d2)) / 4]);
    }

    fn getAbsMax(d0 : i32, d1 : i32, d2 : i32) -> f32 {
        let abs_index = getAIndexFromCoords3D(vec3<i32>(d0,d1,d2)) / 32;
        return scale[abs_index]; 
    }
{% else %}
    fn getA(d0: i32, d1: i32, d2: i32) -> f32 {
        return f32(A[getAIndexFromCoords3D(vec3<i32>(d0, d1, d2))]);
    }
{% endif %}
   
fn getB(d0: i32, d1: i32, d2: i32) -> f32 {
    return f32(B[getBIndexFromCoords3D(vec3<i32>(d0, d1, d2))]);
}
   
{% if FIT_A_OUTER and FIT_INNER %}
    {% if QUANT %}
        fn mm_readA(batch: i32, row: i32, col: i32) -> vec4<f32> {
            var value = vec4<f32>(0.0);
    {% else %}
        fn mm_readA(batch: i32, row: i32, col: i32) -> f32 {
            var value = f32(0.0);
    {% endif %}
        {% if TRANS_A %}
            value = getA(batch, col, row);
        {% else %}
            value = getA(batch, row, col);
        {% endif %}
        return value;
    }
{% else %}
    {% if QUANT %}
        fn mm_readA(batch: i32, row: i32, col: i32) -> vec4<f32> {
            var value = vec4<f32>(0.0);
    {% else %}
        fn mm_readA(batch: i32, row: i32, col: i32) -> f32 {
            var value = f32(0.0);
    {% endif %}
        {% if TRANS_A %}
            if (row < metadata.aShape.z && col < metadata.aShape.y) {
                value = getA(batch, col, row);
            }
        {% else %}
            if (row < metadata.aShape.y && col < metadata.aShape.z) {
                value = getA(batch, row, col);
            }
        {% endif %}
        return value;
    }
{% endif %}

{% if FIT_B_OUTER and FIT_INNER %}
fn mm_readB(batch: i32, row: i32, col: i32) -> f32 {
    var value = f32(0.0);
    {% if TRANS_B %}
        value = getB(batch, col, row);
    {% else %}
        value = getB(batch, row, col);
    {% endif %}
    return value;
}
{% else %}
fn mm_readB(batch: i32, row: i32, col: i32) -> f32 {
    var value = f32(0.0);
    {% if TRANS_B %}
        if (row < metadata.bShape.z && col < metadata.bShape.y) {
            value = getB(batch, col, row);
        }
    {% else %}
        if (row < metadata.bShape.y && col < metadata.bShape.z) {
            value = getB(batch, row, col);
        }
    {% endif %}
    return value;
}
{% endif %}

fn mm_write(batch: i32, row: i32, col: i32, valueIn: f32) {
{% if FIT_A_OUTER and FIT_B_OUTER %}
        var value = valueIn;
        let coords = vec3<i32>(batch, row, col);
        setOutputAtCoords(coords[0], coords[1], coords[2], value);
{% else %}
    if (row < metadata.dimAOuter && col < metadata.dimBOuter) {
        var value = valueIn;
        let coords = vec3<i32>(batch, row, col);
        setOutputAtCoords(coords[0], coords[1], coords[2], valueIn);
    }
{% endif %}
}

var<private> localId: vec3<u32>;
var<private> globalId: vec3<u32>;
var<private> workgroupId: vec3<u32>;

{% if QUANT %}
    @group(0) @binding(0) var<storage, read> A: array<u32>;
    @group(0) @binding(1) var<storage, read> scale: array<f32>;
    @group(0) @binding(2) var<storage, read> B: array<f32>;

    {% if BIAS %}
        @group(0) @binding(3) var<storage, read> bias: array<f32>;
        @group(0) @binding(4) var<storage, read_write> result: array<f32>;
    {% else %}
        @group(0) @binding(3) var<storage, read_write> result: array<f32>;
    {% endif %}

{% else %}
    @group(0) @binding(0) var<storage, read> A: array<f32>;
    @group(0) @binding(1) var<storage, read> B: array<f32>;

    {% if BIAS %}
        @group(0) @binding(2) var<storage, read> bias: array<f32>;
        @group(0) @binding(3) var<storage, read_write> result: array<f32>;
    {% else %}
        @group(0) @binding(2) var<storage, read_write> result: array<f32>;
    {% endif %}
{% endif %}


@group(1) @binding(0) var<uniform> metadata: Meta;

struct Meta {
    aShape: vec3<i32>,
    aStrides: vec3<i32>,
    bShape: vec3<i32>,
    bStrides: vec3<i32>,
    outShape: vec3<i32>,
    outShapeStrides: vec3<i32>,
    dimAOuter: i32,
    dimBOuter: i32,
    dimInner: i32,
}
  
var<workgroup> mm_Asub : array<array<f32, 32>, 32>;
var<workgroup> mm_Bsub : array<array<f32, 32>, 32>;

@compute @workgroup_size(8,8,1) 
fn main(@builtin(local_invocation_id) localId : vec3<u32>,
        @builtin(global_invocation_id) globalId : vec3<u32>,
        @builtin(workgroup_id) workgroupId : vec3<u32>) {
    let batch = i32(globalId.z);
    let batchA = batch % metadata.aShape[0];
    let batchB = batch % metadata.bShape[0];

    let tileRow = i32(localId.y) * {{ ROW_PER_THREAD }};
    let tileCol = i32(localId.x) * {{ ROW_PER_THREAD }};

    let globalRowStart = i32(workgroupId.y) * {{ TILE_DIM }};
    let globalRow = i32(globalId.y) * {{ ROW_PER_THREAD }};
    let globalCol = i32(globalId.x) * {{ ROW_PER_THREAD }};

    let numTiles = (metadata.dimInner - 1) / {{ TILE_DIM }} + 1;
    var kStart = 0;

    var acc: array<array<f32, 4>, 4>;

    let tileRowA = i32(localId.y) * {{ ROW_PER_THREAD }};
    let tileColA = i32(localId.x) * {{ ROW_PER_THREAD }};
    let tileRowB = i32(localId.y) * {{ ROW_PER_THREAD }};
    // Loop over shared dimension.
    for (var t = 0; t < numTiles; t++) {
        // Load one tile of A into local memory.
        for (var innerRow = 0; innerRow < 4; innerRow++) {
            {% if QUANT %}
                let curRow = globalRow + innerRow;
                let curCol = kStart + i32(localId.x) * 4;

                let absmax = getAbsMax(batchA, curRow, curCol);
                let val = mm_readA(batchA, curRow, curCol) * absmax;
                {% for i in range(end=4) %}
                    mm_Asub[tileRowA + innerRow][tileColA + {{ i }}] = val[{{ i }}]; 
                {% endfor %}
            {% else %}
                for (var innerCol = 0; innerCol < 4; innerCol++) {
                    let inputRow = tileRowA + innerRow;
                    let inputCol = tileColA + innerCol;

                    mm_Asub[inputRow][inputCol] = mm_readA(batchA,
                        globalRowStart + inputRow,
                        kStart + inputCol);
                }
            {% endif %}
        }

        // Load one tile of B into local memory.
        for (var innerRow = 0; innerRow < 4; innerRow++) {
            for (var innerCol = 0; innerCol < 4; innerCol++) {
                let inputRow = tileRowB + innerRow;
                let inputCol = tileCol + innerCol;

                mm_Bsub[inputRow][inputCol] = mm_readB(batchB,
                    kStart + inputRow,
                    globalCol + innerCol);
            }
        }
        kStart = kStart + {{ TILE_DIM }};
        workgroupBarrier();

        for (var k = 0; k < {{ TILE_DIM }}; k++) {
            let BCached0 = mm_Bsub[k][tileCol + 0];
            let BCached1 = mm_Bsub[k][tileCol + 1];
            let BCached2 = mm_Bsub[k][tileCol + 2];
            let BCached3 = mm_Bsub[k][tileCol + 3];

            for (var innerRow = 0; innerRow < 4; innerRow++) {
                let ACached = mm_Asub[tileRow + innerRow][k];
                acc[innerRow][0] = fma(ACached, BCached0, acc[innerRow][0]);
                acc[innerRow][1] = fma(ACached, BCached1, acc[innerRow][1]);
                acc[innerRow][2] = fma(ACached, BCached2, acc[innerRow][2]);
                acc[innerRow][3] = fma(ACached, BCached3, acc[innerRow][3]);
            }
        }


        workgroupBarrier();
    }

    var val: f32;
    {% for row in range(end=ROW_PER_THREAD) -%}
        {%- for col in range(end=4) -%}
            {%- if BIAS %}
                {% if TRANS_OUT %}
                    val = acc[{{ row }}][{{ col }}] + bias[globalRow + {{ row }}];
                {% else %}
                    val = acc[{{ row }}][{{ col }}] + bias[globalCol + {{ col }}];
                {% endif %}
            {%- else %}
                val = acc[{{ row }}][{{ col }}];
            {%- endif %}

            {%- if TRANS_OUT %}
                mm_write(batch, globalCol + {{ col }}, globalRow + {{ row }}, val);
            {%- else %}
                mm_write(batch, globalRow + {{ row }}, globalCol + {{ col }}, val);
            {%- endif %}
        {%- endfor -%}
    {%- endfor -%}
} 
