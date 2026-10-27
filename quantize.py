"""Build the other precisions from the exported float32 graphs (export_onnx.py), in the same folder:

  enroll_fp16.onnx, chunk_fp16.onnx   float16 weights with a Cast back to float32, math stays float32
  chunk_int8.onnx                     LSTMs in 8-bit (dynamic quantization)
  chunk_int8_full.onnx                LSTMs and weight matrix multiplies in 8-bit

    python quantize.py onnx/
"""
import sys, tempfile
from pathlib import Path
import numpy as np, onnx
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.quantization import QuantType, quantize_dynamic


def fp16_weights(src, dst):
    m = onnx.load(src)
    g = m.graph
    casts, keep = [], []
    for init in list(g.initializer):
        a = numpy_helper.to_array(init)
        if a.dtype == np.float32 and a.size >= 64:
            keep.append(numpy_helper.from_array(a.astype(np.float16), init.name + "_f16"))
            casts.append(helper.make_node("Cast", [init.name + "_f16"], [init.name], to=TensorProto.FLOAT, name=init.name + "_cast"))
            g.initializer.remove(init)
    g.initializer.extend(keep)
    nodes = list(g.node); del g.node[:]; g.node.extend(casts + nodes)
    onnx.save(m, dst)


def gemm_to_matmul(m):
    """Gemm -> MatMul (+ Add) with the weight transposed once per shared weight. quantize_dynamic's own Gemm handling
    mishandles weights shared between Gemm nodes."""
    g = m.graph
    init = {i.name: i for i in g.initializer}
    made, nodes = set(), []

    def const(name, arr):
        if name not in made:
            g.initializer.append(numpy_helper.from_array(np.ascontiguousarray(arr.astype(np.float32)), name)); made.add(name)
        return name

    for n in g.node:
        if n.op_type != "Gemm":
            nodes.append(n); continue
        at = {a.name: helper.get_attribute_value(a) for a in n.attribute}
        assert at.get("transA", 0) == 0, n.name
        alpha, beta, tb = at.get("alpha", 1.0), at.get("beta", 1.0), at.get("transB", 0)
        W = numpy_helper.to_array(init[n.input[1]])
        wn = const(f"{n.input[1]}_mm_t{tb}_a{alpha}", (W.T if tb else W) * alpha)
        has_c = len(n.input) > 2 and n.input[2]
        mm_out = n.output[0] + "_mm" if has_c else n.output[0]
        nodes.append(helper.make_node("MatMul", [n.input[0], wn], [mm_out], name=n.name + "_MatMul"))
        if has_c:
            cn = const(f"{n.input[2]}_mm_b{beta}", numpy_helper.to_array(init[n.input[2]]) * beta)
            nodes.append(helper.make_node("Add", [mm_out, cn], [n.output[0]], name=n.name + "_Add"))
    del g.node[:]; g.node.extend(nodes)
    return m


def main(d):
    d = Path(d)
    fp16_weights(d / "enroll.onnx", d / "enroll_fp16.onnx")
    fp16_weights(d / "chunk.onnx", d / "chunk_fp16.onnx")
    quantize_dynamic(d / "chunk.onnx", d / "chunk_int8.onnx", weight_type=QuantType.QInt8, op_types_to_quantize=["LSTM"])
    m = gemm_to_matmul(onnx.load(d / "chunk.onnx"))
    weights = {i.name for i in m.graph.initializer}
    # Attention products (both sides computed per call) stay float: one per-tensor scale would cover all 32 bands.
    dynamic = [n.name for n in m.graph.node if n.op_type == "MatMul" and n.input[1] not in weights]
    with tempfile.TemporaryDirectory() as t:
        onnx.save(m, f"{t}/mm.onnx")
        quantize_dynamic(f"{t}/mm.onnx", d / "chunk_int8_full.onnx", weight_type=QuantType.QInt8,
                         op_types_to_quantize=["MatMul", "LSTM"], nodes_to_exclude=dynamic)
    for f in sorted(d.glob("*.onnx")):
        print(f"{f.name:22s} {f.stat().st_size / 1e6:6.1f} MB")


if __name__ == "__main__":
    main(sys.argv[1])
