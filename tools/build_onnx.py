"""가중치 npz -> T 단계를 펼친 순수 텐서 ONNX (opset 17). onnx + numpy 만 필요 (torch 불필요).

뉴런 (sinabs IAF, MultiSpike + MembraneSubtract, threshold 1.0, min_v_mem -1.0) 을 학습 코드와 같은 순서로 쓴다:
    v += x_t ;  s = floor(max(v, 0) / th) ;  v -= s * th ;  v = relu(v - m) + m      (m = min_v_mem)
컨볼루션·풀링은 시간에 상태가 없으므로 [B*T, C, H, W] 로 한 번에 계산하고 IAF 만 T 단계로 펼친다 (수식은 단계별 계산과 같다).
SumPool2d(2) = AveragePool(2) * 4 (2 의 거듭제곱이라 부동소수 오차 없음).
입력 [B, T, 2, 32, 32] float32, 출력 logits [B, 10] (= T 평균, 원래 temporal_reduction='mean'),
--spikes 면 두 번째 출력 layer_spikes [B, 9] (층별 출력 스파이크 합, T 전체).
"""
import argparse, json
import numpy as np
import onnx
from onnx import helper as H, numpy_helper as N, TensorProto as TP

# (종류, 입력채널, 출력채널, 커널, stride, pad) — pyramid_late64 9층, 풀링은 SumPool2d(2) (conv0, conv1, conv2 뒤)
CONV_IDX = [0, 4, 7, 10, 12, 14, 16, 18]   # nn.Sequential 안 위치 (가중치 키 w<idx>)
POOL_AFTER = {0, 4, 7}
STRIDE = {10: 2}
IAF_IDX = {0: 2, 4: 5, 7: 8, 10: 11, 12: 13, 14: 15, 16: 17, 18: 19, 21: 22}   # conv/linear 위치 -> 뒤따르는 IAF 위치


class G:
    def __init__(self):
        self.nodes, self.inits, self.n = [], [], 0

    def name(self, p):
        self.n += 1
        return f"{p}_{self.n}"

    def const(self, arr, p="c"):
        nm = self.name(p)
        self.inits.append(N.from_array(np.asarray(arr), nm))
        return nm

    def op(self, typ, ins, nout=1, **attrs):
        outs = [self.name(typ.lower()) for _ in range(nout)]
        self.nodes.append(H.make_node(typ, ins, outs, **attrs))
        return outs[0] if nout == 1 else outs


def iaf_scan(g, z, B, T, shape_tail, th, mv):
    """z: [B*T, *tail] -> (스파이크 [B*T, *tail], 층 합 [B])"""
    five = g.op("Reshape", [z, g.const(np.array([B, T] + list(shape_tail), dtype=np.int64))])
    xs = g.op("Split", [five], nout=T, axis=1)
    one = g.const(np.float32(th), "th") if th != 1.0 else None; zero = g.const(np.float32(0.0), "zero"); mvc = g.const(np.float32(mv), "mv")
    v = None; outs = []
    for t in range(T):
        x = xs[t]
        v = x if v is None else g.op("Add", [v, x])
        s = g.op("Floor", [g.op("Div", [g.op("Max", [v, zero]), one])]) if th != 1.0 else g.op("Floor", [g.op("Max", [v, zero])])
        v = g.op("Sub", [v, s if th == 1.0 else g.op("Mul", [s, one])])
        v = g.op("Add", [g.op("Relu", [g.op("Sub", [v, mvc])]), mvc])
        outs.append(s)
    sp = g.op("Concat", outs, axis=1)                                    # [B, T, *tail]
    tot = g.op("ReduceSum", [sp, g.const(np.arange(1, 2 + len(shape_tail), dtype=np.int64))], keepdims=0)   # [B]
    flat = g.op("Reshape", [sp, g.const(np.array([B * T] + list(shape_tail), dtype=np.int64))])
    return flat, tot, sp


def build(w, T, B, spikes):
    g = G()
    x = H.make_tensor_value_info("input", TP.FLOAT, [B, T, 2, 32, 32])
    h = g.op("Reshape", ["input", g.const(np.array([B * T, 2, 32, 32], dtype=np.int64))])
    sizes = {0: 32, 4: 16, 7: 8, 10: 4, 12: 2, 14: 2, 16: 2, 18: 2}
    tots = []
    for i in CONV_IDX:
        W = w[f"w{i}"]; k = W.shape[2]
        stride = STRIDE.get(i, 1); pad = k // 2
        h = g.op("Conv", [h, g.const(W, f"w{i}")], kernel_shape=[k, k], strides=[stride, stride], pads=[pad] * 4)
        if i == 0:   # BatchNorm (use_bn1, eval 모드, running 통계)
            h = g.op("BatchNormalization", [h, g.const(w["bn1_g"], "bng"), g.const(w["bn1_b"], "bnb"), g.const(w["bn1_m"], "bnm"), g.const(w["bn1_v"], "bnv")],
                     epsilon=float(w["bn1_eps"]))
        C = W.shape[0]; hw = sizes[i] // stride
        h, tot, _ = iaf_scan(g, h, B, T, [C, hw, hw], float(w[f"th{IAF_IDX[i]}"]), float(w[f"mv{IAF_IDX[i]}"]))
        tots.append(tot)
        if i in POOL_AFTER:
            h = g.op("AveragePool", [h], kernel_shape=[2, 2], strides=[2, 2])
            h = g.op("Mul", [h, g.const(np.float32(4.0), "k2")])
    h = g.op("Reshape", [h, g.const(np.array([B * T, 512], dtype=np.int64))])             # Flatten (C,H,W 순서)
    h = g.op("Gemm", [h, g.const(w["w21"], "wfc")], transB=1)
    h, tot, sp = iaf_scan(g, h, B, T, [10], float(w["th22"]), float(w["mv22"]))
    tots.append(tot)
    # 출력 로짓 = T 평균 (원래 temporal_reduction='mean')
    logits = g.op("Div", [g.op("ReduceSum", [sp, g.const(np.array([1], dtype=np.int64))], keepdims=0), g.const(np.float32(T), "T")])
    g.nodes[-1].output[0] = "logits"
    outs = [H.make_tensor_value_info("logits", TP.FLOAT, [B, 10])]
    if spikes:
        lt = g.op("Concat", [g.op("Reshape", [t_, g.const(np.array([B, 1], dtype=np.int64))]) for t_ in tots], axis=1)
        g.nodes[-1].output[0] = "layer_spikes"
        outs.append(H.make_tensor_value_info("layer_spikes", TP.FLOAT, [B, 9]))
    graph = H.make_graph(g.nodes, f"vggsnn_T{T}_B{B}", [x], outs, g.inits)
    m = H.make_model(graph, opset_imports=[H.make_opsetid("", 17)], producer_name="eip-jetson-bundle")
    m.ir_version = 8
    onnx.checker.check_model(m)
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("weights"); ap.add_argument("out"); ap.add_argument("--T", type=int, required=True)
    ap.add_argument("--batch", type=int, default=1); ap.add_argument("--spikes", action="store_true")
    a = ap.parse_args()
    w = dict(np.load(a.weights))
    onnx.save(build(w, a.T, a.batch, a.spikes), a.out)
