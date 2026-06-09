"""Interactive HTML roofline report.

Bakes a model's config-derived constants + quant coefficients + hardware peaks
into a single self-contained HTML page that re-runs the (closed-form) cost model
**in the browser**: drag the parallelism / workload knobs (tp, dp, batch,
seq_len, chunk, phase, sequence-parallel) and the roofline, the per-category cost
table and the bottleneck summary update live. No server, no external JS libs
(vanilla + canvas), works offline.

The JS cost model mirrors ``descriptors._mimo_v2_family`` + ``parallelism`` +
``ops`` exactly (same op list, same formulas, same mesh semantics: tensor axis
t = tp//dp, fused-MoE EP = devices, SP reduce-scatter/all-gather gated on
should_scatter). It is the closed-form roofline only (View B/C/D); the jaxpr
View F needs a real trace and stays Python-side.
"""

from __future__ import annotations

import json
from collections import Counter

from . import quant
from .report import HardwarePeaks


def _cfg(config, *names, default=None):
    for n in names:
        if config.get(n) is not None:
            return config[n]
    return default


def _quant_bake(qs):
    """Per-role effective byte coefficients (per element) + peak kind, sampled at
    a large block so block-scale overhead is linearised."""
    K = N = M = 4096
    out = {}
    for role, q in qs.items():
        out[role] = {
            "wbpe": q.w_bytes(K, N) / (K * N),
            "wspe": q.weight_scale_bytes(K, N) / (K * N),
            "abpe": q.a_bytes(M, K) / (M * K),
            "aspe": q.act_scale_bytes(M, K) / (M * K),
            "peak": q.peak_kind(),
        }
    return out


def _bake(arch, config, peaks: HardwarePeaks, defaults: dict) -> dict:
    H = _cfg(config, "hidden_size")
    L = _cfg(config, "num_hidden_layers")
    hlp = _cfg(config, "hybrid_layer_pattern", default=[0] * L)
    mlf = _cfg(config, "moe_layer_freq", default=[1] * L)

    def is_swa(i):
        return bool(hlp[i]) if i < len(hlp) else False

    def is_moe(i):
        return bool(mlf[i]) if i < len(mlf) else True

    combo = Counter((is_swa(i), is_moe(i)) for i in range(L))
    qs = quant.quant_specs_from_config(config)
    full = dict(
        nh=_cfg(config, "num_attention_heads"),
        nkv=_cfg(config, "num_key_value_heads"),
        hd=_cfg(config, "head_dim"),
        vhd=_cfg(config, "v_head_dim", default=_cfg(config, "head_dim")),
        window=0,
    )
    swa = dict(
        nh=_cfg(config, "swa_num_attention_heads", default=full["nh"]),
        nkv=_cfg(config, "swa_num_key_value_heads", default=full["nkv"]),
        hd=_cfg(config, "swa_head_dim", default=full["hd"]),
        vhd=_cfg(config, "swa_v_head_dim", default=full["vhd"]),
        window=_cfg(config, "sliding_window_size", default=4096),
    )
    return {
        "arch": arch,
        "H": H,
        "L": L,
        "VOCAB": _cfg(config, "vocab_size"),
        "NEXP": _cfg(config, "n_routed_experts", "num_experts", default=8),
        "TOPK": _cfg(config, "num_experts_per_tok", default=2),
        "MOEF": _cfg(config, "moe_intermediate_size", default=_cfg(config, "intermediate_size")),
        "DENSE_F": _cfg(config, "intermediate_size"),
        "full": full,
        "swa": swa,
        "n_full": combo[(False, True)] + combo[(False, False)],
        "n_swa": combo[(True, True)] + combo[(True, False)],
        "n_moe": combo[(False, True)] + combo[(True, True)],
        "n_dense": combo[(False, False)] + combo[(True, False)],
        "quant": _quant_bake(qs),
        "peaks": {
            "bf16_tflops": peaks.bf16_tflops,
            "fp8_tflops": peaks.fp8_tflops,
            "hbm_gbps": peaks.hbm_gbps,
            "ici_gbps": peaks.ici_gbps,
        },
        "defaults": {
            "tp": defaults.get("tp", 8),
            "dp": defaults.get("dp", 1),
            "batch": defaults.get("batch", 64),
            "seq_len": defaults.get("seq_len", 4096),
            "chunk": defaults.get("chunk", 16384),
            "enable_sp": bool(defaults.get("enable_sp", False)),
            "scatter_min": 128,
        },
    }


def build_html_report(arch, config, peaks: HardwarePeaks, defaults: dict, out_path: str) -> str:
    data = _bake(arch, config, peaks, defaults)
    html = _TEMPLATE.replace("__DATA__", json.dumps(data))
    with open(out_path, "w") as f:
        f.write(html)
    return out_path


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>roofline</title>
<style>
 body{font:13px/1.4 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#0f1419;color:#d6deeb}
 #wrap{display:flex;flex-wrap:wrap;gap:16px;padding:16px}
 #left{flex:0 0 320px} #right{flex:1 1 560px;min-width:520px}
 h1{font-size:16px;margin:0 0 4px} .sub{color:#8aa;font-size:11px;margin-bottom:12px}
 .ctl{margin:10px 0} .ctl label{display:block;color:#9cc;font-size:11px;margin-bottom:3px}
 .ctl .v{color:#ffd479;font-weight:600}
 input[type=range]{width:100%} input[type=number]{width:80px;background:#1c2430;color:#d6deeb;border:1px solid #345;border-radius:4px;padding:2px 5px}
 select{background:#1c2430;color:#d6deeb;border:1px solid #345;border-radius:4px;padding:3px}
 .seg{display:inline-flex;border:1px solid #345;border-radius:5px;overflow:hidden}
 .seg button{background:#1c2430;color:#9cc;border:0;padding:4px 12px;cursor:pointer}
 .seg button.on{background:#2d6cdf;color:#fff}
 #banner{padding:8px 10px;border-radius:5px;margin:8px 0;font-size:12px;display:none}
 #banner.err{display:block;background:#3a1416;border:1px solid #a33;color:#fbb}
 #banner.warn{display:block;background:#332a14;border:1px solid #a83;color:#fe9}
 #summary{background:#161d28;border:1px solid #243;border-radius:6px;padding:10px;margin-top:8px}
 #summary b{color:#ffd479} .pill{display:inline-block;background:#243;border-radius:4px;padding:1px 6px;margin:2px 3px 0 0;font-size:11px}
 canvas{background:#0b0f14;border:1px solid #243;border-radius:6px}
 table{border-collapse:collapse;width:100%;margin-top:10px;font-size:11.5px}
 th,td{padding:3px 7px;text-align:right;border-bottom:1px solid #1c2733} th{color:#9cc;text-align:right}
 td.l,th.l{text-align:left} tr.bound-HBM td{} .tag{font-size:10px;padding:0 4px;border-radius:3px}
 .b-HBM{background:#1c3a5a;color:#9cf} .b-ICI{background:#5a1c3a;color:#f9c} .b-compute{background:#1c5a2a;color:#9fc}
 #tip{position:fixed;pointer-events:none;background:#000d;border:1px solid #456;border-radius:4px;padding:4px 7px;font-size:11px;display:none;z-index:9}
 .legend{font-size:10.5px;color:#9cc;margin-top:4px}
</style></head><body>
<div id="wrap">
 <div id="left">
  <h1>Roofline · <span id="arch"></span></h1>
  <div class="sub">per-device, v7x · drag the knobs, the model recomputes live</div>
  <div id="banner"></div>
  <div class="ctl"><label>phase</label>
    <span class="seg"><button id="ph-decode" class="on">decode</button><button id="ph-prefill">prefill</button></span>
    &nbsp;<label style="display:inline">SP</label> <input type="checkbox" id="sp"></div>
  <div class="ctl"><label>tp_size = devices = mesh total <span class="v" id="tpv"></span></label>
    <input type="range" id="tp" min="1" max="64" step="1"></div>
  <div class="ctl"><label>dp_size (data axis) — tensor axis t = tp/dp = <span class="v" id="tv"></span> <span class="v" id="dpv"></span></label>
    <input type="range" id="dp" min="1" max="64" step="1"></div>
  <div class="ctl"><label>decode batch (tokens) <span class="v" id="batchv"></span></label>
    <input type="range" id="batch" min="1" max="2048" step="1"></div>
  <div class="ctl"><label>decode KV context <span class="v" id="seqv"></span></label>
    <input type="range" id="seq_len" min="256" max="262144" step="256"></div>
  <div class="ctl"><label>prefill chunk tokens <span class="v" id="chunkv"></span></label>
    <input type="range" id="chunk" min="256" max="32768" step="256"></div>
  <div id="summary"></div>
 </div>
 <div id="right">
  <canvas id="cv" width="720" height="500"></canvas>
  <div class="legend" id="legend"></div>
  <table id="tbl"><thead><tr><th class="l">op category</th><th>cnt</th><th>GFLOP</th><th>HBM MB</th><th>ICI MB</th><th>OI</th><th>ideal ms</th><th>%step</th><th>bound</th></tr></thead><tbody></tbody></table>
 </div>
</div>
<div id="tip"></div>
<script>
const D = __DATA__;
const CAT = {moe:"#d62728",linear:"#1f77b4",attention:"#2ca02c",router:"#9467bd",lm_head:"#8c564b",norm:"#e377c2",rope:"#7f7f7f",other:"#bcbd22",embedding:"#17becf"};
const P = D.peaks;
const flops_per_s = k => (k==="fp8"?P.fp8_tflops:P.bf16_tflops)*1e12;
const HBMBW = P.hbm_gbps*1e9, ICIBW = P.ici_gbps*1e9;

// ---- cost primitives (mirror ops.py / descriptors.py) ----
function gemm(m,k,n,role){const q=D.quant[role]; return {flops:2*m*k*n, hbm:(q.wbpe+q.wspe)*k*n+(q.abpe+q.aspe)*m*k+2*m*n, ici:0, peak:q.peak};}
function attention(nq,nkv,hd,vhd,qtok,inter){const f=4*nq*hd*inter; const bq=32;
  const hbm=qtok*nq*hd*2 + qtok*nq*vhd*2 + Math.floor(inter/bq)*nkv*2*hd*2 + qtok*nkv*2*hd*2; return {flops:f,hbm:hbm,ici:0,peak:"bf16"};}
function moe(tpd,le,d,f,role){const q=D.quant[role]; const perw=2*((q.wbpe+q.wspe)*d*f)+(q.wbpe+q.wspe)*f*d; const act=2*tpd*d*2;
  return {flops:2*tpd*3*d*f, hbm:le*perw+act, ici:0, peak:q.peak};}
function rope(m,qs,ks){return {flops:6*(qs+ks)*m, hbm:2*(qs+ks)*m*2, ici:0, peak:"bf16"};}
function rms(m,h){return {flops:4*m*h, hbm:2*m*h*2+h*2, ici:0, peak:"bf16"};}
function elt(m,h,ninp){return {flops:m*h, hbm:(ninp+1)*m*h*2, ici:0, peak:"bf16"};}
function router(m,h,ne){return {flops:2*m*h*ne, hbm:m*h*2+h*ne*4+m*ne*4*5, ici:0, peak:"bf16"};} // approx: gate dot + softmax(2-pass)+topk+norm; minor op, never the bound

function allreduce(msg,p){return p<=1?0:2*(p-1)/p*msg;}
function reducescatter(msg,p){return p<=1?0:(p-1)/p*msg;}

function resolve(s){
  const tp=s.tp, dp=s.dp, devices=tp; const errs=[];
  if(tp%dp!==0) errs.push("tp_size ("+tp+") must be divisible by dp_size ("+dp+")");
  const t=Math.max(1,Math.floor(tp/dp));
  if(D.full.nh % t!==0) errs.push("num_attention_heads ("+D.full.nh+") must be divisible by tensor axis t=tp/dp ("+t+")");
  if(D.NEXP % devices!==0) errs.push("n_routed_experts ("+D.NEXP+") must be divisible by EP=devices ("+devices+")");
  return {t, ep:devices, devices, errs};
}
function kvpd(nkv,t){return t>=nkv?1:Math.ceil(nkv/t);}

function rowReduce(tokens,H,L){ // o_proj / moe-output reduce, SP-aware
  const msg=tokens*H*2;
  const sp = L.sp && tokens>=L.devices*D.defaults.scatter_min && tokens%L.devices===0;
  if(sp) return reducescatter(msg,L.devices)+reducescatter(msg,L.devices); // rs + residual ag
  return allreduce(msg,L.t);
}

function compute(s){
  const L=resolve(s); L.sp=s.enable_sp;
  if(L.errs.length) return {errs:L.errs, L};
  const decode = s.phase==="decode";
  const tokens = decode? s.batch : s.chunk;
  const ctx = decode? s.seq_len : s.chunk;
  const logits_tokens = s.batch;
  const t=L.t, ep=L.ep;
  const cat={}; // category -> {flops,hbm,ici,peak,cnt}
  const add=(c,o,cnt,shard,peak)=>{cnt=cnt||1;shard=shard||1; const e=cat[c]||(cat[c]={flops:0,hbm:0,ici:0,cnt:0,peak:peak||o.peak});
    e.flops+=o.flops*cnt/shard; e.hbm+=o.hbm*cnt/shard; e.ici+=o.ici*cnt/shard; e.cnt+=cnt; if(peak)e.peak=peak;};
  function attn(d,count){ if(count<=0)return;
    const qs=d.nh*d.hd, ks=d.nkv*d.hd, vs=d.nkv*d.vhd, ao=d.nh*d.vhd;
    const effctx = d.window? Math.min(ctx,d.window):ctx;
    const inter = tokens*(decode? effctx : effctx/2);
    add("linear", gemm(tokens,D.H,qs+ks+vs,"qkv"), count, t);
    add("rope", rope(tokens,qs,ks), count, t);
    add("attention", attention(Math.max(1,Math.floor(d.nh/t)),kvpd(d.nkv,t),d.hd,d.vhd,tokens,inter), count, 1, "bf16");
    let o=gemm(tokens,ao,D.H,"o_proj"); add("linear", o, count, t, D.quant.o_proj.peak);
    cat.linear.ici += rowReduce(tokens,D.H,L)*count;
    add("norm", rms(tokens,D.H), 2*count); add("other", elt(tokens,D.H,2), 2*count);
  }
  attn(D.full, D.n_full); attn(D.swa, D.n_swa);
  if(D.n_moe>0){ add("router", router(tokens,D.H,D.NEXP), D.n_moe);
    const tpd=Math.max(1,Math.floor(tokens*D.TOPK/ep)); const remote=ep>1?(ep-1)/ep:0;
    const a2a=2*(tokens*D.TOPK/ep)*D.H*2*remote + rowReduce(tokens,D.H,L);
    let e=moe(tpd, D.NEXP/ep, D.H, D.MOEF, "experts"); e.ici=a2a; add("moe", e, D.n_moe, 1, D.quant.experts.peak);
  }
  if(D.n_dense>0){ add("linear", gemm(tokens,D.H,2*D.DENSE_F,"mlp"), D.n_dense, t, D.quant.mlp.peak);
    add("linear", gemm(tokens,D.DENSE_F,D.H,"mlp"), D.n_dense, t, D.quant.mlp.peak);
    add("other", elt(tokens,D.DENSE_F,1), D.n_dense); }
  add("embedding", elt(tokens,D.H,0), 1);
  add("norm", rms(tokens,D.H), 1);
  add("lm_head", gemm(logits_tokens,D.H,D.VOCAB,"lm_head"), 1, t, D.quant.lm_head.peak);

  const rows=[]; let tflops=0,thbm=0,tici=0;
  for(const c in cat){const e=cat[c];
    const cms=e.flops/flops_per_s(e.peak)*1e3, hms=e.hbm/HBMBW*1e3, ims=e.ici/ICIBW*1e3;
    const ideal=Math.max(cms,hms,ims); const bound= ideal===ims&&ims>0?"ICI":(ideal===cms?"compute":"HBM");
    rows.push({cat:c,cnt:e.cnt,flops:e.flops,hbm:e.hbm,ici:e.ici,peak:e.peak,oi:e.hbm>0?e.flops/e.hbm:0,ideal:ideal,bound:bound});
    tflops+=e.flops;thbm+=e.hbm;tici+=e.ici;}
  const tcompute=rows.reduce((a,r)=>a+r.flops,0)/flops_per_s("bf16")*1e3;
  // total bound from summed resource times (perfect overlap lower bound)
  const Tc=rows.reduce((a,r)=>a+r.flops/flops_per_s(r.peak),0)*1e3;
  const Th=thbm/HBMBW*1e3, Ti=tici/ICIBW*1e3;
  const tot=Math.max(Tc,Th,Ti); const tbound= tot===Ti&&Ti>0?"ICI":(tot===Tc?"compute":"HBM");
  rows.sort((a,b)=>b.ideal-a.ideal);
  for(const r of rows) r.pct=r.ideal/(rows.reduce((a,x)=>a+x.ideal,0))*100;
  return {rows, L, tot, tbound, Tc, Th, Ti, tflops, thbm, tici, decode, tokens};
}

// ---------- rendering ----------
const cv=document.getElementById("cv"), cx=cv.getContext("2d"); let LAST=null;
function draw(R){LAST=R; const W=cv.width,Hh=cv.height, ml=58,mr=14,mt=18,mb=42;
  cx.clearRect(0,0,W,Hh);
  const rows=R.rows.filter(r=>r.flops>0&&r.hbm>0);
  const ceil=(rows.some(r=>r.peak==="fp8")?P.fp8_tflops:P.bf16_tflops);
  const oiv=rows.map(r=>r.oi), perfs=rows.map(r=>r.flops/(r.ideal/1e3)/1e12);
  let xmin=Math.min(...oiv)/3, xmax=Math.max(...oiv)*3;
  let ymax=ceil*1.4, ymin=Math.min(...perfs.filter(p=>p>0),ceil)/50; if(!isFinite(ymin)||ymin<=0)ymin=ceil/1000;
  const lx=v=>ml+(Math.log10(v)-Math.log10(xmin))/(Math.log10(xmax)-Math.log10(xmin))*(W-ml-mr);
  const ly=v=>mt+(Math.log10(ymax)-Math.log10(v))/(Math.log10(ymax)-Math.log10(ymin))*(Hh-mt-mb);
  // grid + ticks
  cx.strokeStyle="#1c2733";cx.fillStyle="#789";cx.font="10px sans-serif";cx.lineWidth=1;
  for(let e=-3;e<=6;e++){const x=Math.pow(10,e); if(x<xmin||x>xmax)continue; cx.beginPath();cx.moveTo(lx(x),mt);cx.lineTo(lx(x),Hh-mb);cx.stroke(); cx.fillText("1e"+e,lx(x)-8,Hh-mb+12);}
  for(let e=-2;e<=4;e++){const y=Math.pow(10,e); if(y<ymin||y>ymax)continue; cx.beginPath();cx.moveTo(ml,ly(y));cx.lineTo(W-mr,ly(y));cx.stroke(); cx.fillText((y>=1?y:y).toString(),4,ly(y)+3);}
  cx.fillStyle="#9cc";cx.fillText("operational intensity (FLOP/HBM-byte)",W/2-90,Hh-6);
  cx.save();cx.translate(12,Hh/2+70);cx.rotate(-Math.PI/2);cx.fillText("attainable TFLOP/s",0,0);cx.restore();
  // roof: HBM diagonal capped by ceiling
  cx.strokeStyle="#cdd";cx.lineWidth=2;cx.beginPath();let first=true;
  for(let i=0;i<=120;i++){const x=xmin*Math.pow(xmax/xmin,i/120); const y=Math.min(x*HBMBW/1e12,ceil); const px=lx(x),py=ly(y); if(first){cx.moveTo(px,py);first=false;}else cx.lineTo(px,py);} cx.stroke();
  // ceilings
  cx.setLineDash([5,4]);cx.strokeStyle="#667";cx.beginPath();cx.moveTo(ml,ly(P.bf16_tflops));cx.lineTo(W-mr,ly(P.bf16_tflops));cx.stroke();
  cx.fillStyle="#889";cx.fillText("bf16 "+P.bf16_tflops.toFixed(0),W-mr-70,ly(P.bf16_tflops)-3);
  if(rows.some(r=>r.peak==="fp8")){cx.beginPath();cx.moveTo(ml,ly(P.fp8_tflops));cx.lineTo(W-mr,ly(P.fp8_tflops));cx.stroke();cx.fillText("fp8 "+P.fp8_tflops.toFixed(0),W-mr-70,ly(P.fp8_tflops)-3);}
  cx.setLineDash([]);
  const ridge=ceil/(HBMBW/1e12); if(ridge>xmin&&ridge<xmax){cx.strokeStyle="#334";cx.beginPath();cx.moveTo(lx(ridge),mt);cx.lineTo(lx(ridge),Hh-mb);cx.stroke();cx.fillStyle="#778";cx.fillText("ridge "+ridge.toFixed(0),lx(ridge)+3,mt+10);}
  // points
  const smax=Math.max(...rows.map(r=>r.ideal))||1; R._pts=[];
  for(const r of rows){const x=r.oi, y=r.flops/(r.ideal/1e3)/1e12; const px=lx(x),py=ly(y); const rad=4+11*(r.ideal/smax);
    cx.fillStyle=CAT[r.cat]||"#888"; cx.strokeStyle="#000";cx.lineWidth=1;
    if(r.bound==="ICI"){cx.save();cx.translate(px,py);cx.rotate(Math.PI/4);cx.fillRect(-rad,-2,2*rad,4);cx.fillRect(-2,-rad,4,2*rad);cx.restore();}
    else{cx.beginPath();cx.arc(px,py,rad,0,7);cx.fill();cx.stroke();}
    R._pts.push({px,py,rad,r});
    if(r.ideal>0.03*smax){cx.fillStyle="#cde";cx.font="10px sans-serif";cx.fillText(r.cat,px+rad+2,py+3);}
  }
}
function fmt(x,d){d=d===undefined?1:d; return x>=1000?(x/1000).toFixed(d)+"k":x.toFixed(d);}
function render(){const s=state(); const R=compute(s);
  const ban=document.getElementById("banner");
  if(R.errs){ban.className="err";ban.innerHTML="✗ invalid layout:<br>"+R.errs.join("<br>");document.querySelector("#tbl tbody").innerHTML="";cx.clearRect(0,0,cv.width,cv.height);renderSummaryErr(R);return;}
  ban.className="";ban.style.display="none";
  draw(R);
  // table
  let tb=""; const totIdeal=R.rows.reduce((a,r)=>a+r.ideal,0);
  for(const r of R.rows){ tb+="<tr><td class='l'><span style='color:"+(CAT[r.cat]||'#888')+"'>●</span> "+r.cat+(r.peak==="fp8"?" <span class='tag' style='background:#3a2a0a;color:#fd8'>fp8</span>":"")+"</td><td>"+r.cnt+"</td><td>"+fmt(r.flops/1e9)+"</td><td>"+fmt(r.hbm/1e6)+"</td><td>"+fmt(r.ici/1e6)+"</td><td>"+r.oi.toFixed(1)+"</td><td>"+r.ideal.toFixed(3)+"</td><td>"+r.pct.toFixed(0)+"%</td><td><span class='tag b-"+r.bound+"'>"+r.bound+"</span></td></tr>";}
  document.querySelector("#tbl tbody").innerHTML=tb;
  // summary
  renderSummary(R);
}
function renderSummary(R){const L=R.L; const s=state();
  document.getElementById("summary").innerHTML =
   "<b>mesh</b> data="+L.dp+" × tensor="+L.t+" = "+L.devices+" devices &nbsp; <b>EP</b>="+L.ep+(s.enable_sp?" &nbsp;<span class='pill'>+SP</span>":"")
   +"<br><b>"+(R.decode?"decode":"prefill")+"</b> · tokens="+R.tokens
   +"<br><b>bound: <span class='tag b-"+R.tbound+"'>"+R.tbound+"</span></b> &nbsp; step≈"+R.tot.toFixed(2)+" ms"
   +"<div style='margin-top:6px'><span class='pill'>compute "+R.Tc.toFixed(2)+"ms</span><span class='pill'>HBM "+R.Th.toFixed(2)+"ms</span><span class='pill'>ICI "+R.Ti.toFixed(2)+"ms</span></div>"
   +"<div style='margin-top:6px;color:#8aa;font-size:11px'>util@bound vs v7x: compute "+(R.Tc/R.tot*100).toFixed(0)+"% · HBM "+(R.Th/R.tot*100).toFixed(0)+"% · ICI "+(R.Ti/R.tot*100).toFixed(0)+"%</div>";
}
function renderSummaryErr(R){document.getElementById("summary").innerHTML="<span style='color:#f99'>fix the layout to see results</span>";}

// dp options = divisors of tp
function divisors(n){const a=[];for(let i=1;i<=n;i++)if(n%i===0)a.push(i);return a;}
function state(){return {tp:+g("tp").value, dp:+g("dp").value, batch:+g("batch").value, seq_len:+g("seq_len").value, chunk:+g("chunk").value, phase:PHASE, enable_sp:g("sp").checked};}
function g(id){return document.getElementById(id);}
let PHASE="decode";
function syncLabels(){g("tpv").textContent=g("tp").value; g("dpv").textContent="dp="+g("dp").value;
  g("tv").textContent="t="+Math.max(1,Math.floor(g("tp").value/g("dp").value));
  g("batchv").textContent=g("batch").value; g("seqv").textContent=g("seq_len").value; g("chunkv").textContent=g("chunk").value;}
function init(){
  g("arch").textContent=D.arch;
  const d=D.defaults; g("tp").value=d.tp; g("dp").value=d.dp; g("batch").value=d.batch; g("seq_len").value=d.seq_len; g("chunk").value=d.chunk; g("sp").checked=d.enable_sp;
  ["tp","dp","batch","seq_len","chunk"].forEach(id=>g(id).addEventListener("input",()=>{syncLabels();render();}));
  g("sp").addEventListener("change",render);
  g("ph-decode").onclick=()=>{PHASE="decode";g("ph-decode").className="on";g("ph-prefill").className="";render();};
  g("ph-prefill").onclick=()=>{PHASE="prefill";g("ph-prefill").className="on";g("ph-decode").className="";render();};
  // legend
  g("legend").innerHTML=Object.keys(CAT).map(c=>"<span style='color:"+CAT[c]+"'>●</span> "+c).join(" &nbsp; ")+" &nbsp; ✕=ICI-bound (below roof)";
  // hover tooltip
  cv.addEventListener("mousemove",e=>{const rect=cv.getBoundingClientRect();const mx=e.clientX-rect.left,my=e.clientY-rect.top;
    let hit=null; if(LAST&&LAST._pts)for(const p of LAST._pts){if((mx-p.px)**2+(my-p.py)**2<(p.rad+4)**2){hit=p;break;}}
    const tip=g("tip"); if(hit){const r=hit.r; tip.style.display="block";tip.style.left=(e.clientX+12)+"px";tip.style.top=(e.clientY+8)+"px";
      tip.innerHTML="<b>"+r.cat+"</b> (x"+r.cnt+")<br>"+fmt(r.flops/1e9)+" GFLOP · "+fmt(r.hbm/1e6)+" MB HBM · "+fmt(r.ici/1e6)+" MB ICI<br>OI="+r.oi.toFixed(1)+" · ideal "+r.ideal.toFixed(3)+"ms · <b>"+r.bound+"</b>";}
    else tip.style.display="none";});
  cv.addEventListener("mouseleave",()=>g("tip").style.display="none");
  syncLabels(); render();
}
init();
</script></body></html>"""
