"""Interactive HTML roofline report.

Bakes a model's config-derived constants + quant coefficients + hardware peaks
into a single self-contained HTML page that re-runs the (closed-form) cost model
**in the browser**: pick the parallelism layout (tp/dp constrained to valid mesh
combos so an illegal layout can't be selected) and drag the workload knobs
(batch, seq_len, chunk, phase, sequence-parallel); the roofline, the per-category
cost table and the bottleneck summary update live. No server, no external JS
libs (vanilla + high-DPI canvas), works offline.

The JS cost model mirrors ``descriptors._mimo_v2_family`` + ``parallelism`` +
``ops`` (same op list/formulas/mesh semantics: tensor axis t = tp//dp, fused-MoE
EP = devices, MoE global tokens = per-DP tokens * dp, SP reduce-scatter/all-gather
gated on should_scatter). Closed-form roofline only (View B/C/D); jaxpr View F
needs a real trace and stays Python-side.
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
 body{font:13px/1.45 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f6f8fa;color:#1c2330}
 #wrap{display:flex;flex-wrap:wrap;gap:20px;padding:20px}
 #left{flex:0 0 300px} #right{flex:1 1 760px;min-width:680px}
 h1{font-size:17px;margin:0 0 2px} .sub{color:#667;font-size:11px;margin-bottom:14px}
 .ctl{margin:12px 0} .ctl label{display:block;color:#445;font-size:11px;margin-bottom:4px;font-weight:600}
 .ctl .v{color:#0a5;font-weight:700}
 input[type=range]{width:100%} select{background:#fff;color:#1c2330;border:1px solid #bcc;border-radius:5px;padding:4px 8px;font-size:13px}
 .seg{display:inline-flex;border:1px solid #bcc;border-radius:6px;overflow:hidden}
 .seg button{background:#fff;color:#556;border:0;padding:5px 14px;cursor:pointer;font-size:13px}
 .seg button.on{background:#2563eb;color:#fff}
 #summary{background:#fff;border:1px solid #dde;border-radius:8px;padding:12px;margin-top:10px;box-shadow:0 1px 3px #0001}
 #summary b{color:#0a5} .pill{display:inline-block;background:#eef;border-radius:5px;padding:2px 7px;margin:3px 4px 0 0;font-size:11px;color:#335}
 canvas{background:#fff;border:1px solid #dde;border-radius:8px;box-shadow:0 1px 4px #0001}
 table{border-collapse:collapse;width:100%;margin-top:12px;font-size:12px;background:#fff;border-radius:8px;overflow:hidden;box-shadow:0 1px 3px #0001}
 th,td{padding:4px 9px;text-align:right;border-bottom:1px solid #eef} th{color:#556;background:#f1f4f8} td.l,th.l{text-align:left}
 .tag{font-size:10px;padding:1px 5px;border-radius:4px}
 .b-HBM{background:#dbeafe;color:#1e40af} .b-ICI{background:#fce7f3;color:#9d174d} .b-compute{background:#dcfce7;color:#166534}
 #tip{position:fixed;pointer-events:none;background:#1c2330;color:#fff;border-radius:5px;padding:5px 8px;font-size:11px;display:none;z-index:9;box-shadow:0 2px 8px #0004}
 .legend{font-size:11px;color:#556;margin-top:6px}
</style></head><body>
<div id="wrap">
 <div id="left">
  <h1>Roofline · <span id="arch"></span></h1>
  <div class="sub">per-device · v7x · adjust the layout/workload, recomputed live in-browser</div>
  <div class="ctl"><label>phase</label>
    <span class="seg"><button id="ph-decode" class="on">decode</button><button id="ph-prefill">prefill</button></span>
    &nbsp; <label style="display:inline">SP</label> <input type="checkbox" id="sp"></div>
  <div class="ctl"><label>tp_size = devices = mesh total</label><select id="tp"></select></div>
  <div class="ctl"><label>dp_size — tensor axis t = tp/dp = <span class="v" id="tv"></span></label><select id="dp"></select></div>
  <div class="ctl"><label>decode batch (tokens) <span class="v" id="batchv"></span></label>
    <input type="range" id="batch" min="1" max="2048" step="1"></div>
  <div class="ctl"><label>decode KV context <span class="v" id="seqv"></span></label>
    <input type="range" id="seq_len" min="256" max="262144" step="256"></div>
  <div class="ctl"><label>prefill chunk tokens <span class="v" id="chunkv"></span></label>
    <input type="range" id="chunk" min="256" max="32768" step="256"></div>
  <div id="summary"></div>
 </div>
 <div id="right">
  <canvas id="cv"></canvas>
  <div class="legend" id="legend"></div>
  <table id="tbl"><thead><tr><th class="l">op category</th><th>cnt</th><th>GFLOP</th><th>HBM MB</th><th>ICI MB</th><th>OI</th><th>ideal ms</th><th>%step</th><th>bound</th></tr></thead><tbody></tbody></table>
 </div>
</div>
<div id="tip"></div>
<script>
const D = __DATA__;
const CAT = {moe:"#dc2626",linear:"#2563eb",attention:"#16a34a",router:"#9333ea",lm_head:"#b45309",norm:"#db2777",rope:"#6b7280",other:"#a16207",embedding:"#0891b2"};
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
function router(m,h,ne){return {flops:2*m*h*ne, hbm:m*h*2+h*ne*4+m*ne*4*5, ici:0, peak:"bf16"};} // approx; never the bound

function allreduce(msg,p){return p<=1?0:2*(p-1)/p*msg;}
function reducescatter(msg,p){return p<=1?0:(p-1)/p*msg;}
function resolve(s){const tp=s.tp, dp=s.dp, devices=tp; const t=Math.max(1,Math.floor(tp/dp)); return {t, ep:devices, devices, dp};}
function kvpd(nkv,t){return t>=nkv?1:Math.ceil(nkv/t);}
function rowReduce(tokens,H,L){const msg=tokens*H*2;
  const sp = L.sp && tokens>=L.devices*D.defaults.scatter_min && tokens%L.devices===0;
  if(sp) return reducescatter(msg,L.devices)+reducescatter(msg,L.devices);
  return allreduce(msg,L.t);}

function compute(s){
  const L=resolve(s); L.sp=s.enable_sp;
  const decode = s.phase==="decode";
  const tokens = decode? s.batch : s.chunk;
  const ctx = decode? s.seq_len : s.chunk;
  const t=L.t, ep=L.ep;
  const cat={};
  const add=(c,o,cnt,shard,peak)=>{cnt=cnt||1;shard=shard||1; const e=cat[c]||(cat[c]={flops:0,hbm:0,ici:0,cnt:0,peak:peak||o.peak});
    e.flops+=o.flops*cnt/shard; e.hbm+=o.hbm*cnt/shard; e.ici+=o.ici*cnt/shard; e.cnt+=cnt; if(peak)e.peak=peak;};
  function attn(d,count){ if(count<=0)return;
    const qs=d.nh*d.hd, ks=d.nkv*d.hd, vs=d.nkv*d.vhd, ao=d.nh*d.vhd;
    const effctx = d.window? Math.min(ctx,d.window):ctx;
    const inter = tokens*(decode? effctx : effctx/2);
    add("linear", gemm(tokens,D.H,qs+ks+vs,"qkv"), count, t);
    add("rope", rope(tokens,qs,ks), count, t);
    add("attention", attention(Math.max(1,Math.floor(d.nh/t)),kvpd(d.nkv,t),d.hd,d.vhd,tokens,inter), count, 1, "bf16");
    add("linear", gemm(tokens,ao,D.H,"o_proj"), count, t, D.quant.o_proj.peak);
    cat.linear.ici += rowReduce(tokens,D.H,L)*count;
    add("norm", rms(tokens,D.H), 2*count); add("other", elt(tokens,D.H,2), 2*count);
  }
  attn(D.full, D.n_full); attn(D.swa, D.n_swa);
  if(D.n_moe>0){ add("router", router(tokens,D.H,D.NEXP), D.n_moe);
    const moe_tokens=tokens*L.dp;  // global tokens into MoE = per-DP tokens * dp groups
    const tpd=Math.max(1,Math.floor(moe_tokens*D.TOPK/ep)); const remote=ep>1?(ep-1)/ep:0;
    const a2a=2*(moe_tokens*D.TOPK/ep)*D.H*2*remote + rowReduce(tokens,D.H,L);
    let e=moe(tpd, D.NEXP/ep, D.H, D.MOEF, "experts"); e.ici=a2a; add("moe", e, D.n_moe, 1, D.quant.experts.peak);
  }
  if(D.n_dense>0){ add("linear", gemm(tokens,D.H,2*D.DENSE_F,"mlp"), D.n_dense, t, D.quant.mlp.peak);
    add("linear", gemm(tokens,D.DENSE_F,D.H,"mlp"), D.n_dense, t, D.quant.mlp.peak);
    add("other", elt(tokens,D.DENSE_F,1), D.n_dense); }
  add("embedding", elt(tokens,D.H,0), 1);
  add("norm", rms(tokens,D.H), 1);
  add("lm_head", gemm(s.batch,D.H,D.VOCAB,"lm_head"), 1, t, D.quant.lm_head.peak);

  const rows=[]; let thbm=0,tici=0;
  for(const c in cat){const e=cat[c];
    const cms=e.flops/flops_per_s(e.peak)*1e3, hms=e.hbm/HBMBW*1e3, ims=e.ici/ICIBW*1e3;
    const ideal=Math.max(cms,hms,ims); const bound= ideal===ims&&ims>0?"ICI":(ideal===cms?"compute":"HBM");
    rows.push({cat:c,cnt:e.cnt,flops:e.flops,hbm:e.hbm,ici:e.ici,peak:e.peak,oi:e.hbm>0?e.flops/e.hbm:0,ideal:ideal,bound:bound});
    thbm+=e.hbm;tici+=e.ici;}
  const Tc=rows.reduce((a,r)=>a+r.flops/flops_per_s(r.peak),0)*1e3, Th=thbm/HBMBW*1e3, Ti=tici/ICIBW*1e3;
  const tot=Math.max(Tc,Th,Ti); const tbound= tot===Ti&&Ti>0?"ICI":(tot===Tc?"compute":"HBM");
  rows.sort((a,b)=>b.ideal-a.ideal); const sumIdeal=rows.reduce((a,x)=>a+x.ideal,0);
  for(const r of rows) r.pct=sumIdeal>0?r.ideal/sumIdeal*100:0;
  return {rows, L, tot, tbound, Tc, Th, Ti, decode, tokens};
}

// ---------- rendering (high-DPI canvas, light theme) ----------
const cv=document.getElementById("cv"), cx=cv.getContext("2d"); let LAST=null; const CW=1040, CH=650;
function setupCanvas(){const dpr=window.devicePixelRatio||1; cv.style.width=CW+"px"; cv.style.height=CH+"px";
  cv.width=Math.round(CW*dpr); cv.height=Math.round(CH*dpr); cx.setTransform(dpr,0,0,dpr,0,0);}
function draw(R){LAST=R; const W=CW,Hh=CH, ml=66,mr=18,mt=22,mb=48;
  cx.clearRect(0,0,W,Hh);
  const rows=R.rows.filter(r=>r.flops>0&&r.hbm>0);
  const ceil=(rows.some(r=>r.peak==="fp8")?P.fp8_tflops:P.bf16_tflops);
  const oiv=rows.map(r=>r.oi), perfs=rows.map(r=>r.flops/(r.ideal/1e3)/1e12);
  let xmin=Math.min(...oiv)/3, xmax=Math.max(...oiv)*3; if(!(xmin>0))xmin=0.01;
  let ymax=ceil*1.5, ymin=Math.min(...perfs.filter(p=>p>0),ceil)/80; if(!isFinite(ymin)||ymin<=0)ymin=ceil/1000;
  const lx=v=>ml+(Math.log10(v)-Math.log10(xmin))/(Math.log10(xmax)-Math.log10(xmin))*(W-ml-mr);
  const ly=v=>mt+(Math.log10(ymax)-Math.log10(v))/(Math.log10(ymax)-Math.log10(ymin))*(Hh-mt-mb);
  cx.strokeStyle="#eef1f5";cx.fillStyle="#889";cx.font="11px sans-serif";cx.lineWidth=1;
  for(let e=-3;e<=7;e++){const x=Math.pow(10,e); if(x<xmin||x>xmax)continue; cx.beginPath();cx.moveTo(lx(x),mt);cx.lineTo(lx(x),Hh-mb);cx.stroke(); cx.fillText("1e"+e,lx(x)-8,Hh-mb+14);}
  for(let e=-3;e<=4;e++){const y=Math.pow(10,e); if(y<ymin||y>ymax)continue; cx.beginPath();cx.moveTo(ml,ly(y));cx.lineTo(W-mr,ly(y));cx.stroke(); cx.fillText("1e"+e,6,ly(y)+3);}
  cx.fillStyle="#445";cx.font="12px sans-serif";cx.fillText("operational intensity (FLOP / HBM-byte)",W/2-110,Hh-8);
  cx.save();cx.translate(16,Hh/2+80);cx.rotate(-Math.PI/2);cx.fillText("attainable TFLOP/s",0,0);cx.restore();
  // roof: HBM diagonal capped by compute ceiling
  cx.strokeStyle="#334155";cx.lineWidth=2.5;cx.beginPath();let first=true;
  for(let i=0;i<=160;i++){const x=xmin*Math.pow(xmax/xmin,i/160); const y=Math.min(x*HBMBW/1e12,ceil); const px=lx(x),py=ly(y); if(first){cx.moveTo(px,py);first=false;}else cx.lineTo(px,py);} cx.stroke();
  cx.setLineDash([6,4]);cx.strokeStyle="#94a3b8";cx.lineWidth=1.2;cx.beginPath();cx.moveTo(ml,ly(P.bf16_tflops));cx.lineTo(W-mr,ly(P.bf16_tflops));cx.stroke();
  cx.fillStyle="#64748b";cx.fillText("bf16 "+P.bf16_tflops.toFixed(0)+" TF/s",W-mr-118,ly(P.bf16_tflops)-4);
  if(rows.some(r=>r.peak==="fp8")){cx.beginPath();cx.moveTo(ml,ly(P.fp8_tflops));cx.lineTo(W-mr,ly(P.fp8_tflops));cx.stroke();cx.fillText("fp8 "+P.fp8_tflops.toFixed(0)+" TF/s",W-mr-118,ly(P.fp8_tflops)-4);}
  cx.setLineDash([]);
  const ridge=ceil/(HBMBW/1e12); if(ridge>xmin&&ridge<xmax){cx.strokeStyle="#cbd5e1";cx.lineWidth=1;cx.beginPath();cx.moveTo(lx(ridge),mt);cx.lineTo(lx(ridge),Hh-mb);cx.stroke();cx.fillStyle="#94a3b8";cx.fillText("ridge OI="+ridge.toFixed(0),lx(ridge)+4,mt+12);}
  const smax=Math.max(...rows.map(r=>r.ideal))||1; R._pts=[];
  for(const r of rows){const x=r.oi, y=r.flops/(r.ideal/1e3)/1e12; const px=lx(x),py=ly(y); const rad=6+15*(r.ideal/smax);
    cx.fillStyle=CAT[r.cat]||"#888";
    if(r.bound==="ICI"){cx.save();cx.translate(px,py);cx.rotate(Math.PI/4);cx.lineWidth=3.5;cx.strokeStyle=CAT[r.cat]||"#888";cx.beginPath();cx.moveTo(-rad,0);cx.lineTo(rad,0);cx.moveTo(0,-rad);cx.lineTo(0,rad);cx.stroke();cx.restore();}
    else{cx.strokeStyle="#fff";cx.lineWidth=1.5;cx.beginPath();cx.arc(px,py,rad,0,7);cx.fill();cx.stroke();}
    R._pts.push({px,py,rad,r});
    if(r.ideal>0.03*smax){cx.fillStyle="#1c2330";cx.font="11px sans-serif";cx.fillText(r.cat,px+rad+3,py+4);}
  }
}
function fmt(x,d){d=d===undefined?1:d; return Math.abs(x)>=1000?(x/1000).toFixed(d)+"k":x.toFixed(d);}
function render(){const s=state(); const R=compute(s); draw(R);
  let tb=""; for(const r of R.rows){ tb+="<tr><td class='l'><span style='color:"+(CAT[r.cat]||'#888')+"'>●</span> "+r.cat+(r.peak==="fp8"?" <span class='tag' style='background:#fef3c7;color:#92400e'>fp8</span>":"")+"</td><td>"+r.cnt+"</td><td>"+fmt(r.flops/1e9)+"</td><td>"+fmt(r.hbm/1e6)+"</td><td>"+fmt(r.ici/1e6)+"</td><td>"+r.oi.toFixed(1)+"</td><td>"+r.ideal.toFixed(3)+"</td><td>"+r.pct.toFixed(0)+"%</td><td><span class='tag b-"+r.bound+"'>"+r.bound+"</span></td></tr>";}
  document.querySelector("#tbl tbody").innerHTML=tb;
  const L=R.L; const s2=state();
  document.getElementById("summary").innerHTML =
   "<b>mesh</b> data="+L.dp+" × tensor="+L.t+" = "+L.devices+" devices &nbsp; <b>EP</b>="+L.ep+(s2.enable_sp?" &nbsp;<span class='pill'>+SP</span>":"")
   +"<br><b>"+(R.decode?"decode":"prefill")+"</b> · tokens/DP="+R.tokens+" · MoE global="+(R.tokens*L.dp)
   +"<br><b>bound: <span class='tag b-"+R.tbound+"'>"+R.tbound+"</span></b> &nbsp; step ≈ "+R.tot.toFixed(2)+" ms"
   +"<div style='margin-top:6px'><span class='pill'>compute "+R.Tc.toFixed(2)+"ms</span><span class='pill'>HBM "+R.Th.toFixed(2)+"ms</span><span class='pill'>ICI "+R.Ti.toFixed(2)+"ms</span></div>";
}

function divisors(n){const a=[];for(let i=1;i<=n;i++)if(n%i===0)a.push(i);return a;}
function validDp(tp){return divisors(tp).filter(d=>D.full.nh%(Math.floor(tp/d))===0);}
function g(id){return document.getElementById(id);}
let PHASE="decode";
function state(){return {tp:+g("tp").value, dp:+g("dp").value, batch:+g("batch").value, seq_len:+g("seq_len").value, chunk:+g("chunk").value, phase:PHASE, enable_sp:g("sp").checked};}
function fillDp(){const tp=+g("tp").value; const cur=+g("dp").value; const opts=validDp(tp);
  g("dp").innerHTML=opts.map(d=>"<option value='"+d+"'>"+d+"</option>").join("");
  g("dp").value = opts.includes(cur)? cur : opts[opts.length-1];}
function syncLabels(){g("tv").textContent="t="+Math.max(1,Math.floor(g("tp").value/g("dp").value));
  g("batchv").textContent=g("batch").value; g("seqv").textContent=g("seq_len").value; g("chunkv").textContent=g("chunk").value;}
function init(){
  g("arch").textContent=D.arch; setupCanvas();
  const d=D.defaults;
  // tp options = mesh totals the fused MoE allows (EP=tp must divide n_experts) with a valid dp
  const tpopts=divisors(D.NEXP).filter(x=>x<=1024 && validDp(x).length>0);
  g("tp").innerHTML=tpopts.map(x=>"<option value='"+x+"'>"+x+"</option>").join("");
  g("tp").value = tpopts.includes(d.tp)? d.tp : tpopts[tpopts.length-1];
  fillDp(); if(validDp(+g("tp").value).includes(d.dp)) g("dp").value=d.dp;
  g("batch").value=d.batch; g("seq_len").value=d.seq_len; g("chunk").value=d.chunk; g("sp").checked=d.enable_sp;
  g("tp").addEventListener("change",()=>{fillDp();syncLabels();render();});
  g("dp").addEventListener("change",()=>{syncLabels();render();});
  ["batch","seq_len","chunk"].forEach(id=>g(id).addEventListener("input",()=>{syncLabels();render();}));
  g("sp").addEventListener("change",render);
  g("ph-decode").onclick=()=>{PHASE="decode";g("ph-decode").className="on";g("ph-prefill").className="";render();};
  g("ph-prefill").onclick=()=>{PHASE="prefill";g("ph-prefill").className="on";g("ph-decode").className="";render();};
  g("legend").innerHTML=Object.keys(CAT).map(c=>"<span style='color:"+CAT[c]+"'>●</span> "+c).join(" &nbsp; ")+" &nbsp; ✕ = ICI-bound (below roof)";
  cv.addEventListener("mousemove",e=>{const rect=cv.getBoundingClientRect();const mx=e.clientX-rect.left,my=e.clientY-rect.top;
    let hit=null; if(LAST&&LAST._pts)for(const p of LAST._pts){if((mx-p.px)**2+(my-p.py)**2<(p.rad+5)**2){hit=p;break;}}
    const tip=g("tip"); if(hit){const r=hit.r; tip.style.display="block";tip.style.left=(e.clientX+12)+"px";tip.style.top=(e.clientY+8)+"px";
      tip.innerHTML="<b>"+r.cat+"</b> (×"+r.cnt+")<br>"+fmt(r.flops/1e9)+" GFLOP · "+fmt(r.hbm/1e6)+" MB HBM · "+fmt(r.ici/1e6)+" MB ICI<br>OI="+r.oi.toFixed(1)+" · ideal "+r.ideal.toFixed(3)+"ms · <b>"+r.bound+"</b>";}
    else tip.style.display="none";});
  cv.addEventListener("mouseleave",()=>g("tip").style.display="none");
  syncLabels(); render();
}
init();
</script></body></html>"""
