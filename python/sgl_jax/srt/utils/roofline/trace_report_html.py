"""Self-contained HTML report driven by the REAL traced forward.

Unlike ``report_html`` (which bakes the hand-written analytic model + live knobs),
this report is grounded in the actual ``standalone_trace`` jaxpr of a specific
model: the roofline + cost table come from ``trace_analyze`` (validated to match
the analytic descriptor), and -- the trace's unique value -- it also shows the
REAL per-op code-path index (every category -> its ``models/*.py`` call chain),
the REAL Pallas kernels (RPAd / RPAm / fused-moe-v2 with per-device avals), and
the primitive histogram. Prefill + decode are both baked; a toggle switches them.
Pure static (the trace is the operating point); no JS cost model.
"""

from __future__ import annotations

import json

from . import trace_analyze
from .report import HardwarePeaks, ModelRoofline


def _phase_data(model: ModelRoofline, peaks: HardwarePeaks) -> dict:
    rows = []
    tot = tc = th = ti = 0.0
    for r in model.by_category():
        cm, hm, im, idl = (
            r.compute_ms(peaks),
            r.hbm_ms(peaks),
            r.ici_ms(peaks),
            r.ideal_ms(peaks),
        )
        tot += idl
        tc += cm
        th += hm
        ti += im
        rows.append(
            dict(
                cat=r.category,
                count=r.count,
                flops=r.flops,
                hbm=r.hbm_bytes,
                ici=r.ici_bytes,
                oi=(r.flops / r.hbm_bytes if r.hbm_bytes else 0),
                compute_ms=cm,
                hbm_ms=hm,
                ici_ms=im,
                ideal=idl,
                bound=r.bound(peaks),
                peak=r.peak_kind,
                source=r.source,
            )
        )
    for row in rows:
        row["pct"] = (row["ideal"] / tot * 100) if tot else 0
    t = model.total()
    return dict(
        rows=rows,
        total_ms=tot,
        Tc=tc,
        Th=th,
        Ti=ti,
        tbound=t.bound(peaks),
        meta=model.meta,
    )


def build_trace_report(
    arch: str,
    config: dict,
    layout,
    results: dict,  # {"prefill": ModelRoofline, "decode": ModelRoofline}
    records: dict,
    peaks: HardwarePeaks,
    out_path: str,
) -> str:
    phases = {ph: _phase_data(m, peaks) for ph, m in results.items() if m is not None}
    data = {
        "arch": arch,
        "parallelism": {
            "tp": layout.tp_total,
            "dp": layout.dp,
            "t": layout.t,
            "ep": layout.ep,
            "devices": layout.devices,
            "enable_sp": layout.enable_sp,
        },
        "peaks": {
            "bf16": peaks.bf16_tflops,
            "fp8": peaks.fp8_tflops,
            "hbm": peaks.hbm_gbps,
            "ici": peaks.ici_gbps,
        },
        "phases": phases,
        "codepath": trace_analyze.code_path_index(records, config),
    }
    html = _TEMPLATE.replace("__DATA__", json.dumps(data))
    with open(out_path, "w") as f:
        f.write(html)
    return out_path


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Roofline — traced forward</title>
<style>
:root{--bg:#fff;--fg:#1c2330;--mut:#667;--line:#e6e9ef;--accent:#0d9488}
*{box-sizing:border-box}
body{margin:0;font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:var(--fg);background:#f6f7f9}
.wrap{max-width:1180px;margin:0 auto;padding:18px}
h1{font-size:19px;margin:0 0 2px} .sub{color:var(--mut);font-size:12px;margin-bottom:14px}
.card{background:#fff;border:1px solid var(--line);border-radius:10px;padding:16px;margin-bottom:14px}
.pill{display:inline-block;background:#f1f5f9;border-radius:6px;padding:1px 8px;margin:2px 4px 2px 0;font-size:12px}
.pill.sp{background:#ecfdf5;color:#047857} .pill.fp8{background:#fef3c7;color:#92400e}
#summary{font-size:13px;line-height:1.7}
.tabs{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px}
.tab{padding:6px 13px;border:1px solid var(--line);border-radius:8px;background:#fff;cursor:pointer;font-size:13px;color:#445}
.tab.on{background:var(--accent);color:#fff;border-color:var(--accent)}
.toggle{float:right} .toggle button{padding:5px 12px;border:1px solid var(--line);background:#fff;cursor:pointer;font-size:12px}
.toggle button:first-child{border-radius:7px 0 0 7px} .toggle button:last-child{border-radius:0 7px 7px 0}
.toggle button.on{background:#1c2330;color:#fff;border-color:#1c2330}
canvas{display:block;width:100%}
table{border-collapse:collapse;width:100%;font-size:12.5px;margin-top:6px}
th,td{padding:5px 8px;border-bottom:1px solid var(--line);text-align:right}
th.l,td.l{text-align:left} th{color:var(--mut);font-weight:600}
.tag{display:inline-block;border-radius:5px;padding:0 7px;font-size:11px}
.b-HBM{background:#dbeafe;color:#1e40af} .b-compute{background:#dcfce7;color:#166534}
.b-ICI{background:#fce7f3;color:#9d174d} .b-none{background:#f1f5f9;color:#667}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11.5px}
.chain{color:#475569} .chain b{color:#0f172a} .arrow{color:#94a3b8}
.note{font-size:11px;color:var(--mut);margin:2px 0 10px}
.kgrp{margin:4px 0;padding:8px 10px;border:1px solid var(--line);border-radius:8px}
.kgrp .kn{font-weight:600;color:#0f172a}
.hide{display:none}
</style></head>
<body><div class="wrap">
<h1>Roofline — <span id="arch"></span> <span style="color:var(--mut);font-weight:400">· traced forward</span></h1>
<div class="sub">per-device v7x · costs from the real <span class="mono">standalone_trace</span> jaxpr (validated == analytic descriptor) · weights/quant + context applied analytically</div>
<div class="card"><div class="toggle" id="phsel"><button data-ph="prefill" class="on">prefill</button><button data-ph="decode">decode</button></div>
<div id="summary"></div></div>
<div class="tabs" id="tabs"></div>
<div class="card"><div id="pane"></div></div>
</div>
<script>
const D=__DATA__;
const CAT={moe:"#d62728",linear:"#1f77b4",o_proj:"#0d9488",attention:"#2ca02c",router:"#9467bd",lm_head:"#8c564b",norm:"#e377c2",rope:"#7f7f7f",other:"#bcbd22",embedding:"#17becf"};
const P=D.peaks; let PH="prefill"; let TAB="rf";
const g=id=>document.getElementById(id);
function fmt(x){const a=Math.abs(x);if(a===0)return"0";if(a>=100)return x.toFixed(0);if(a>=1)return x.toFixed(1);if(a>=0.01)return x.toFixed(2);if(a>=1e-4)return x.toFixed(4);return x.toExponential(1);}
g("arch").textContent=D.arch;

function summary(){const L=D.parallelism,ph=D.phases[PH],m=ph.meta;
 g("summary").innerHTML=
  "<b>mesh</b> data="+L.dp+" × tensor="+L.t+" = "+L.devices+" devices &nbsp;<b>EP</b>="+L.ep+(L.enable_sp?" <span class='pill sp'>+SP</span>":"")
  +"<br><b>"+PH+"</b> · tokens/DP="+m.tokens_per_dp+" · global tokens="+m.global_tokens+" · seq_len="+m.seq_len
  +" · layers: "+m.n_attn_full+" full + "+m.n_attn_swa+" SWA attn, "+m.n_moe+" MoE"
  +"<br><b>bound:</b> <span class='tag b-"+ph.tbound+"'>"+ph.tbound+"</span> &nbsp; step ≈ <b>"+ph.total_ms.toFixed(2)+" ms</b>"
  +" <span class='pill'>compute "+ph.Tc.toFixed(2)+"ms</span><span class='pill'>HBM "+ph.Th.toFixed(2)+"ms</span><span class='pill'>ICI "+ph.Ti.toFixed(2)+"ms</span>";}

const TABS=[["rf","Roofline"],["tbl","Cost table"],["cp","Code path"],["kn","Kernels"],["op","Ops"]];
function tabbar(){g("tabs").innerHTML=TABS.map(t=>"<div class='tab"+(t[0]===TAB?" on":"")+"' data-t='"+t[0]+"'>"+t[1]+"</div>").join("");
 g("tabs").querySelectorAll(".tab").forEach(e=>e.onclick=()=>{TAB=e.dataset.t;render();});}

function render(){summary();tabbar();const ph=D.phases[PH];const pane=g("pane");
 if(TAB==="rf"){pane.innerHTML="<canvas id='cv'></canvas>";drawRoofline(ph);}
 else if(TAB==="tbl")pane.innerHTML=costTable(ph);
 else if(TAB==="cp")pane.innerHTML=codePath();
 else if(TAB==="kn")pane.innerHTML=kernels();
 else if(TAB==="op")pane.innerHTML=ops();}

function costTable(ph){let h="<table><thead><tr><th class='l'>category</th><th>count</th><th>TFLOP</th><th>HBM GB</th><th>ICI GB</th><th>OI</th><th>compute ms</th><th>HBM ms</th><th>ICI ms</th><th>ideal ms</th><th>%</th><th>bound</th></tr></thead><tbody>";
 for(const r of ph.rows)h+="<tr><td class='l'><span style='color:"+(CAT[r.cat]||'#888')+"'>●</span> "+r.cat+(r.peak==='fp8'?" <span class='tag fp8'>fp8</span>":"")+"</td><td>"+r.count+"</td><td>"+fmt(r.flops/1e12)+"</td><td>"+fmt(r.hbm/1e9)+"</td><td>"+fmt(r.ici/1e9)+"</td><td>"+r.oi.toFixed(1)+"</td><td>"+fmt(r.compute_ms)+"</td><td>"+fmt(r.hbm_ms)+"</td><td>"+fmt(r.ici_ms)+"</td><td>"+r.ideal.toFixed(3)+"</td><td>"+r.pct.toFixed(0)+"%</td><td><span class='tag b-"+r.bound+"'>"+r.bound+"</span></td></tr>";
 h+="<tr style='font-weight:700'><td class='l'>Σ</td><td></td><td></td><td></td><td></td><td></td><td>"+fmt(ph.Tc)+"</td><td>"+fmt(ph.Th)+"</td><td>"+fmt(ph.Ti)+"</td><td>"+ph.total_ms.toFixed(3)+"</td><td>100%</td><td><span class='tag b-"+ph.tbound+"'>"+ph.tbound+"</span></td></tr>";
 return h+"</tbody></table>";}

function codePath(){const C=D.codepath;let h="<div class='note'>Every op group, its category, and the REAL <span class='mono'>models/*.py</span> call chain captured from the traced forward (innermost frame = op kind, outer = role). This is the actual code path, not a hand-written descriptor.</div>";
 h+="<table><thead><tr><th class='l'>role</th><th class='l'>category</th><th>count</th><th class='l'>code path (innermost → caller)</th></tr></thead><tbody>";
 for(const r of C.gemms){const chain=(r.stack||[]).slice(0,5).map((f,i)=>i===0?("<b>"+f+"</b>"):f).join(" <span class='arrow'>←</span> ");
  h+="<tr><td class='l'>"+r.role+"</td><td class='l'><span style='color:"+(CAT[r.category]||'#888')+"'>●</span> "+r.category+"</td><td>"+r.count+"</td><td class='l mono chain'>"+chain+"</td></tr>";}
 return h+"</tbody></table>";}

function kernels(){const C=D.codepath;let h="<div class='note'>Pallas kernels found in the traced forward ("+C.num_eqns_all.toLocaleString()+" total equations, "+C.num_eqns_top.toLocaleString()+" at top level). Real kernel names + per-device input/output avals + the shard_map call site.</div>";
 for(const k of C.kernels){const av=a=>(a||[]).map(x=>x.dtype+"["+x.shape.join(",")+"]").join(", ");
  h+="<div class='kgrp'><div class='kn'><span style='color:"+(CAT[k.kind==='attention'?'attention':(k.kind==='moe'?'moe':'other')]||'#888')+"'>●</span> "+k.name+" <span class='pill'>×"+k.count+"</span> <span class='pill'>"+k.kind+"</span></div>"
   +"<div class='mono' style='color:#667;font-size:11px;margin-top:3px'>in: "+av(k.in_avals)+"<br>out: "+av(k.out_avals)+"<br>@ "+k.ctx+"</div></div>";}
 return h;}

function ops(){const C=D.codepath;let h="<div class='note'>Primitive histogram of the full recursive jaxpr (top 30). The real lowering of the forward, all the way into kernel bodies.</div>";
 h+="<table style='width:auto'><thead><tr><th>count</th><th class='l'>primitive</th></tr></thead><tbody>";
 for(const p of C.primitives)h+="<tr><td>"+p[1].toLocaleString()+"</td><td class='l mono'>"+p[0]+"</td></tr>";
 return h+"</tbody></table>";}

// ---------- roofline canvas ----------
function drawRoofline(ph){const cv=g("cv"),cx=cv.getContext("2d");const dpr=window.devicePixelRatio||1;
 const W=Math.max(560,g("pane").clientWidth),Hh=600;cv.style.width=W+"px";cv.style.height=Hh+"px";cv.width=W*dpr;cv.height=Hh*dpr;cx.setTransform(dpr,0,0,dpr,0,0);
 const ml=84,mr=22,mt=20,mb=48;const HBMBW=P.hbm*1e9;
 const rows=ph.rows.filter(r=>r.flops>0&&r.hbm>0);if(!rows.length){cx.fillText("no costed ops",20,20);return;}
 const ceil=rows.some(r=>r.peak==="fp8")?P.fp8:P.bf16;
 const oiv=rows.map(r=>r.oi);let xmin=Math.min(...oiv)/3,xmax=Math.max(...oiv)*3;if(!(xmin>0))xmin=.01;
 const perfs=rows.map(r=>r.flops/(r.ideal/1e3)/1e12);let ymax=ceil*2.2,ymin=Math.min(...perfs.filter(p=>p>0),ceil)/80;if(!(ymin>0))ymin=ceil/1000;
 const lx=v=>ml+(Math.log10(v)-Math.log10(xmin))/(Math.log10(xmax)-Math.log10(xmin))*(W-ml-mr);
 const ly=v=>mt+(Math.log10(ymax)-Math.log10(v))/(Math.log10(ymax)-Math.log10(ymin))*(Hh-mt-mb);
 cx.clearRect(0,0,W,Hh);cx.strokeStyle="#eef1f5";cx.lineWidth=1;cx.font="11px sans-serif";
 for(let e=-3;e<=7;e++){const x=Math.pow(10,e);if(x<xmin||x>xmax)continue;cx.beginPath();cx.moveTo(lx(x),mt);cx.lineTo(lx(x),Hh-mb);cx.stroke();cx.fillStyle="#889";cx.textAlign="center";cx.fillText("1e"+e,lx(x),Hh-mb+15);}
 cx.textAlign="right";for(let e=-3;e<=4;e++){const y=Math.pow(10,e);if(y<ymin||y>ymax)continue;cx.beginPath();cx.moveTo(ml,ly(y));cx.lineTo(W-mr,ly(y));cx.stroke();cx.fillStyle="#889";cx.fillText("1e"+e,ml-8,ly(y)+3);}
 cx.textAlign="center";cx.fillStyle="#445";cx.font="12px sans-serif";cx.fillText("operational intensity (FLOP / HBM-byte)",(ml+W-mr)/2,Hh-6);
 cx.save();cx.translate(20,(mt+Hh-mb)/2);cx.rotate(-Math.PI/2);cx.fillText("attainable TFLOP/s",0,0);cx.restore();cx.textAlign="left";
 cx.strokeStyle="#334155";cx.lineWidth=2.5;cx.beginPath();let f=1;
 for(let i=0;i<=160;i++){const x=xmin*Math.pow(xmax/xmin,i/160),y=Math.min(x*HBMBW/1e12,ceil);if(f){cx.moveTo(lx(x),ly(y));f=0;}else cx.lineTo(lx(x),ly(y));}cx.stroke();
 cx.setLineDash([6,4]);cx.strokeStyle="#94a3b8";cx.lineWidth=1.2;
 for(const C2 of [P.bf16,P.fp8]){cx.beginPath();cx.moveTo(ml,ly(C2));cx.lineTo(W-mr,ly(C2));cx.stroke();}
 cx.setLineDash([]);cx.fillStyle="#64748b";cx.font="11px sans-serif";
 cx.fillText("bf16 "+P.bf16.toFixed(0)+" TF/s",ml+8,ly(P.bf16)-4);cx.fillText("fp8 "+P.fp8.toFixed(0)+" TF/s",ml+8,ly(P.fp8)-4);
 const smax=Math.max(...rows.map(r=>r.ideal))||1;
 for(const r of rows){const x=r.oi,y=r.flops/(r.ideal/1e3)/1e12,px=lx(x),py=ly(y),rad=6+15*(r.ideal/smax);
  cx.fillStyle=CAT[r.cat]||"#888";
  if(r.bound==="ICI"){cx.save();cx.translate(px,py);cx.rotate(Math.PI/4);cx.lineWidth=3.5;cx.strokeStyle=CAT[r.cat]||"#888";cx.beginPath();cx.moveTo(-rad,0);cx.lineTo(rad,0);cx.moveTo(0,-rad);cx.lineTo(0,rad);cx.stroke();cx.restore();}
  else{cx.strokeStyle="#fff";cx.lineWidth=1.5;cx.beginPath();cx.arc(px,py,rad,0,7);cx.fill();cx.stroke();}
  if(r.ideal>0.03*smax){cx.fillStyle="#1c2330";cx.font="11px sans-serif";const tx=px+rad+4>W-mr-50?px-rad-4-cx.measureText(r.cat).width:px+rad+4;cx.fillText(r.cat,tx,py+4);}}
}
g("phsel").querySelectorAll("button").forEach(b=>b.onclick=()=>{PH=b.dataset.ph;g("phsel").querySelectorAll("button").forEach(x=>x.classList.toggle("on",x===b));render();});
window.addEventListener("resize",()=>{if(TAB==="rf")render();});
render();
</script></body></html>
"""
