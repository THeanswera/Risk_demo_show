"""
Визуальная шкала сравнения ИПР
"""

import math

from utils.helpers import clamp


def risk_gauge_html(r_value: float, r_norm: float) -> str:
  
    r_value = float(r_value) if (r_value and r_value > 0) else 0.0
    r_norm = float(r_norm) if (r_norm and r_norm > 0) else 1e-12

    values = [v for v in [r_value, r_norm] if v > 0]
    if not values:
        return "<p>Нет данных для отображения</p>"

    log_min = math.floor(math.log10(min(values)))
    log_max = math.ceil(math.log10(max(values)))
    if (log_max - log_min) < 6:
        center = (log_max + log_min) / 2.0
        log_min = math.floor(center - 3)
        log_max = math.ceil(center + 3)
    if log_max == log_min:
        log_max = log_min + 1

    def y_log_pct(v: float) -> float:
        v = max(v, 1e-12)
        exp = math.log10(v)
        exp = clamp(exp, log_min, log_max)
        t = (log_max - exp) / (log_max - log_min)
        return clamp(100.0 * t, 0.0, 100.0)

    ticks = list(range(int(log_min), int(log_max) + 1))
    ticks_html = "\n".join([
        f'<div class="tick" style="top:{y_log_pct(10.0**p):.4f}%">'
        f'<div class="tick-line"></div><div class="tick-label">10<sup>{p}</sup></div></div>'
        for p in ticks
    ])

    passed = r_value <= r_norm
    status_color = "#4ade80" if passed else "#f87171"
    status_text = "ДОПУСТИМЫЙ" if passed else "ПРЕВЫШЕН"

    pct_of_norm = (r_value / r_norm) * 100.0 if r_norm > 0 else 0.0
    pct_max = max(100.0, pct_of_norm, 1.0)
    h_norm = clamp(100.0 * (100.0 / pct_max), 0.0, 100.0)
    h_val = clamp(100.0 * (pct_of_norm / pct_max), 0.0, 100.0)

    html = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<style>
  :root {{
    --bg: rgba(255,255,255,0.03);
    --bd: rgba(255,255,255,0.15);
    --txt: rgba(255,255,255,0.90);
    --muted: rgba(255,255,255,0.70);
    --grid: rgba(255,255,255,0.15);
    --norm: #f87171;
    --val: {status_color};
  }}
  body {{ margin:0; background:transparent; font-family: system-ui, Segoe UI, Arial; color:var(--txt); }}
  .wrap {{ display:grid; grid-template-columns: 240px 1fr; gap:16px; padding:8px 2px; }}
  .card {{ background:var(--bg); border:1px solid var(--bd); border-radius:14px; padding:14px; }}
  .title {{ font-size:12px; color:var(--muted); margin:0 0 10px 0; }}

  .scale-box {{ position:relative; height:300px; border-radius:12px; background:rgba(0,0,0,0.15); border:1px solid rgba(255,255,255,0.10); overflow:hidden; }}
  .axis {{ position:absolute; left:16px; top:10px; bottom:10px; width:1px; background:rgba(255,255,255,0.18); }}
  .tick {{ position:absolute; left:18px; right:10px; transform:translateY(-50%); display:flex; gap:8px; align-items:center; }}
  .tick-line {{ height:1px; flex:1; background:var(--grid); }}
  .tick-label {{ font-size:11px; color:var(--muted); white-space:nowrap; }}

  .marker {{ position:absolute; left:10px; right:10px; transform:translateY(-50%); pointer-events:none; }}
  .marker-line {{ height:4px; border-radius:999px; }}
  .marker-text {{ font-size:11px; font-weight:750; line-height:1.1; white-space:nowrap; text-shadow: 0 1px 10px rgba(0,0,0,0.65); }}
  .m-norm .marker-line {{ background:var(--norm); }}
  .m-norm .marker-text {{ color:var(--norm); }}
  .m-val .marker-line {{ background:var(--val); }}
  .m-val .marker-text {{ color:var(--val); }}

  .bars-box {{ position:relative; height:300px; border-radius:12px; background:rgba(0,0,0,0.15); border:1px solid rgba(255,255,255,0.10); overflow:hidden; padding:12px; }}
  .bars {{ position:absolute; left:12px; right:12px; top:40px; bottom:16px; display:flex; gap:24px; align-items:stretch; }}
  .col {{ flex:1; display:flex; flex-direction:column; height:100%; }}
  .barwrap {{ flex:1; display:flex; align-items:flex-end; justify-content:center; }}
  .bar {{ width:70%; border-radius:12px 12px 8px 8px; box-shadow:0 8px 30px rgba(0,0,0,0.25); }}
  .b-norm {{ background:var(--norm); }}
  .b-val {{ background:var(--val); }}
  .lbl {{ margin-top:10px; text-align:center; }}
  .name {{ font-weight:800; font-size:13px; }}
  .val {{ font-size:12px; color:var(--muted); margin-top:2px; }}

  .status {{ text-align:center; margin-top:12px; padding:8px; border-radius:8px; font-weight:800; font-size:14px;
    background: rgba({'74,222,128' if passed else '248,113,113'},0.15);
    color: var(--val); border: 1px solid var(--val); }}
</style>
</head>
<body>
<div class="wrap">
  <div class="card">
    <div class="title">Логарифмическая шкала ИПР</div>
    <div class="scale-box">
      <div class="axis"></div>
      {ticks_html}
      <div class="marker m-norm" style="top:{y_log_pct(r_norm):.4f}%">
        <div class="marker-line"></div>
        <div class="marker-text" style="transform:translateY(-24px);">R<sub>норм</sub> = {r_norm:.1e}</div>
      </div>
      {"" if r_value <= 0 else f'''
      <div class="marker m-val" style="top:{y_log_pct(r_value):.4f}%">
        <div class="marker-line"></div>
        <div class="marker-text" style="transform:translateY(8px);">R = {r_value:.2e}</div>
      </div>
      '''}
    </div>
  </div>
  <div class="card">
    <div class="title">Сравнение R с R<sub>норм</sub></div>
    <div class="bars-box">
      <div class="bars">
        <div class="col">
          <div class="barwrap"><div class="bar b-norm" style="height:{h_norm:.4f}%;"></div></div>
          <div class="lbl">
            <div class="name" style="color:var(--norm)">R<sub>норм</sub></div>
            <div class="val">{r_norm:.1e} год⁻¹</div>
            <div class="val">100%</div>
          </div>
        </div>
        <div class="col">
          <div class="barwrap"><div class="bar b-val" style="height:{h_val:.4f}%;"></div></div>
          <div class="lbl">
            <div class="name" style="color:var(--val)">R<sub>расч</sub></div>
            <div class="val">{r_value:.2e} год⁻¹</div>
            <div class="val">{pct_of_norm:.1f}%</div>
          </div>
        </div>
      </div>
    </div>
    <div class="status">РИСК {status_text}</div>
  </div>
</div>
</body>
</html>
"""
    return html
