"""
Калькулятор индивидуального пожарного риска (ИПР)
по Приказу МЧС России от 14.11.2022 № 1140

Методика определения расчётных величин пожарного риска
в зданиях, сооружениях и пожарных отсеках различных классов
функциональной пожарной опасности.

Реализованы формулы (1)–(8) методики, справочные таблицы
из Приложений 3, 4, 9.
"""

import math
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


# ═══════════════════════════════════════════════════════
# КОНСТАНТЫ МЕТОДИКИ №1140
# ═══════════════════════════════════════════════════════
R_NORM = 1e-6          # Нормативное значение ИПР, год⁻¹ (п. 8)
P_E_MAX = 0.999        # Максимальная вероятность эвакуации (формула 6)
K_STD = 0.8            # Стандартное значение K_обн, K_СОУЭ, K_ПДЗ (п. 41, 44, 45)
K_MAX = 0.99           # Верхний предел для коэффициентов
K_AP_VALUE = 0.9       # Значение K_ап при соответствии АУП (п. 15)
P_DOOR_OPEN = 0.3      # Вероятность открытого положения двери (п. 48, формула 8)
P_DOOR_CLOSED = 0.7    # Вероятность закрытого положения двери (п. 48, формула 8)


# ═══════════════════════════════════════════════════════
# СПРАВОЧНЫЕ ТАБЛИЦЫ ИЗ ПРИЛОЖЕНИЙ К МЕТОДИКЕ
# ═══════════════════════════════════════════════════════

# Приложение 3, Таблица П3.1 — Частота возникновения пожара
FIRE_FREQ_TABLE = {
    "Общеобразовательные организации": 1.16e-2,
    "Организации начального профессионального образования (ПТУ)": 1.98e-2,
    "Организации среднего профессионального образования (ССУЗ)": 2.69e-2,
    "Дошкольные образовательные организации": 1.3e-3,
    "Детские оздоровительные лагеря, летние детские дачи": 1.26e-3,
    "Санатории, дома отдыха, пансионаты": 2.99e-2,
    "Амбулатории, поликлиники, диспансеры, медпункты": 8.88e-3,
    "Здания розничной торговли": 2.03e-2,
    "Здания рыночной торговли": 1.13e-2,
    "Здания организаций общественного питания": 3.88e-2,
    "Гостиницы, мотели": 2.81e-2,
    "Спортивные сооружения": 1.83e-3,
    "Здания зрелищных и культурно-просветительных учреждений": 6.90e-3,
    "Библиотеки": 1.16e-3,
    "Музеи": 1.38e-2,
    "Больницы": 1.3e-2,
    "Образовательные организации с наличием интерната": 7.7e-3,
    "Специализированные дома престарелых и инвалидов": 7.7e-3,
    "Дома жилые многоквартирные": 2.6e-2,
    "Дома жилые одноквартирные": 1.9e-3,
    "Стоянки автомобилей": 4.5e-2,
    "Здания производственного и складского назначения": 1.9e-2,
    "Здания религиозного назначения": 3.2e-3,
    "Иное (Q_п = 4·10⁻²)": 4.0e-2,
}

# Приложение 4, Таблица П4.1 — Время начала эвакуации (максимальное) по классу Ф
T_NE_TABLE = {
    "Ф1.1, Ф1.3, Ф1.4 — Дошкольные, дома престарелых, жилые (могут спать, знакомы с путями)": 6.0,
    "Ф1.2 — Гостиницы, общежития, санатории (могут спать, не знакомы с путями)": 6.0,
    "Ф2, Ф3 — Зрелищные, обслуживание населения (бодрствуют, могут не знать путей)": 3.0,
    "Ф4 — Научные, образовательные, органы управления (бодрствуют, знакомы с путями)": 3.0,
    "Ф5 — Стоянки, производственные помещения": 3.0,
}

# Приложение 9, Таблица П9.1 — Параметры пожарной нагрузки
FIRE_LOAD_TABLE = pd.DataFrame([
    {"Вид помещения": "Жилые помещения гостиниц, общежитий и т.д.",
     "Q_н (МДж/кг)": 13.8, "D_m (Нп·м²/кг)": 270, "ψ_уд (кг/(м²·с))": 0.0145,
     "v_л (м/с)": 0.0045, "L_уд (кг/кг)": 1.03, "CO₂ (кг/кг)": 0.203,
     "CO (кг/кг)": 0.0022, "HCl (кг/кг)": 0.014},
    {"Вид помещения": "Столовая, зал ресторана и т.д.",
     "Q_н (МДж/кг)": 13.8, "D_m (Нп·м²/кг)": 82, "ψ_уд (кг/(м²·с))": 0.0145,
     "v_л (м/с)": 0.0045, "L_уд (кг/кг)": 1.437, "CO₂ (кг/кг)": 1.285,
     "CO (кг/кг)": 0.0022, "HCl (кг/кг)": 0.006},
    {"Вид помещения": "Зал театра, кинотеатра, клуба, цирка и т.д.",
     "Q_н (МДж/кг)": 13.8, "D_m (Нп·м²/кг)": 270, "ψ_уд (кг/(м²·с))": 0.0145,
     "v_л (м/с)": 0.0055, "L_уд (кг/кг)": 1.03, "CO₂ (кг/кг)": 0.203,
     "CO (кг/кг)": 0.0022, "HCl (кг/кг)": 0.014},
    {"Вид помещения": "Гардеробы",
     "Q_н (МДж/кг)": 16.7, "D_m (Нп·м²/кг)": 61, "ψ_уд (кг/(м²·с))": 0.025,
     "v_л (м/с)": 0.007, "L_уд (кг/кг)": 2.56, "CO₂ (кг/кг)": 0.88,
     "CO (кг/кг)": 0.063, "HCl (кг/кг)": 0.0},
    {"Вид помещения": "Хранилища библиотек, архивы",
     "Q_н (МДж/кг)": 14.5, "D_m (Нп·м²/кг)": 49.5, "ψ_уд (кг/(м²·с))": 0.011,
     "v_л (м/с)": 0.008, "L_уд (кг/кг)": 1.154, "CO₂ (кг/кг)": 1.1087,
     "CO (кг/кг)": 0.0974, "HCl (кг/кг)": 0.0},
    {"Вид помещения": "Музеи, выставки",
     "Q_н (МДж/кг)": 13.8, "D_m (Нп·м²/кг)": 270, "ψ_уд (кг/(м²·с))": 0.0145,
     "v_л (м/с)": 0.0055, "L_уд (кг/кг)": 1.03, "CO₂ (кг/кг)": 0.203,
     "CO (кг/кг)": 0.0022, "HCl (кг/кг)": 0.014},
    {"Вид помещения": "Подсобные и бытовые помещения",
     "Q_н (МДж/кг)": 14.0, "D_m (Нп·м²/кг)": 53.0, "ψ_уд (кг/(м²·с))": 0.0129,
     "v_л (м/с)": 0.0042, "L_уд (кг/кг)": 1.161, "CO₂ (кг/кг)": 0.642,
     "CO (кг/кг)": 0.0317, "HCl (кг/кг)": 0.0},
    {"Вид помещения": "Административные помещения, учебные классы, палаты больниц",
     "Q_н (МДж/кг)": 14.0, "D_m (Нп·м²/кг)": 47.7, "ψ_уд (кг/(м²·с))": 0.0137,
     "v_л (м/с)": 0.0045, "L_уд (кг/кг)": 1.369, "CO₂ (кг/кг)": 1.478,
     "CO (кг/кг)": 0.03, "HCl (кг/кг)": 0.0058},
    {"Вид помещения": "Магазины",
     "Q_н (МДж/кг)": 15.8, "D_m (Нп·м²/кг)": 270, "ψ_уд (кг/(м²·с))": 0.015,
     "v_л (м/с)": 0.0055, "L_уд (кг/кг)": 1.25, "CO₂ (кг/кг)": 0.85,
     "CO (кг/кг)": 0.043, "HCl (кг/кг)": 0.023},
    {"Вид помещения": "Зал вокзала",
     "Q_н (МДж/кг)": 13.8, "D_m (Нп·м²/кг)": 270, "ψ_уд (кг/(м²·с))": 0.0145,
     "v_л (м/с)": 0.0055, "L_уд (кг/кг)": 1.03, "CO₂ (кг/кг)": 0.203,
     "CO (кг/кг)": 0.0022, "HCl (кг/кг)": 0.014},
    {"Вид помещения": "Стоянки легковых автомобилей",
     "Q_н (МДж/кг)": 31.7, "D_m (Нп·м²/кг)": 487, "ψ_уд (кг/(м²·с))": 0.023,
     "v_л (м/с)": 0.0068, "L_уд (кг/кг)": 2.64, "CO₂ (кг/кг)": 1.3,
     "CO (кг/кг)": 0.097, "HCl (кг/кг)": 0.011},
    {"Вид помещения": "Стоянки легковых автомобилей с двухуровневым хранением",
     "Q_н (МДж/кг)": 31.7, "D_m (Нп·м²/кг)": 487, "ψ_уд (кг/(м²·с))": 0.023,
     "v_л (м/с)": 0.0136, "L_уд (кг/кг)": 2.64, "CO₂ (кг/кг)": 1.3,
     "CO (кг/кг)": 0.097, "HCl (кг/кг)": 0.011},
    {"Вид помещения": "Стадионы",
     "Q_н (МДж/кг)": 26.4, "D_m (Нп·м²/кг)": 78, "ψ_уд (кг/(м²·с))": 0.004,
     "v_л (м/с)": 0.004, "L_уд (кг/кг)": 2.09, "CO₂ (кг/кг)": 1.8,
     "CO (кг/кг)": 0.127, "HCl (кг/кг)": 0.0},
    {"Вид помещения": "Спортзалы",
     "Q_н (МДж/кг)": 16.7, "D_m (Нп·м²/кг)": 61, "ψ_уд (кг/(м²·с))": 0.024,
     "v_л (м/с)": 0.0045, "L_уд (кг/кг)": 2.56, "CO₂ (кг/кг)": 0.88,
     "CO (кг/кг)": 0.063, "HCl (кг/кг)": 0.0},
])


# ═══════════════════════════════════════════════════════
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ═══════════════════════════════════════════════════════

def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def safe_float(x, default=0.0) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return float(default)
        return v
    except Exception:
        return float(default)


def force_rerun():
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()


def ensure_unique_positive_int_ids(df: pd.DataFrame, col: str, start_from: int = 1) -> pd.DataFrame:
    df = df.copy()
    if col not in df.columns:
        df.insert(0, col, np.arange(start_from, start_from + len(df), dtype=int))
        return df
    ids = pd.to_numeric(df[col], errors="coerce")
    used = set()
    next_id = start_from
    if ids.notna().any():
        next_id = int(ids.dropna().max()) + 1
    new_ids = []
    for v in ids.to_list():
        if pd.isna(v):
            while next_id in used:
                next_id += 1
            new_ids.append(next_id)
            used.add(next_id)
            next_id += 1
            continue
        iv = int(v)
        if iv <= 0 or iv in used:
            while next_id in used:
                next_id += 1
            new_ids.append(next_id)
            used.add(next_id)
            next_id += 1
        else:
            new_ids.append(iv)
            used.add(iv)
    df[col] = new_ids
    return df


def fmt_sci(x, digits: int = 2) -> str:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return ""
        return f"{v:.{digits}e}"
    except Exception:
        return str(x)


def format_df_scientific(df: pd.DataFrame, sci_cols: list, digits: int = 2) -> pd.DataFrame:
    out = df.copy()
    for c in sci_cols:
        if c in out.columns:
            out[c] = out[c].apply(lambda v: fmt_sci(v, digits=digits) if pd.notna(v) else "")
    return out


# ═══════════════════════════════════════════════════════
# ФОРМУЛЫ МЕТОДИКИ №1140
# ═══════════════════════════════════════════════════════

def p_presence(t_pr_hours: float) -> float:
    """P_пр,i = t_пр,i / 24 — формула (5)"""
    return clamp(safe_float(t_pr_hours) / 24.0, 0.0, 1.0)


def p_evac(t_p: float, t_bl: float, t_ne: float, t_ck: float) -> float:
    """
    P_э,i,j — формула (6) №1140 (кусочная).

    t_p  — расчётное время эвакуации, мин
    t_bl — время блокирования путей эвакуации, мин
    t_ne — время начала эвакуации, мин
    t_ck — время существования скоплений, мин
    """
    t_p = safe_float(t_p)
    t_bl = safe_float(t_bl)
    t_ne = max(1e-9, safe_float(t_ne))
    t_ck = safe_float(t_ck)

    border = 0.8 * t_bl

    # Ветвь 3: t_ск > 6 или t_p ≥ 0.8·t_бл → P_э = 0
    if t_ck > 6.0:
        return 0.0

    # Ветвь 2: полная эвакуация
    if (t_p + t_ne) <= border:
        return P_E_MAX

    # Ветвь 1: промежуточное значение
    if (t_p < border) and (border < (t_p + t_ne)):
        val = P_E_MAX * ((border - t_p) / t_ne)
        return clamp(val, 0.0, P_E_MAX)

    return 0.0


def k_pz(k_obn: float, k_soue: float, k_pdz: float) -> float:
    """
    K_п.з,i — формула (7) №1140
    K_п.з,i = 1 - (1 - K_обн,i · K_СОУЭ,i) · (1 - K_обн,i · K_ПДЗ,i)
    """
    k_obn = clamp(safe_float(k_obn), 0.0, K_MAX)
    k_soue = clamp(safe_float(k_soue), 0.0, K_MAX)
    k_pdz = clamp(safe_float(k_pdz), 0.0, K_MAX)
    val = 1.0 - (1.0 - k_obn * k_soue) * (1.0 - k_obn * k_pdz)
    return clamp(val, 0.0, 1.0)


def r_ij(q_p: float, k_ap: float, p_pr: float, p_e: float, k_pz_val: float) -> float:
    """
    R_i,j — формула (4) №1140
    R_i,j = Q_п,i · (1 - K_ап,i) · P_пр,i · (1 - P_э,i,j) · (1 - K_п.з,i)
    """
    q_p = max(0.0, safe_float(q_p))
    k_ap = clamp(safe_float(k_ap), 0.0, K_AP_VALUE)
    p_pr = clamp(safe_float(p_pr), 0.0, 1.0)
    p_e = clamp(safe_float(p_e), 0.0, P_E_MAX)
    k_pz_val = clamp(safe_float(k_pz_val), 0.0, 1.0)
    val = q_p * (1.0 - k_ap) * p_pr * (1.0 - p_e) * (1.0 - k_pz_val)
    return max(0.0, float(val))


def r_i_with_doors(r_open: float, r_closed: float) -> float:
    """
    Формула (8) №1140 — учёт противопожарных дверей (п. 48).
    R_i = P_откр · R_i(откр) + P_закр · R_i(закр)
    P_откр = 0.3, P_закр = 0.7
    """
    return P_DOOR_OPEN * safe_float(r_open) + P_DOOR_CLOSED * safe_float(r_closed)


def t_ne_formula(f_pom: float) -> float:
    """
    t_НЭ = 5 + 0.01 · F_пом (с) — формула (П4.1), результат в минутах
    """
    t_sec = 5.0 + 0.01 * safe_float(f_pom)
    return t_sec / 60.0


# ═══════════════════════════════════════════════════════
# ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
# ═══════════════════════════════════════════════════════

def risk_gauge_html(r_value: float, r_norm: float) -> str:
    """Визуальная шкала ИПР: логарифмическая + столбцы % от R_норм"""
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


# ═══════════════════════════════════════════════════════
# ОСНОВНОЙ РАСЧЁТ
# ═══════════════════════════════════════════════════════

def compute_all(
    df_scen_in: pd.DataFrame,
    df_grp_in: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    """
    Полный расчёт ИПР по формулам (2)–(7) Методики №1140.
    Возвращает: (df_scen, df_rows, df_agg, r_total)
    """
    df_scen = df_scen_in.copy()
    df_grp = df_grp_in.copy()

    df_scen["Сценарий i"] = pd.to_numeric(df_scen["Сценарий i"], errors="coerce").fillna(0).astype(int)
    df_grp["Сценарий i"] = pd.to_numeric(df_grp["Сценарий i"], errors="coerce").fillna(0).astype(int)

    if df_scen["Сценарий i"].duplicated().any():
        df_scen = df_scen.drop_duplicates(subset=["Сценарий i"], keep="first").copy()

    for c in ["Q_п,i (год⁻¹)", "t_пр,i (ч/сут)", "t_бл,i (мин)", "K_ап,i"]:
        if c in df_scen.columns:
            df_scen[c] = pd.to_numeric(df_scen[c], errors="coerce").fillna(0.0)

    for c in ["t_р,i,j (мин)", "t_н.э,i,j (мин)", "t_ск,i,j (мин)"]:
        if c in df_grp.columns:
            df_grp[c] = pd.to_numeric(df_grp[c], errors="coerce").fillna(0.0)

    # K_обн, K_СОУЭ, K_ПДЗ — 0 или 0.8 строго по №1140
    for col_bool, col_k in [
        ("ПС соответствует? (K_обн=0.8)", "K_обн,i"),
        ("СОУЭ соответствует? (K_СОУЭ=0.8)", "K_СОУЭ,i"),
        ("ПДЗ соответствует? (K_ПДЗ=0.8)", "K_ПДЗ,i"),
    ]:
        if col_bool in df_scen.columns:
            df_scen[col_k] = np.where(df_scen[col_bool].astype(bool), K_STD, 0.0)
        else:
            df_scen[col_k] = 0.0

    # P_пр — формула (5)
    if "t_пр,i (ч/сут)" in df_scen.columns:
        df_scen["P_пр,i"] = df_scen["t_пр,i (ч/сут)"].apply(p_presence)
    else:
        df_scen["P_пр,i"] = 0.5

    # K_п.з — формула (7)
    df_scen["K_п.з,i"] = df_scen.apply(
        lambda r: k_pz(r["K_обн,i"], r["K_СОУЭ,i"], r["K_ПДЗ,i"]),
        axis=1
    )

    # Объединяем группы со сценариями
    merge_cols = ["Сценарий i", "Q_п,i (год⁻¹)", "K_ап,i", "t_бл,i (мин)",
                  "P_пр,i", "K_п.з,i", "K_обн,i", "K_СОУЭ,i", "K_ПДЗ,i"]
    merge_cols = [c for c in merge_cols if c in df_scen.columns]

    df_rows = df_grp.merge(df_scen[merge_cols], on="Сценарий i", how="left")

    missing = df_rows["t_бл,i (мин)"].isna() | df_rows["Q_п,i (год⁻¹)"].isna()
    if missing.any():
        df_rows = df_rows.loc[~missing].copy()

    # P_э — формула (6)
    df_rows["P_э,i,j"] = df_rows.apply(
        lambda r: p_evac(
            r["t_р,i,j (мин)"], r["t_бл,i (мин)"], r["t_н.э,i,j (мин)"], r["t_ск,i,j (мин)"]
        ),
        axis=1
    )

    # R_i,j — формула (4)
    df_rows["R_i,j"] = df_rows.apply(
        lambda r: r_ij(
            r["Q_п,i (год⁻¹)"], r["K_ап,i"], r["P_пр,i"], r["P_э,i,j"], r["K_п.з,i"]
        ),
        axis=1
    )

    # R_i = max_j {R_i,j} — формула (3)
    agg = df_rows.groupby("Сценарий i", as_index=False).agg(
        **{"R_i = max_j(R_i,j)": ("R_i,j", "max")}
    )

    # R = max_i {R_i} — формула (2)
    r_total = float(agg["R_i = max_j(R_i,j)"].max()) if len(agg) else 0.0

    agg["R <= R_норм?"] = agg["R_i = max_j(R_i,j)"].apply(
        lambda x: "Да" if x <= R_NORM else "Нет"
    )

    total_row = pd.DataFrame([{
        "Сценарий i": "ИТОГО (R = max)",
        "R_i = max_j(R_i,j)": r_total,
        "R <= R_норм?": "Да" if r_total <= R_NORM else "Нет",
    }])
    agg_out = pd.concat([agg, total_row], ignore_index=True)

    return df_scen, df_rows, agg_out, r_total


# ═══════════════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════════════

st.set_page_config(
    page_title="Калькулятор ИПР по Приказу МЧС №1140",
    page_icon="🔥",
    layout="wide",
)

st.title("🔥 Калькулятор индивидуального пожарного риска")
st.caption("По Методике определения расчётных величин пожарного риска — Приказ МЧС России от 14.11.2022 № 1140")

# ─────────────────────────────────────────────────────
# Данные по умолчанию
# ─────────────────────────────────────────────────────

default_scenarios = pd.DataFrame([{
    "Сценарий i": 1,
    "Тип здания": "Иное (Q_п = 4·10⁻²)",
    "Q_п,i (год⁻¹)": 4.0e-2,
    "t_пр,i (ч/сут)": 12.0,
    "t_бл,i (мин)": 12.0,
    "K_ап,i": 0.9,
    "ПС соответствует? (K_обн=0.8)": True,
    "СОУЭ соответствует? (K_СОУЭ=0.8)": True,
    "ПДЗ соответствует? (K_ПДЗ=0.8)": True,
}])

default_groups = pd.DataFrame([
    {
        "ID": 1, "Сценарий i": 1, "Группа j": "Основной контингент",
        "t_р,i,j (мин)": 6.0, "t_н.э,i,j (мин)": 1.5, "t_ск,i,j (мин)": 1.0,
    },
    {
        "ID": 2, "Сценарий i": 1, "Группа j": "Маломобильные",
        "t_р,i,j (мин)": 7.0, "t_н.э,i,j (мин)": 2.0, "t_ск,i,j (мин)": 2.5,
    },
])

if "df_scen" not in st.session_state:
    st.session_state.df_scen = default_scenarios.copy()
if "df_grp" not in st.session_state:
    st.session_state.df_grp = default_groups.copy()
if "use_fire_doors" not in st.session_state:
    st.session_state.use_fire_doors = False
if "r_door_open" not in st.session_state:
    st.session_state.r_door_open = 0.0
if "r_door_closed" not in st.session_state:
    st.session_state.r_door_closed = 0.0

st.session_state.df_grp = ensure_unique_positive_int_ids(st.session_state.df_grp, "ID", start_from=1)

if "selected_group_id" not in st.session_state:
    st.session_state.selected_group_id = int(st.session_state.df_grp["ID"].iloc[0]) if len(st.session_state.df_grp) else 1


# ─────────────────────────────────────────────────────
# SIDEBAR: Справочники и настройки
# ─────────────────────────────────────────────────────

with st.sidebar:
    st.header("📋 Справочники и настройки")

    # --- Справочник Q_п ---
    st.subheader("Частота пожара Q_п,i")
    st.caption("Приложение 3, Таблица П3.1 (п. 15)")
    building_type = st.selectbox(
        "Тип здания",
        list(FIRE_FREQ_TABLE.keys()),
        index=list(FIRE_FREQ_TABLE.keys()).index("Иное (Q_п = 4·10⁻²)"),
        key="building_type_select"
    )
    q_p_ref = FIRE_FREQ_TABLE[building_type]
    st.metric("Q_п (год⁻¹)", f"{q_p_ref:.2e}")

    # Выбор сценария для применения Q_п
    df_scen_sidebar = st.session_state.df_scen.copy()
    df_scen_sidebar["Сценарий i"] = pd.to_numeric(df_scen_sidebar["Сценарий i"], errors="coerce").fillna(0).astype(int)
    scen_ids_for_qp = sorted(df_scen_sidebar.loc[df_scen_sidebar["Сценарий i"] > 0, "Сценарий i"].unique().tolist())
    if not scen_ids_for_qp:
        scen_ids_for_qp = [1]

    target_scen_qp = st.selectbox(
        "Применить к сценарию №",
        scen_ids_for_qp,
        key="target_scen_qp"
    )

    if st.button(f"📥 Применить Q_п к сценарию {target_scen_qp}", use_container_width=True):
        df = st.session_state.df_scen.copy()
        df["Сценарий i"] = pd.to_numeric(df["Сценарий i"], errors="coerce").fillna(0).astype(int)
        mask = df["Сценарий i"] == int(target_scen_qp)
        if mask.any():
            idx = df.index[mask][0]
            df.at[idx, "Q_п,i (год⁻¹)"] = q_p_ref
            df.at[idx, "Тип здания"] = building_type
            st.session_state.df_scen = df
            force_rerun()

    st.divider()

    # --- Справочник t_нэ ---
    st.subheader("Время начала эвакуации t_н.э")
    st.caption("Приложение 4 (п. 35)")

    t_ne_mode = st.radio(
        "Способ определения",
        ["По Таблице П4.1 (класс Ф)", "По формуле П4.1 (площадь помещения)"],
        key="t_ne_mode"
    )

    if t_ne_mode.startswith("По Таблице"):
        func_class = st.selectbox("Класс функциональной пожарной опасности", list(T_NE_TABLE.keys()))
        t_ne_ref = T_NE_TABLE[func_class]
        st.metric("t_н.э (мин)", f"{t_ne_ref:.1f}")
    else:
        f_pom = st.number_input("Площадь помещения F_пом (м²)", min_value=1.0, value=100.0, step=10.0)
        t_ne_ref = t_ne_formula(f_pom)
        st.metric("t_н.э (мин) = (5 + 0.01·F) / 60", f"{t_ne_ref:.3f}")
        st.caption(f"= {t_ne_ref * 60:.1f} секунд")

    # Выбор группы для применения t_н.э
    df_grp_sidebar = st.session_state.df_grp.copy()
    df_grp_sidebar = ensure_unique_positive_int_ids(df_grp_sidebar, "ID", start_from=1)
    if len(df_grp_sidebar) > 0:
        grp_labels_tne = df_grp_sidebar.apply(
            lambda r: f"ID {int(r['ID'])} | сцен. {int(r['Сценарий i'])} | {r.get('Группа j', '')}",
            axis=1
        ).to_list()

        target_grp_tne = st.selectbox(
            "Применить t_н.э к группе",
            grp_labels_tne,
            key="target_grp_tne"
        )

        if st.button(f"📥 Применить t_н.э к группе", use_container_width=True):
            sel_idx_tne = grp_labels_tne.index(target_grp_tne)
            grp_id_tne = int(df_grp_sidebar.iloc[sel_idx_tne]["ID"])
            grp_df = st.session_state.df_grp.copy()
            mask = grp_df["ID"].astype(int) == grp_id_tne
            if mask.any():
                idx = grp_df.index[mask][0]
                grp_df.at[idx, "t_н.э,i,j (мин)"] = t_ne_ref
                st.session_state.df_grp = grp_df
                force_rerun()

    st.divider()

    # --- Настройка слайдеров для выбранной группы ---
    st.subheader("Временные параметры")
    st.caption("Настройка для выбранной группы через слайдеры")

    df_grp_sel = st.session_state.df_grp.copy()
    if len(df_grp_sel) > 0:
        df_grp_sel["label"] = df_grp_sel.apply(
            lambda r: f"ID {int(r['ID'])} | сцен. {int(r['Сценарий i'])} | {r.get('Группа j', '')}",
            axis=1
        )
        labels = df_grp_sel["label"].to_list()
        sel_label = st.selectbox("Группа для настройки", labels, index=0, key="slider_group_select")

        sel_id = int(df_grp_sel.loc[df_grp_sel["label"] == sel_label, "ID"].iloc[0])
        st.session_state.selected_group_id = sel_id

        grp_df = st.session_state.df_grp.copy()
        row_idx = grp_df.index[grp_df["ID"] == sel_id][0]
        scen_id = int(grp_df.at[row_idx, "Сценарий i"])

        scen_df = st.session_state.df_scen.copy()
        scen_df["Сценарий i"] = pd.to_numeric(scen_df["Сценарий i"], errors="coerce").fillna(0).astype(int)
        scen_match = scen_df.index[scen_df["Сценарий i"] == scen_id]

        if len(scen_match) > 0:
            scen_idx = scen_match[0]
            t_bl_cur = safe_float(scen_df.at[scen_idx, "t_бл,i (мин)"], 12.0)

            t_bl_new = st.slider("t_бл,i (мин) — время блокирования", 0.5, 180.0, float(t_bl_cur), 0.5)
            t_p_new = st.slider("t_р,i,j (мин) — расчётное время эвакуации", 0.0, 180.0,
                                float(grp_df.at[row_idx, "t_р,i,j (мин)"]), 0.5)
            t_ne_new = st.slider("t_н.э,i,j (мин) — время начала эвакуации", 0.0, 60.0,
                                 float(grp_df.at[row_idx, "t_н.э,i,j (мин)"]), 0.5)
            t_ck_new = st.slider("t_ск,i,j (мин) — время скоплений", 0.0, 30.0,
                                 float(grp_df.at[row_idx, "t_ск,i,j (мин)"]), 0.5)

            scen_df.at[scen_idx, "t_бл,i (мин)"] = t_bl_new
            grp_df.at[row_idx, "t_р,i,j (мин)"] = t_p_new
            grp_df.at[row_idx, "t_н.э,i,j (мин)"] = t_ne_new
            grp_df.at[row_idx, "t_ск,i,j (мин)"] = t_ck_new

            st.session_state.df_scen = scen_df
            st.session_state.df_grp = grp_df
        else:
            st.warning("Для выбранной группы не найден сценарий.")
    else:
        st.info("Нет групп. Добавьте группу в таблице.")


# ─────────────────────────────────────────────────────
# ШАГ 1: СЦЕНАРИИ ПОЖАРА
# ─────────────────────────────────────────────────────

st.subheader("📝 Шаг 1. Сценарии пожара (п. 9–13)")
st.caption(
    "Определите сценарии пожара. Для каждого сценария задайте: частоту возникновения пожара Q_п,i "
    "(Приложение 3), время присутствия людей t_пр, время блокирования t_бл, "
    "коэффициент АУП K_ап (п. 15), и параметры систем ПС/СОУЭ/ПДЗ (п. 41, 44, 45)."
)

df_scen_raw = st.session_state.df_scen.copy()
if "Сценарий i" not in df_scen_raw.columns:
    df_scen_raw["Сценарий i"] = np.arange(1, len(df_scen_raw) + 1, dtype=int)
df_scen_raw["Сценарий i"] = pd.to_numeric(df_scen_raw["Сценарий i"], errors="coerce").fillna(0).astype(int)
df_scen_raw = df_scen_raw.loc[df_scen_raw["Сценарий i"] > 0].copy()

scen_list = sorted(df_scen_raw["Сценарий i"].unique().tolist())
if len(scen_list) == 0:
    scen_list = [1]

c1, c2, c3 = st.columns([1.2, 1.8, 2.0])
with c1:
    if st.button("Добавить сценарий", use_container_width=True):
        df = st.session_state.df_scen.copy()
        if len(df) == 0:
            next_i = 1
        else:
            df["Сценарий i"] = pd.to_numeric(df["Сценарий i"], errors="coerce").fillna(0).astype(int)
            next_i = int(df["Сценарий i"].max()) + 1
        new_row = {
            "Сценарий i": next_i,
            "Тип здания": "Иное (Q_п = 4·10⁻²)",
            "Q_п,i (год⁻¹)": 4.0e-2,
            "t_пр,i (ч/сут)": 12.0,
            "t_бл,i (мин)": 12.0,
            "K_ап,i": 0.9,
            "ПС соответствует? (K_обн=0.8)": True,
            "СОУЭ соответствует? (K_СОУЭ=0.8)": True,
            "ПДЗ соответствует? (K_ПДЗ=0.8)": True,
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        st.session_state.df_scen = df
        force_rerun()

with c2:
    scen_del = st.selectbox("Удалить сценарий", scen_list, key="scen_del_select")
    if st.button("Удалить сценарий", use_container_width=True):
        df_s = st.session_state.df_scen.copy()
        df_s["Сценарий i"] = pd.to_numeric(df_s["Сценарий i"], errors="coerce").fillna(0).astype(int)
        df_s = df_s.loc[df_s["Сценарий i"] != int(scen_del)].copy()
        st.session_state.df_scen = df_s
        df_g = st.session_state.df_grp.copy()
        df_g["Сценарий i"] = pd.to_numeric(df_g["Сценарий i"], errors="coerce").fillna(0).astype(int)
        df_g = df_g.loc[df_g["Сценарий i"] != int(scen_del)].copy()
        st.session_state.df_grp = df_g
        force_rerun()

with c3:
    st.info("Удаление сценария удаляет связанные группы.", icon="ℹ️")

# Расчётные столбцы для preview
df_scen_preview = st.session_state.df_scen.copy()
df_scen_preview["Сценарий i"] = pd.to_numeric(df_scen_preview["Сценарий i"], errors="coerce").fillna(0).astype(int)
df_scen_preview = df_scen_preview.loc[df_scen_preview["Сценарий i"] > 0].copy()

for col_bool, col_k in [
    ("ПС соответствует? (K_обн=0.8)", "K_обн (расч.)"),
    ("СОУЭ соответствует? (K_СОУЭ=0.8)", "K_СОУЭ (расч.)"),
    ("ПДЗ соответствует? (K_ПДЗ=0.8)", "K_ПДЗ (расч.)"),
]:
    if col_bool in df_scen_preview.columns:
        df_scen_preview[col_k] = df_scen_preview[col_bool].astype(bool).map(lambda x: K_STD if x else 0.0)

df_scen_edit = st.data_editor(
    df_scen_preview,
    num_rows="dynamic",
    use_container_width=True,
    disabled=["K_обн (расч.)", "K_СОУЭ (расч.)", "K_ПДЗ (расч.)"],
    column_config={
        "Сценарий i": st.column_config.NumberColumn(min_value=1, step=1),
        "Тип здания": st.column_config.SelectboxColumn(options=list(FIRE_FREQ_TABLE.keys())),
        "Q_п,i (год⁻¹)": st.column_config.NumberColumn(format="%.4e"),
        "t_пр,i (ч/сут)": st.column_config.NumberColumn(format="%.1f", min_value=0.0, max_value=24.0),
        "t_бл,i (мин)": st.column_config.NumberColumn(format="%.2f", min_value=0.1),
        "K_ап,i": st.column_config.NumberColumn(format="%.2f", min_value=0.0, max_value=0.9),
        "ПС соответствует? (K_обн=0.8)": st.column_config.CheckboxColumn("ПС (K_обн)"),
        "СОУЭ соответствует? (K_СОУЭ=0.8)": st.column_config.CheckboxColumn("СОУЭ (K_СОУЭ)"),
        "ПДЗ соответствует? (K_ПДЗ=0.8)": st.column_config.CheckboxColumn("ПДЗ (K_ПДЗ)"),
    },
    key="editor_scenarios"
)

drop_cols = ["K_обн (расч.)", "K_СОУЭ (расч.)", "K_ПДЗ (расч.)"]
df_scen_store = df_scen_edit.drop(columns=[c for c in drop_cols if c in df_scen_edit.columns], errors="ignore").copy()
df_scen_store["Сценарий i"] = pd.to_numeric(df_scen_store["Сценарий i"], errors="coerce").fillna(0).astype(int)
df_scen_store = df_scen_store.loc[df_scen_store["Сценарий i"] > 0].drop_duplicates(subset=["Сценарий i"], keep="first").copy()
st.session_state.df_scen = df_scen_store


# ─────────────────────────────────────────────────────
# ШАГ 2: ГРУППЫ ЭВАКУИРУЕМОГО КОНТИНГЕНТА
# ─────────────────────────────────────────────────────

st.subheader("📝 Шаг 2. Группы эвакуируемого контингента (п. 14, Приложение 2)")
st.caption(
    "Для каждого сценария задайте группы людей: "
    "t_р — расчётное время эвакуации (Приложения 5–8); "
    "t_н.э — время начала эвакуации (Приложение 4); "
    "t_ск — время существования скоплений (плотность > 0.5 м²/м²). "
    "Влияют на P_э по формуле (6)."
)

df_scen_for_groups = st.session_state.df_scen.copy()
df_scen_for_groups["Сценарий i"] = pd.to_numeric(df_scen_for_groups["Сценарий i"], errors="coerce").fillna(0).astype(int)
scen_list2 = sorted(df_scen_for_groups.loc[df_scen_for_groups["Сценарий i"] > 0, "Сценарий i"].unique().tolist())
if len(scen_list2) == 0:
    scen_list2 = [1]

df_grp_raw = st.session_state.df_grp.copy()
df_grp_raw = ensure_unique_positive_int_ids(df_grp_raw, "ID", start_from=1)
df_grp_raw["Сценарий i"] = pd.to_numeric(df_grp_raw["Сценарий i"], errors="coerce").fillna(scen_list2[0]).astype(int)
st.session_state.df_grp = df_grp_raw

g1, g2, g3 = st.columns([1.3, 1.7, 2.0])
with g1:
    scen_for_new_group = st.selectbox("Сценарий для новой группы", scen_list2, key="add_group_scen")
    if st.button("Добавить группу", use_container_width=True):
        df = st.session_state.df_grp.copy()
        df = ensure_unique_positive_int_ids(df, "ID", start_from=1)
        next_id = int(pd.to_numeric(df["ID"], errors="coerce").fillna(0).astype(int).max()) + 1 if len(df) else 1
        new_row = {
            "ID": next_id, "Сценарий i": int(scen_for_new_group),
            "Группа j": "Новая группа",
            "t_р,i,j (мин)": 6.0, "t_н.э,i,j (мин)": 1.5, "t_ск,i,j (мин)": 1.0,
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        st.session_state.df_grp = df
        st.session_state.selected_group_id = int(next_id)
        force_rerun()

with g2:
    df_tmp = st.session_state.df_grp.copy()
    df_tmp = ensure_unique_positive_int_ids(df_tmp, "ID", start_from=1)
    id_list2 = sorted(pd.to_numeric(df_tmp["ID"], errors="coerce").dropna().astype(int).unique().tolist())
    if not id_list2:
        id_list2 = [1]
    gid_del = st.selectbox("Удалить группу по ID", id_list2, key="del_group_id")
    if st.button("Удалить группу", use_container_width=True):
        df = st.session_state.df_grp.copy()
        df["ID"] = pd.to_numeric(df["ID"], errors="coerce").fillna(0).astype(int)
        df = df.loc[df["ID"] != int(gid_del)].copy()
        df = ensure_unique_positive_int_ids(df, "ID", start_from=1)
        st.session_state.df_grp = df
        if len(df) > 0:
            st.session_state.selected_group_id = int(df["ID"].iloc[0])
        force_rerun()

with g3:
    st.info("Группы привязаны к сценариям через поле «Сценарий i».", icon="ℹ️")

df_grp_raw2 = st.session_state.df_grp.copy()
df_grp_raw2 = ensure_unique_positive_int_ids(df_grp_raw2, "ID", start_from=1)

df_grp_edit = st.data_editor(
    df_grp_raw2,
    num_rows="dynamic",
    use_container_width=True,
    disabled=["ID"],
    column_config={
        "ID": st.column_config.NumberColumn(min_value=1, step=1),
        "Сценарий i": st.column_config.NumberColumn(min_value=1, step=1),
        "Группа j": st.column_config.TextColumn(),
        "t_р,i,j (мин)": st.column_config.NumberColumn(format="%.3f"),
        "t_н.э,i,j (мин)": st.column_config.NumberColumn(format="%.3f"),
        "t_ск,i,j (мин)": st.column_config.NumberColumn(format="%.3f"),
    },
    key="editor_groups"
)

df_grp_edit = ensure_unique_positive_int_ids(df_grp_edit, "ID", start_from=1)
df_grp_edit["Сценарий i"] = pd.to_numeric(df_grp_edit["Сценарий i"], errors="coerce").fillna(scen_list2[0]).astype(int)
st.session_state.df_grp = df_grp_edit


# ─────────────────────────────────────────────────────
# ШАГ 3: РАСЧЁТ
# ─────────────────────────────────────────────────────

df_scen_calc, df_rows_calc, df_agg, r_total = compute_all(
    st.session_state.df_scen,
    st.session_state.df_grp,
)


# ─────────────────────────────────────────────────────
# ШАГ 4: ПРОТИВОПОЖАРНЫЕ ДВЕРИ — формула (8), п. 48
# ─────────────────────────────────────────────────────

st.subheader("📝 Шаг 3. Учёт противопожарных дверей — формула (8), п. 48")

use_fire_doors = st.checkbox(
    "Учитывать противопожарные двери на путях эвакуации",
    value=st.session_state.use_fire_doors,
    key="cb_fire_doors",
    help="Если на путях эвакуации есть противопожарные двери, калитки в противопожарных воротах, "
         "расчёт выполняется для двух случаев: дверь открыта и дверь закрыта (п. 48)."
)
st.session_state.use_fire_doors = use_fire_doors

r_final = r_total

if use_fire_doors:
    st.caption(
        "По п. 48 Методики: R_i = P_откр · R_i(откр) + P_закр · R_i(закр), "
        "где P_откр = 0.3, P_закр = 0.7"
    )

    dc1, dc2 = st.columns(2)
    with dc1:
        r_open_input = st.number_input(
            "R_i (дверь открыта), год⁻¹",
            min_value=0.0,
            value=float(r_total),
            format="%.2e",
            key="r_door_open_input",
            help="Значение R при открытой противопожарной двери. "
                 "По умолчанию равно текущему расчётному R (без учёта двери)."
        )
    with dc2:
        r_closed_input = st.number_input(
            "R_i (дверь закрыта), год⁻¹",
            min_value=0.0,
            value=0.0,
            format="%.2e",
            key="r_door_closed_input",
            help="Значение R при закрытой противопожарной двери. "
                 "ОФП через закрытую дверь не распространяются, поэтому может быть значительно ниже."
        )

    r_final = r_i_with_doors(r_open_input, r_closed_input)
    st.metric(
        "R (с учётом дверей) по формуле (8)",
        f"{r_final:.2e} год⁻¹",
        help=f"= {P_DOOR_OPEN} × {r_open_input:.2e} + {P_DOOR_CLOSED} × {r_closed_input:.2e}"
    )


# ─────────────────────────────────────────────────────
# РЕЗУЛЬТАТЫ
# ─────────────────────────────────────────────────────

st.subheader("📊 Шаг 4. Результаты расчёта — формулы (1)–(2)")

passed = r_final <= R_NORM

m1, m2, m3 = st.columns(3)
with m1:
    st.metric("R (расчётная величина ИПР), год⁻¹", f"{r_final:.6g}")
with m2:
    st.metric("R_норм (нормативное значение), год⁻¹", f"{R_NORM:.1e}")
with m3:
    if passed:
        st.success("R <= R_норм: Пожарный риск ДОПУСТИМЫЙ (формула 1)")
    else:
        st.error("R > R_норм: Пожарный риск ПРЕВЫШЕН (формула 1)")

# Визуальная шкала
components.html(risk_gauge_html(r_final, R_NORM), height=420, scrolling=False)


# ─────────────────────────────────────────────────────
# ДЕТАЛИЗАЦИЯ РАСЧЁТА
# ─────────────────────────────────────────────────────

st.subheader("📋 Детализация: агрегирование по сценариям — формулы (2)–(3)")
st.caption("R_i = max_j{R_i,j} — формула (3); R = max_i{R_i} — формула (2)")

df_agg_view = format_df_scientific(df_agg, sci_cols=["R_i = max_j(R_i,j)"], digits=2)
st.dataframe(df_agg_view, use_container_width=True)


st.subheader("📋 Детализация: ИПР по группам — формула (4)")

cols_show = [
    "ID", "Сценарий i", "Группа j",
    "Q_п,i (год⁻¹)", "K_ап,i",
    "t_бл,i (мин)", "t_р,i,j (мин)", "t_н.э,i,j (мин)", "t_ск,i,j (мин)",
    "P_пр,i", "P_э,i,j", "K_п.з,i", "R_i,j"
]
cols_show = [c for c in cols_show if c in df_rows_calc.columns]
df_rows_view = df_rows_calc[cols_show].copy()
df_rows_view = format_df_scientific(df_rows_view, sci_cols=["R_i,j", "Q_п,i (год⁻¹)"], digits=2)
st.dataframe(df_rows_view, use_container_width=True)


st.subheader("📋 Детализация: коэффициенты по сценариям — формула (7)")

cols_scen = [
    "Сценарий i", "Тип здания", "Q_п,i (год⁻¹)", "K_ап,i",
    "K_обн,i", "K_СОУЭ,i", "K_ПДЗ,i", "K_п.з,i", "P_пр,i",
]
cols_scen = [c for c in cols_scen if c in df_scen_calc.columns]
df_scen_view = df_scen_calc[cols_scen].copy()
df_scen_view = format_df_scientific(df_scen_view, sci_cols=["Q_п,i (год⁻¹)"], digits=2)
st.dataframe(df_scen_view, use_container_width=True)


# ─────────────────────────────────────────────────────
# ДИАГНОСТИКА P_э
# ─────────────────────────────────────────────────────

st.subheader("🔍 Диагностика P_э,i,j — формула (6)")

sel_id = int(st.session_state.selected_group_id)
if len(df_rows_calc) > 0 and "ID" in df_rows_calc.columns:
    ids = df_rows_calc["ID"].astype(int).to_list()
    default_index = ids.index(sel_id) if sel_id in ids else 0
    sel_diag_id = st.selectbox("ID строки для диагностики P_э", ids, index=default_index)

    row = df_rows_calc.loc[df_rows_calc["ID"].astype(int) == int(sel_diag_id)].iloc[0]
    t_bl = safe_float(row.get("t_бл,i (мин)", 0.0))
    t_p = safe_float(row.get("t_р,i,j (мин)", 0.0))
    t_ne = safe_float(row.get("t_н.э,i,j (мин)", 0.0))
    t_ck = safe_float(row.get("t_ск,i,j (мин)", 0.0))
    border = 0.8 * t_bl

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("0.8·t_бл (мин)", f"{border:.3f}")
    c2.metric("t_р (мин)", f"{t_p:.3f}")
    c3.metric("t_р + t_н.э (мин)", f"{(t_p + t_ne):.3f}")
    c4.metric("t_ск (мин)", f"{t_ck:.3f}")

    st.metric("P_э,i,j", f"{safe_float(row.get('P_э,i,j', 0.0)):.4f}")

    if t_ck > 6:
        st.warning("t_ск > 6 мин → по формуле (6): P_э = 0")
    elif (t_p + t_ne) <= border:
        st.success("t_р + t_н.э <= 0.8·t_бл и t_ск <= 6 → P_э = 0.999")
    elif (t_p < border) and (border < (t_p + t_ne)):
        st.info("0.8·t_бл попадает между t_р и t_р + t_н.э → промежуточное значение (ветвь 1 формулы 6)")
    else:
        st.warning("t_р >= 0.8·t_бл → по формуле (6): P_э = 0")
else:
    st.info("Нет строк групп для диагностики.")


# ─────────────────────────────────────────────────────
# ФОРМУЛЫ — разворачиваемый блок
# ─────────────────────────────────────────────────────

st.subheader("📐 Формулы Методики №1140")

with st.expander("Показать формулы (1)–(8)", expanded=False):
    st.markdown("**Условие соответствия — формула (1), п. 8:**")
    st.latex(r"R \le R_{\text{норм}}, \quad R_{\text{норм}} = 10^{-6}\ \text{год}^{-1}")

    st.markdown("**Формула (2) — расчётная величина ИПР (п. 9):**")
    st.latex(r"R = \max\{R_1, \ldots, R_i, \ldots, R_K\}")

    st.markdown("**Формула (3) — ИПР по сценарию (п. 14):**")
    st.latex(r"R_i = \max\{R_{i,1}, \ldots, R_{i,j}, \ldots, R_{i,m}\}")

    st.markdown("**Формула (4) — ИПР для группы (п. 15):**")
    st.latex(r"R_{i,j} = Q_{\text{п},i} \cdot (1 - K_{\text{ап},i}) \cdot P_{\text{пр},i} \cdot (1 - P_{\text{э},i,j}) \cdot (1 - K_{\text{п.з},i})")

    st.markdown("**Формула (5) — вероятность присутствия (п. 16):**")
    st.latex(r"P_{\text{пр},i} = \frac{t_{\text{пр},i}}{24}")

    st.markdown("**Формула (6) — вероятность эвакуации (п. 17):**")
    st.latex(r"""
P_{\text{э},i,j} =
\begin{cases}
0{,}999 \cdot \dfrac{0{,}8 \, t_{\text{бл},i} - t_{\text{р},i,j}}{t_{\text{н.э},i,j}},
  & t_{\text{р},i,j} < 0{,}8 \, t_{\text{бл},i} < t_{\text{р},i,j} + t_{\text{н.э},i,j},\ t_{\text{ск}} \le 6 \\[6pt]
0{,}999,
  & t_{\text{р},i,j} + t_{\text{н.э},i,j} \le 0{,}8 \, t_{\text{бл},i},\ t_{\text{ск}} \le 6 \\[6pt]
0,
  & t_{\text{р},i,j} \ge 0{,}8 \, t_{\text{бл},i}\ \text{или}\ t_{\text{ск}} > 6
\end{cases}
""")

    st.markdown("**Формула (7) — коэффициент противопожарной защиты (п. 21):**")
    st.latex(r"K_{\text{п.з},i} = 1 - (1 - K_{\text{обн},i} \cdot K_{\text{СОУЭ},i}) \cdot (1 - K_{\text{обн},i} \cdot K_{\text{ПДЗ},i})")

    st.markdown("**Формула (8) — учёт противопожарных дверей (п. 48):**")
    st.latex(r"R_i = P_{\text{откр}} \cdot R_i^{\text{откр}} + P_{\text{закр}} \cdot R_i^{\text{закр}}, \quad P_{\text{откр}} = 0{,}3,\ P_{\text{закр}} = 0{,}7")

    st.markdown("**Формула (П4.1) — время начала эвакуации для помещения очага (Приложение 4):**")
    st.latex(r"t_{\text{н.э}} = \frac{5 + 0{,}01 \cdot F_{\text{пом}}}{60}\ \text{мин}")


# ─────────────────────────────────────────────────────
# СПРАВОЧНЫЕ ТАБЛИЦЫ — разворачиваемый блок
# ─────────────────────────────────────────────────────

st.subheader("📚 Справочные таблицы из Приложений к Методике")

with st.expander("Приложение 3 — Частота возникновения пожара (Таблица П3.1)", expanded=False):
    df_freq = pd.DataFrame([
        {"№": i + 1, "Наименование здания": k, "Q_п (год⁻¹)": v}
        for i, (k, v) in enumerate(FIRE_FREQ_TABLE.items()) if k != "Иное (Q_п = 4·10⁻²)"
    ])
    df_freq = format_df_scientific(df_freq, sci_cols=["Q_п (год⁻¹)"], digits=2)
    st.dataframe(df_freq, use_container_width=True, hide_index=True)

with st.expander("Приложение 4 — Время начала эвакуации (Таблица П4.1)", expanded=False):
    df_tne = pd.DataFrame([
        {"№": i + 1, "Класс и характеристика": k, "t_н.э (мин)": v}
        for i, (k, v) in enumerate(T_NE_TABLE.items())
    ])
    st.dataframe(df_tne, use_container_width=True, hide_index=True)

with st.expander("Приложение 9 — Параметры пожарной нагрузки (Таблица П9.1)", expanded=False):
    st.dataframe(FIRE_LOAD_TABLE, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────
# ВЫГРУЗКА
# ─────────────────────────────────────────────────────

st.subheader("💾 Выгрузка результатов")

csv_rows = df_rows_calc.to_csv(index=False).encode("utf-8-sig")
csv_scen = df_scen_calc.to_csv(index=False).encode("utf-8-sig")
csv_agg = df_agg.to_csv(index=False).encode("utf-8-sig")

c1, c2, c3 = st.columns(3)
with c1:
    st.download_button("CSV: группы (построчно)", data=csv_rows, file_name="ipr_groups.csv", mime="text/csv")
with c2:
    st.download_button("CSV: сценарии (коэффициенты)", data=csv_scen, file_name="ipr_scenarios.csv", mime="text/csv")
with c3:
    st.download_button("CSV: агрегирование (итог)", data=csv_agg, file_name="ipr_aggregate.csv", mime="text/csv")
