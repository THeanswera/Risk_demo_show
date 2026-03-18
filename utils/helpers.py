"""
Вспомогательные функции общего назначения.
"""

import math

import numpy as np
import pandas as pd
import streamlit as st


def clamp(x: float, lo: float, hi: float) -> float:
    """Ограничить значение x диапазоном [lo, hi]."""
    return float(max(lo, min(hi, x)))


def safe_float(x, default: float = 0.0) -> float:
    """Безопасное преобразование в float; NaN/Inf заменяются на default."""
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return float(default)
        return v
    except Exception:
        return float(default)


def force_rerun() -> None:
    """Принудительный перезапуск страницы Streamlit."""
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()


def ensure_unique_positive_int_ids(
    df: pd.DataFrame, col: str, start_from: int = 1
) -> pd.DataFrame:
    """Гарантировать уникальные положительные целые идентификаторы в столбце col."""
    df = df.copy()
    if col not in df.columns:
        df.insert(0, col, np.arange(start_from, start_from + len(df), dtype=int))
        return df
    ids = pd.to_numeric(df[col], errors="coerce")
    used: set[int] = set()
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
    """Форматировать число в научную нотацию."""
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return ""
        return f"{v:.{digits}e}"
    except Exception:
        return str(x)


def format_df_scientific(
    df: pd.DataFrame, sci_cols: list, digits: int = 2
) -> pd.DataFrame:
    """Применить научную нотацию к указанным столбцам DataFrame."""
    out = df.copy()
    for c in sci_cols:
        if c in out.columns:
            out[c] = out[c].apply(lambda v: fmt_sci(v, digits=digits) if pd.notna(v) else "")
    return out


# Словарь красивых названий столбцов (внутреннее имя → отображаемое)
PRETTY: dict[str, str] = {
    "Сценарий i": "Сценарий i",
    "Тип здания": "Тип здания",
    "Q_п,i (год⁻¹)": "Qп,ᵢ (год⁻¹)",
    "t_пр,i (ч/сут)": "tпр,ᵢ (ч/сут)",
    "t_бл,i (мин)": "tбл,ᵢ (мин)",
    "K_ап,i": "Kап,ᵢ",
    "ПС соответствует? (K_обн=0.8)": "ПС (Kобн)",
    "СОУЭ соответствует? (K_СОУЭ=0.8)": "СОУЭ (Kсоуэ)",
    "ПДЗ соответствует? (K_ПДЗ=0.8)": "ПДЗ (Kпдз)",
    "K_обн (расч.)": "Kобн (расч.)",
    "K_СОУЭ (расч.)": "Kсоуэ (расч.)",
    "K_ПДЗ (расч.)": "Kпдз (расч.)",
    "K_обн,i": "Kобн,ᵢ",
    "K_СОУЭ,i": "Kсоуэ,ᵢ",
    "K_ПДЗ,i": "Kпдз,ᵢ",
    "K_п.з,i": "Kп.з,ᵢ",
    "P_пр,i": "Pпр,ᵢ",
    "P_э,i,j": "Pэ,ᵢ,ⱼ",
    "t_р,i,j (мин)": "tр,ᵢ,ⱼ (мин)",
    "t_н.э,i,j (мин)": "tн.э,ᵢ,ⱼ (мин)",
    "t_ск,i,j (мин)": "tск,ᵢ,ⱼ (мин)",
    "R_i,j": "Rᵢ,ⱼ",
    "R_i = max_j(R_i,j)": "Rᵢ = maxⱼ(Rᵢ,ⱼ)",
    "R <= R_норм?": "R ≤ Rнорм?",
    "Группа j": "Группа j",
}


def prettify_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Переименовать столбцы DataFrame для красивого отображения."""
    return df.rename(columns={k: v for k, v in PRETTY.items() if k in df.columns})
