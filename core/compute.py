"""
Основная логика расчёта ИПР по формулам (2)–(7) Методики №1140.
"""

from typing import Tuple

import numpy as np
import pandas as pd

from core.constants import K_STD, R_NORM
from core.formulas import p_presence, k_pz, p_evac, r_ij
from utils.helpers import safe_float


def compute_all(
    df_scen_in: pd.DataFrame,
    df_grp_in: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    """
    Полный расчёт ИПР по формулам (2)–(7) Методики №1140.

    Возвращает
    ----------
    df_scen : DataFrame со сценариями и расчётными коэффициентами
    df_rows : DataFrame с результатами по группам
    df_agg  : DataFrame с агрегированием по сценариям + итоговая строка
    r_total : расчётная величина ИПР (формула 2)
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

    # K_обн, K_СОУЭ, K_ПДЗ - 0 или 0.8 строго по №1140
    for col_bool, col_k in [
        ("ПС соответствует? (K_обн=0.8)", "K_обн,i"),
        ("СОУЭ соответствует? (K_СОУЭ=0.8)", "K_СОУЭ,i"),
        ("ПДЗ соответствует? (K_ПДЗ=0.8)", "K_ПДЗ,i"),
    ]:
        if col_bool in df_scen.columns:
            df_scen[col_k] = np.where(df_scen[col_bool].astype(bool), K_STD, 0.0)
        else:
            df_scen[col_k] = 0.0

    # P_пр - формула (5)
    if "t_пр,i (ч/сут)" in df_scen.columns:
        df_scen["P_пр,i"] = df_scen["t_пр,i (ч/сут)"].apply(p_presence)
    else:
        df_scen["P_пр,i"] = 0.5

    # K_п.з - формула (7)
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

    # P_э - формула (6)
    df_rows["P_э,i,j"] = df_rows.apply(
        lambda r: p_evac(
            r["t_р,i,j (мин)"], r["t_бл,i (мин)"], r["t_н.э,i,j (мин)"], r["t_ск,i,j (мин)"]
        ),
        axis=1
    )

    # R_i,j - формула (4)
    df_rows["R_i,j"] = df_rows.apply(
        lambda r: r_ij(
            r["Q_п,i (год⁻¹)"], r["K_ап,i"], r["P_пр,i"], r["P_э,i,j"], r["K_п.з,i"]
        ),
        axis=1
    )

    # R_i = max_j {R_i,j} - формула (3)
    agg = df_rows.groupby("Сценарий i", as_index=False).agg(
        **{"R_i = max_j(R_i,j)": ("R_i,j", "max")}
    )

    # R = max_i {R_i} - формула (2)
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
