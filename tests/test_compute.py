"""
Интеграционные тесты compute_all() — Методика №1140.
Запуск: pytest tests/test_compute.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest

from core.compute import compute_all
from core.constants import R_NORM


# ─────────────────────────────────────────────────────
# Фикстуры
# ─────────────────────────────────────────────────────

@pytest.fixture()
def default_scenarios():
    return pd.DataFrame([{
        "Сценарий i": 1,
        "Тип здания": "Иное (Q_п = 4·10⁻²)",
        "Qп,i (год⁻¹)": 4.0e-2,
        "tпр,i (ч/сут)": 12.0,
        "tбл,i (мин)": 12.0,
        "Kап,i": 0.9,
        "ПС соответствует? (Kобн=0.8)": True,
        "СОУЭ соответствует? (KСОУЭ=0.8)": True,
        "ПДЗ соответствует? (KПДЗ=0.8)": True,
    }])


@pytest.fixture()
def default_groups():
    return pd.DataFrame([
        {
            "ID": 1, "Сценарий i": 1, "Группа j": "Основной контингент",
            "t_р,i,j (мин)": 6.0, "t_н.э,i,j (мин)": 1.5, "t_ск,i,j (мин)": 1.0,
        },
        {
            "ID": 2, "Сценарий i": 1, "Группа j": "Маломобильные",
            "t_р,i,j (мин)": 7.0, "t_н.э,i,j (мин)": 2.0, "t_ск,i,j (мин)": 2.5,
        },
    ])


# ─────────────────────────────────────────────────────
# Тесты базового сценария
# ─────────────────────────────────────────────────────

def test_compute_all_returns_tuple(default_scenarios, default_groups):
    result = compute_all(default_scenarios, default_groups)
    assert len(result) == 4


def test_compute_all_r_total_nonnegative(default_scenarios, default_groups):
    _, _, _, r_total = compute_all(default_scenarios, default_groups)
    assert r_total >= 0.0


def test_compute_all_r_total_reasonable(default_scenarios, default_groups):
    """R должен быть в разумном диапазоне (0, 1]."""
    _, _, _, r_total = compute_all(default_scenarios, default_groups)
    assert r_total <= 1.0


def test_compute_all_rows_not_empty(default_scenarios, default_groups):
    _, df_rows, _, _ = compute_all(default_scenarios, default_groups)
    assert len(df_rows) > 0


def test_compute_all_agg_contains_total(default_scenarios, default_groups):
    _, _, df_agg, _ = compute_all(default_scenarios, default_groups)
    assert "ИТОГО (R = max)" in df_agg["Сценарий i"].astype(str).values


def test_compute_all_agg_r_total_match(default_scenarios, default_groups):
    """R в итоговой строке совпадает с r_total."""
    _, _, df_agg, r_total = compute_all(default_scenarios, default_groups)
    total_row = df_agg[df_agg["Сценарий i"].astype(str) == "ИТОГО (R = max)"]
    assert float(total_row["R_i = max_j(R_i,j)"].iloc[0]) == pytest.approx(r_total)


def test_compute_all_default_is_acceptable(default_scenarios, default_groups):
    """Базовый сценарий по умолчанию должен давать допустимый риск."""
    _, _, _, r_total = compute_all(default_scenarios, default_groups)
    assert r_total <= R_NORM


def test_compute_all_p_evac_in_rows(default_scenarios, default_groups):
    """В результатах должна быть колонка P_э,i,j."""
    _, df_rows, _, _ = compute_all(default_scenarios, default_groups)
    assert "P_э,i,j" in df_rows.columns


def test_compute_all_rij_in_rows(default_scenarios, default_groups):
    """В результатах должна быть колонка R_i,j."""
    _, df_rows, _, _ = compute_all(default_scenarios, default_groups)
    assert "R_i,j" in df_rows.columns


def test_compute_all_k_pz_in_scen(default_scenarios, default_groups):
    """В обработанных сценариях должна быть колонка K_п.з,i."""
    df_scen, _, _, _ = compute_all(default_scenarios, default_groups)
    assert "K_п.з,i" in df_scen.columns


# ─────────────────────────────────────────────────────
# Тест: превышение риска при плохих параметрах
# ─────────────────────────────────────────────────────

def test_compute_all_high_risk_scenario():
    """Нет систем защиты + медленная эвакуация → риск > нормы."""
    scen = pd.DataFrame([{
        "Сценарий i": 1,
        "Тип здания": "Стоянки автомобилей",
        "Q_п,i (год⁻¹)": 4.5e-2,
        "t_пр,i (ч/сут)": 24.0,
        "t_бл,i (мин)": 5.0,
        "K_ап,i": 0.0,
        "ПС соответствует? (K_обн=0.8)": False,
        "СОУЭ соответствует? (K_СОУЭ=0.8)": False,
        "ПДЗ соответствует? (K_ПДЗ=0.8)": False,
    }])
    grp = pd.DataFrame([{
        "ID": 1, "Сценарий i": 1, "Группа j": "Персонал",
        "t_р,i,j (мин)": 10.0, "t_н.э,i,j (мин)": 5.0, "t_ск,i,j (мин)": 8.0,
    }])
    _, _, _, r_total = compute_all(scen, grp)
    assert r_total > R_NORM


# ─────────────────────────────────────────────────────
# Тест: несколько сценариев
# ─────────────────────────────────────────────────────

def test_compute_all_multi_scenario():
    """R = max по всем сценариям."""
    scen = pd.DataFrame([
        {
            "Сценарий i": 1, "Тип здания": "Иное (Q_п = 4·10⁻²)",
            "Q_п,i (год⁻¹)": 4.0e-2, "t_пр,i (ч/сут)": 12.0,
            "t_бл,i (мин)": 12.0, "K_ап,i": 0.9,
            "ПС соответствует? (K_обн=0.8)": True,
            "СОУЭ соответствует? (K_СОУЭ=0.8)": True,
            "ПДЗ соответствует? (K_ПДЗ=0.8)": True,
        },
        {
            "Сценарий i": 2, "Тип здания": "Больницы",
            "Q_п,i (год⁻¹)": 1.3e-2, "t_пр,i (ч/сут)": 24.0,
            "t_бл,i (мин)": 8.0, "K_ап,i": 0.9,
            "ПС соответствует? (K_обн=0.8)": True,
            "СОУЭ соответствует? (K_СОУЭ=0.8)": True,
            "ПДЗ соответствует? (K_ПДЗ=0.8)": False,
        },
    ])
    grp = pd.DataFrame([
        {
            "ID": 1, "Сценарий i": 1, "Группа j": "Группа A",
            "t_р,i,j (мин)": 6.0, "t_н.э,i,j (мин)": 1.5, "t_ск,i,j (мин)": 1.0,
        },
        {
            "ID": 2, "Сценарий i": 2, "Группа j": "Пациенты",
            "t_р,i,j (мин)": 9.0, "t_н.э,i,j (мин)": 6.0, "t_ск,i,j (мин)": 2.0,
        },
    ])
    _, df_rows, df_agg, r_total = compute_all(scen, grp)
    # R = max(R1, R2)
    r1 = df_rows.loc[df_rows["Сценарий i"] == 1, "R_i,j"].max()
    r2 = df_rows.loc[df_rows["Сценарий i"] == 2, "R_i,j"].max()
    assert r_total == pytest.approx(max(r1, r2))


# ─────────────────────────────────────────────────────
# Тест: пустые данные
# ─────────────────────────────────────────────────────

def test_compute_all_empty_groups(default_scenarios):
    """Пустые группы → r_total = 0."""
    empty_grp = pd.DataFrame(columns=[
        "ID", "Сценарий i", "Группа j",
        "t_р,i,j (мин)", "t_н.э,i,j (мин)", "t_ск,i,j (мин)",
    ])
    _, _, _, r_total = compute_all(default_scenarios, empty_grp)
    assert r_total == pytest.approx(0.0)
