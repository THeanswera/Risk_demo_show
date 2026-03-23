"""
Unit-тесты формул Методики №1140 (формулы 4–7, П4.1).
Запуск: pytest tests/test_formulas.py
"""

import sys
import os

# Добавляем корень проекта в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from core.formulas import k_pz, p_evac, p_presence, r_ij, t_ne_formula
from utils.helpers import clamp, safe_float


# ─────────────────────────────────────────────────────
# Формула (5): P_пр
# ─────────────────────────────────────────────────────

def test_p_presence_half_day():
    assert p_presence(12.0) == pytest.approx(0.5)


def test_p_presence_full_day():
    assert p_presence(24.0) == pytest.approx(1.0)


def test_p_presence_zero():
    assert p_presence(0.0) == pytest.approx(0.0)


def test_p_presence_clamp_over_24():
    """Значения > 24 часов зажимаются к 1.0."""
    assert p_presence(30.0) == pytest.approx(1.0)


def test_p_presence_8_hours():
    assert p_presence(8.0) == pytest.approx(8.0 / 24.0)


# ─────────────────────────────────────────────────────
# Формула (6): P_э — три ветви по п. 17 Методики №1140
# ИСПРАВЛЕНО: формула изменена на 1 − (tр + tнэ) / tбл
# ─────────────────────────────────────────────────────

def test_p_evac_branch1_full():
    """tр + tн.э < 0.8·tбл и tск ≤ 6 → 0.999."""
    # 2 + 1 = 3 < 0.8 * 10 = 8
    assert p_evac(t_p=2.0, t_bl=10.0, t_ne=1.0, t_ck=1.0) == pytest.approx(0.999)


def test_p_evac_branch1_exactly_border():
    """tр + tн.э = 0.8·tбл → НЕ попадает в ветвь 1 (строгое неравенство)."""
    # 2 + 6 = 8 = 0.8 * 10 → промежуточная ветвь: 1 - 8/10 = 0.2
    result = p_evac(t_p=2.0, t_bl=10.0, t_ne=6.0, t_ck=1.0)
    assert result == pytest.approx(0.2)


def test_p_evac_branch2_intermediate():
    """0.8·tбл ≤ tр + tнэ < tбл → Pэ = 1 − (tр + tнэ) / tбл."""
    # 7 + 2 = 9, 0.8*10=8, 8 ≤ 9 < 10 → 1 - 9/10 = 0.1
    result = p_evac(t_p=7.0, t_bl=10.0, t_ne=2.0, t_ck=1.0)
    assert result == pytest.approx(0.1)


def test_p_evac_branch2_value():
    """Проверка конкретного промежуточного значения."""
    # 5 + 4 = 9, 0.8*10=8, 8 ≤ 9 < 10 → 1 - 9/10 = 0.1
    result = p_evac(t_p=5.0, t_bl=10.0, t_ne=4.0, t_ck=1.0)
    assert result == pytest.approx(0.1)


def test_p_evac_branch3_zero():
    """tр + tнэ ≥ tбл → Pэ = 0."""
    # 8 + 3 = 11 >= 10
    assert p_evac(t_p=8.0, t_bl=10.0, t_ne=3.0, t_ck=1.0) == pytest.approx(0.0)


def test_p_evac_branch3_exact():
    """tр + tнэ = tбл → Pэ = 0."""
    # 7 + 3 = 10 = 10
    assert p_evac(t_p=7.0, t_bl=10.0, t_ne=3.0, t_ck=1.0) == pytest.approx(0.0)


def test_p_evac_zero_crowd():
    """tск > 6 → 0 (независимо от остальных параметров)."""
    assert p_evac(t_p=1.0, t_bl=100.0, t_ne=1.0, t_ck=7.0) == pytest.approx(0.0)


def test_p_evac_crowd_exactly_6():
    """tск = 6 — граница, эвакуация возможна."""
    assert p_evac(t_p=1.0, t_bl=100.0, t_ne=1.0, t_ck=6.0) == pytest.approx(0.999)


def test_p_evac_zero_t_bl():
    """tбл = 0 → эвакуация невозможна (граничный случай)."""
    assert p_evac(t_p=1.0, t_bl=0.0, t_ne=1.0, t_ck=1.0) == pytest.approx(0.0)


def test_p_evac_max_cap():
    """Pэ ≤ 0.999 всегда (п. 17 Методики №1140)."""
    result = p_evac(t_p=0.01, t_bl=100.0, t_ne=0.01, t_ck=0.0)
    assert result <= 0.999


# ─────────────────────────────────────────────────────
# Формула (7): K_п.з
# ─────────────────────────────────────────────────────

def test_k_pz_all_systems():
    """Все системы с K=0.8 → стандартный расчёт."""
    result = k_pz(0.8, 0.8, 0.8)
    expected = 1.0 - (1.0 - 0.8 * 0.8) * (1.0 - 0.8 * 0.8)
    assert result == pytest.approx(expected)


def test_k_pz_all_std():
    """Результат для K=0.8 равен 1 - (1-0.64)^2 = 1 - 0.1296 = 0.8704."""
    assert k_pz(0.8, 0.8, 0.8) == pytest.approx(0.8704)


def test_k_pz_zeros():
    """Все нули → K_п.з = 0."""
    assert k_pz(0, 0, 0) == pytest.approx(0.0)


def test_k_pz_no_ps():
    """ПС отсутствует: K_обн = 0."""
    result = k_pz(0.0, 0.8, 0.8)
    assert result == pytest.approx(0.0)


def test_k_pz_binary_values():
    """K_обн/K_СОУЭ/K_ПДЗ строго 0 или 0.8 (п. 41, 44, 45 Методики №1140)."""
    # Все возможные комбинации бинарных значений
    for k1 in [0.0, 0.8]:
        for k2 in [0.0, 0.8]:
            for k3 in [0.0, 0.8]:
                result = k_pz(k1, k2, k3)
                assert 0.0 <= result <= 1.0


# ─────────────────────────────────────────────────────
# Формула (4): R_i,j
# ─────────────────────────────────────────────────────

def test_r_ij_with_full_evac():
    """Полная эвакуация (Pэ=0.999) → очень малый риск."""
    result = r_ij(q_p=4e-2, k_ap=0.9, p_pr=0.5, p_e=0.999, k_pz_val=0.87)
    assert result < 1e-6


def test_r_ij_no_protection():
    """Нет систем защиты (Pэ=0, K_п.з=0, K_ап=0) → R = Q_п × P_пр."""
    result = r_ij(q_p=4e-2, k_ap=0.0, p_pr=0.5, p_e=0.0, k_pz_val=0.0)
    assert result == pytest.approx(4e-2 * 1.0 * 0.5 * 1.0 * 1.0)


def test_r_ij_nonnegative():
    """R_i,j всегда неотрицательный."""
    assert r_ij(q_p=0.0, k_ap=0.9, p_pr=0.5, p_e=0.5, k_pz_val=0.5) == pytest.approx(0.0)


def test_r_ij_default_scenario():
    """Базовый сценарий по умолчанию: воспроизводимый результат."""
    # Q_п = 4e-2, K_ап = 0.9, P_пр = 0.5 (12ч/24ч), P_э = 0.999, K_п.з = 0.8704
    result = r_ij(q_p=4e-2, k_ap=0.9, p_pr=0.5, p_e=0.999, k_pz_val=0.8704)
    expected = 4e-2 * (1 - 0.9) * 0.5 * (1 - 0.999) * (1 - 0.8704)
    assert result == pytest.approx(expected, rel=1e-4)


def test_r_ij_k_ap_binary():
    """K_ап строго 0 или 0.9 (п. 15 Методики №1140)."""
    r_with_aup = r_ij(q_p=4e-2, k_ap=0.9, p_pr=1.0, p_e=0.0, k_pz_val=0.0)
    r_without_aup = r_ij(q_p=4e-2, k_ap=0.0, p_pr=1.0, p_e=0.0, k_pz_val=0.0)
    assert r_with_aup == pytest.approx(4e-2 * 0.1)
    assert r_without_aup == pytest.approx(4e-2 * 1.0)


# ─────────────────────────────────────────────────────
# Формула t_нэ (Приложение 4, П4.1)
# ─────────────────────────────────────────────────────

def test_t_ne_formula_100m2():
    """F = 100 м² → t = (5 + 0.01×100)/60 = 6/60 = 0.1 мин."""
    assert t_ne_formula(100.0) == pytest.approx(6.0 / 60.0)


def test_t_ne_formula_zero():
    assert t_ne_formula(0.0) == pytest.approx(5.0 / 60.0)


def test_t_ne_formula_units():
    """Результат t_ne_formula — минуты (секунды / 60)."""
    f_pom = 500.0
    result = t_ne_formula(f_pom)
    expected_sec = 5.0 + 0.01 * f_pom  # = 10 секунд
    expected_min = expected_sec / 60.0   # = 0.1667 минут
    assert result == pytest.approx(expected_min)


# ─────────────────────────────────────────────────────
# Агрегирование: max, не sum (формулы 2, 3)
# ─────────────────────────────────────────────────────

def test_aggregation_uses_max():
    """Проверка, что compute_all использует max (формулы 2 и 3)."""
    import pandas as pd
    from core.compute import compute_all

    # Оба сценария без защиты, чтобы R1 и R2 были сопоставимы
    scen = pd.DataFrame([
        {
            "Сценарий i": 1, "Тип здания": "Иное (Qп = 4·10⁻²)",
            "Q_п,i (год⁻¹)": 4.0e-2, "t_пр,i (ч/сут)": 24.0,
            "t_бл,i (мин)": 5.0, "АУП соответствует? (K_ап=0.9)": False,
            "ПС соответствует? (K_обн=0.8)": False,
            "СОУЭ соответствует? (K_СОУЭ=0.8)": False,
            "ПДЗ соответствует? (K_ПДЗ=0.8)": False,
        },
        {
            "Сценарий i": 2, "Тип здания": "Иное (Qп = 4·10⁻²)",
            "Q_п,i (год⁻¹)": 2.0e-2, "t_пр,i (ч/сут)": 24.0,
            "t_бл,i (мин)": 5.0, "АУП соответствует? (K_ап=0.9)": False,
            "ПС соответствует? (K_обн=0.8)": False,
            "СОУЭ соответствует? (K_СОУЭ=0.8)": False,
            "ПДЗ соответствует? (K_ПДЗ=0.8)": False,
        },
    ])
    grp = pd.DataFrame([
        {"ID": 1, "Сценарий i": 1, "Группа j": "A",
         "t_р,i,j (мин)": 8.0, "t_н.э,i,j (мин)": 3.0, "t_ск,i,j (мин)": 1.0},
        {"ID": 2, "Сценарий i": 2, "Группа j": "B",
         "t_р,i,j (мин)": 8.0, "t_н.э,i,j (мин)": 3.0, "t_ск,i,j (мин)": 1.0},
    ])
    _, df_rows, _, r_total = compute_all(scen, grp)
    r1 = df_rows.loc[df_rows["Сценарий i"] == 1, "R_i,j"].max()
    r2 = df_rows.loc[df_rows["Сценарий i"] == 2, "R_i,j"].max()
    # R = max(R1, R2), не sum
    assert r_total == pytest.approx(max(r1, r2))
    # При двух ненулевых значениях sum > max
    assert r1 > 0 and r2 > 0
    assert r_total < r1 + r2


# ─────────────────────────────────────────────────────
# Вспомогательные функции
# ─────────────────────────────────────────────────────

def test_safe_float_nan():
    import math
    assert safe_float(float("nan")) == pytest.approx(0.0)


def test_safe_float_inf():
    assert safe_float(float("inf")) == pytest.approx(0.0)


def test_safe_float_string():
    assert safe_float("3.14") == pytest.approx(3.14)


def test_safe_float_bad_string():
    assert safe_float("abc") == pytest.approx(0.0)


def test_clamp_hi():
    assert clamp(1.5, 0.0, 1.0) == pytest.approx(1.0)


def test_clamp_lo():
    assert clamp(-0.5, 0.0, 1.0) == pytest.approx(0.0)


def test_clamp_mid():
    assert clamp(0.5, 0.0, 1.0) == pytest.approx(0.5)
