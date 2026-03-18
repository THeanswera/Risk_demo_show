"""
Unit-тесты формул Методики №1140 (формулы 4–8).
Запуск: pytest tests/test_formulas.py
"""

import sys
import os

# Добавляем корень проекта в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from core.formulas import k_pz, p_evac, p_presence, r_i_with_doors, r_ij, t_ne_formula
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
# Формула (6): P_э — три ветви
# ─────────────────────────────────────────────────────

def test_p_evac_full_evac():
    """tр + tн.э ≤ 0.8·tбл и tск ≤ 6 → 0.999."""
    assert p_evac(t_p=2.0, t_bl=10.0, t_ne=3.0, t_ck=1.0) == pytest.approx(0.999)


def test_p_evac_full_evac_exactly_border():
    """tр + tн.э = 0.8·tбл → 0.999."""
    assert p_evac(t_p=2.0, t_bl=10.0, t_ne=6.0, t_ck=1.0) == pytest.approx(0.999)


def test_p_evac_partial():
    """0.8·tбл попадает между tр и tр + tн.э → промежуточное значение (0..0.999)."""
    result = p_evac(t_p=5.0, t_bl=10.0, t_ne=5.0, t_ck=1.0)
    assert 0 < result < 0.999


def test_p_evac_partial_value():
    """Проверка конкретного промежуточного значения: 0.999 × (8 - 5) / 5 = 0.5994."""
    result = p_evac(t_p=5.0, t_bl=10.0, t_ne=5.0, t_ck=1.0)
    expected = 0.999 * (8.0 - 5.0) / 5.0
    assert result == pytest.approx(expected)


def test_p_evac_zero_late():
    """tр ≥ 0.8·tбл → 0."""
    assert p_evac(t_p=9.0, t_bl=10.0, t_ne=1.0, t_ck=1.0) == pytest.approx(0.0)


def test_p_evac_zero_crowd():
    """tск > 6 → 0 (независимо от остальных параметров)."""
    assert p_evac(t_p=1.0, t_bl=100.0, t_ne=1.0, t_ck=7.0) == pytest.approx(0.0)


def test_p_evac_zero_crowd_exactly_6():
    """tск = 6 — граница, полная эвакуация разрешена."""
    assert p_evac(t_p=1.0, t_bl=100.0, t_ne=1.0, t_ck=6.0) == pytest.approx(0.999)


def test_p_evac_zero_t_bl():
    """tбл = 0 → эвакуация невозможна (граничный случай)."""
    assert p_evac(t_p=1.0, t_bl=0.0, t_ne=1.0, t_ck=1.0) == pytest.approx(0.0)


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


def test_k_pz_clamp():
    """Значения > K_MAX (0.99) зажимаются."""
    result = k_pz(1.0, 1.0, 1.0)
    expected = k_pz(0.99, 0.99, 0.99)
    assert result == pytest.approx(expected)


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


# ─────────────────────────────────────────────────────
# Формула (8): r_i_with_doors
# ─────────────────────────────────────────────────────

def test_r_doors_open_only():
    """Только открытая дверь: R_final = 0.3 × R_open."""
    result = r_i_with_doors(r_open=1e-6, r_closed=0.0)
    assert result == pytest.approx(0.3 * 1e-6)


def test_r_doors_closed_only():
    """Только закрытая дверь: R_final = 0.7 × R_closed."""
    result = r_i_with_doors(r_open=0.0, r_closed=1e-6)
    assert result == pytest.approx(0.7 * 1e-6)


def test_r_doors_combination():
    """Комбинация: R_final = 0.3 × R_open + 0.7 × R_closed."""
    r_o, r_c = 2e-6, 5e-7
    result = r_i_with_doors(r_open=r_o, r_closed=r_c)
    assert result == pytest.approx(0.3 * r_o + 0.7 * r_c)


# ─────────────────────────────────────────────────────
# Формула t_не (Приложение 4, П4.1)
# ─────────────────────────────────────────────────────

def test_t_ne_formula_100m2():
    """F = 100 м² → t = (5 + 0.01×100)/60 = 6/60 = 0.1 мин."""
    assert t_ne_formula(100.0) == pytest.approx(6.0 / 60.0)


def test_t_ne_formula_zero():
    assert t_ne_formula(0.0) == pytest.approx(5.0 / 60.0)


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
