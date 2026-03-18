"""
Формулы Методики определения расчётных величин пожарного риска
(Приказ МЧС России от 14.11.2022 № 1140), формулы (1)–(8).
"""

from utils.helpers import clamp, safe_float
from core.constants import P_E_MAX, K_MAX, K_AP_VALUE, P_DOOR_OPEN, P_DOOR_CLOSED


def p_presence(t_pr_hours: float) -> float:
    """P_пр,i = t_пр,i / 24 — формула (5)."""
    return clamp(safe_float(t_pr_hours) / 24.0, 0.0, 1.0)


def p_evac(t_p: float, t_bl: float, t_ne: float, t_ck: float) -> float:
    """
    P_э,i,j — формула (6) №1140 (кусочная).

    Параметры
    ----------
    t_p  : расчётное время эвакуации, мин
    t_bl : время блокирования путей эвакуации, мин
    t_ne : время начала эвакуации, мин
    t_ck : время существования скоплений, мин
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
    K_п.з,i — формула (7) №1140.
    K_п.з,i = 1 - (1 - K_обн,i · K_СОУЭ,i) · (1 - K_обн,i · K_ПДЗ,i)
    """
    k_obn = clamp(safe_float(k_obn), 0.0, K_MAX)
    k_soue = clamp(safe_float(k_soue), 0.0, K_MAX)
    k_pdz = clamp(safe_float(k_pdz), 0.0, K_MAX)
    val = 1.0 - (1.0 - k_obn * k_soue) * (1.0 - k_obn * k_pdz)
    return clamp(val, 0.0, 1.0)


def r_ij(q_p: float, k_ap: float, p_pr: float, p_e: float, k_pz_val: float) -> float:
    """
    R_i,j — формула (4) №1140.
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
    t_НЭ = (5 + 0.01 · F_пом) / 60 (мин) — формула (П4.1), Приложение 4.
    """
    t_sec = 5.0 + 0.01 * safe_float(f_pom)
    return t_sec / 60.0
