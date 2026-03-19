"""
Генерация Word-отчёта (docx) с примером расчёта ИПР.
Требует: python-docx
"""

import io
from typing import Optional

import pandas as pd

from core.constants import (
    K_AP_VALUE, K_STD, P_DOOR_CLOSED, P_DOOR_OPEN, R_NORM
)
from core.formulas import k_pz, p_evac, p_presence
from utils.helpers import safe_float


def generate_report_docx(
    df_scen: pd.DataFrame,
    df_grp: pd.DataFrame,
    df_scen_calc: pd.DataFrame,
    df_rows_calc: pd.DataFrame,
    df_agg: pd.DataFrame,
    r_total: float,
    r_final: float,
    use_fire_doors: bool = False,
) -> bytes:
    """
    Генерирует Word-документ с полным примером расчёта ИПР.

    Параметры
    ----------
    df_scen       : исходные данные сценариев (из session_state)
    df_grp        : исходные данные групп (из session_state)
    df_scen_calc  : сценарии с расчётными коэффициентами
    df_rows_calc  : результаты расчёта по группам
    df_agg        : агрегирование по сценариям
    r_total       : расчётная величина ИПР без учёта дверей
    r_final       : итоговая величина ИПР (с учётом дверей если применимо)
    use_fire_doors: флаг использования формулы (8)

    Возвращает
    ----------
    bytes — содержимое .docx файла
    """
    try:
        from docx import Document
        from docx.shared import Pt, RGBColor
        from docx.enum.text import WD_ALIGN_PARAGRAPH
    except ImportError as exc:
        raise ImportError(
            "Для генерации отчёта установите python-docx: pip install python-docx"
        ) from exc

    doc = Document()

    # ── Настройка стилей ────────────────────────────────────
    style_normal = doc.styles["Normal"]
    style_normal.font.name = "Times New Roman"
    style_normal.font.size = Pt(12)

    def _add_heading(text: str, level: int = 1) -> None:
        h = doc.add_heading(text, level=level)
        h.runs[0].font.name = "Times New Roman"

    def _add_para(text: str, bold: bool = False) -> None:
        p = doc.add_paragraph()
        run = p.add_run(text)
        run.font.name = "Times New Roman"
        run.font.size = Pt(12)
        run.bold = bold

    def _add_table(headers: list[str], rows: list[list]) -> None:
        table = doc.add_table(rows=1 + len(rows), cols=len(headers))
        table.style = "Table Grid"
        hdr_cells = table.rows[0].cells
        for i, h in enumerate(headers):
            hdr_cells[i].text = h
            for para in hdr_cells[i].paragraphs:
                for run in para.runs:
                    run.bold = True
                    run.font.name = "Times New Roman"
                    run.font.size = Pt(11)
        for row_data in rows:
            row_cells = table.add_row().cells
            for i, val in enumerate(row_data):
                row_cells[i].text = str(val)
                for para in row_cells[i].paragraphs:
                    for run in para.runs:
                        run.font.name = "Times New Roman"
                        run.font.size = Pt(11)

    # Титульный раздел
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_title = title.add_run(
        "РАСЧЁТ ИНДИВИДУАЛЬНОГО ПОЖАРНОГО РИСКА\n"
        "по Методике (Приказ МЧС России от 14.11.2022 № 1140)"
    )
    run_title.bold = True
    run_title.font.name = "Times New Roman"
    run_title.font.size = Pt(14)

    doc.add_paragraph()

    # Раздел 1: Исходные данные 
    _add_heading("1. Исходные данные", level=1)

    _add_heading("1.1. Сценарии пожара", level=2)
    scen_headers = [
        "Сценарий i", "Тип здания", "Qп (год⁻¹)", "tпр (ч/сут)",
        "tбл (мин)", "Kап", "ПС", "СОУЭ", "ПДЗ",
    ]
    scen_rows = []
    for _, r in df_scen.iterrows():
        scen_rows.append([
            int(r.get("Сценарий i", 0)),
            r.get("Тип здания", ""),
            f"{safe_float(r.get('Q_п,i (год⁻¹)', 0)):.3e}",
            f"{safe_float(r.get('t_пр,i (ч/сут)', 0)):.1f}",
            f"{safe_float(r.get('t_бл,i (мин)', 0)):.2f}",
            f"{safe_float(r.get('K_ап,i', 0)):.2f}",
            "Да" if r.get("ПС соответствует? (K_обн=0.8)", False) else "Нет",
            "Да" if r.get("СОУЭ соответствует? (K_СОУЭ=0.8)", False) else "Нет",
            "Да" if r.get("ПДЗ соответствует? (K_ПДЗ=0.8)", False) else "Нет",
        ])
    _add_table(scen_headers, scen_rows)

    doc.add_paragraph()
    _add_heading("1.2. Группы эвакуируемого контингента", level=2)
    grp_headers = ["ID", "Сценарий i", "Группа j", "tр (мин)", "tн.э (мин)", "tск (мин)"]
    grp_rows = []
    for _, r in df_grp.iterrows():
        grp_rows.append([
            int(r.get("ID", 0)),
            int(r.get("Сценарий i", 0)),
            r.get("Группа j", ""),
            f"{safe_float(r.get('t_р,i,j (мин)', 0)):.2f}",
            f"{safe_float(r.get('t_н.э,i,j (мин)', 0)):.2f}",
            f"{safe_float(r.get('t_ск,i,j (мин)', 0)):.2f}",
        ])
    _add_table(grp_headers, grp_rows)

    # Раздел 2: K_п.з
    doc.add_paragraph()
    _add_heading("2. Расчёт коэффициентов противопожарной защиты — формула (7)", level=1)

    kpz_rows = []
    for _, r in df_scen_calc.iterrows():
        si = int(r.get("Сценарий i", 0))
        k_obn = safe_float(r.get("K_обн,i", 0))
        k_soue = safe_float(r.get("K_СОУЭ,i", 0))
        k_pdz = safe_float(r.get("K_ПДЗ,i", 0))
        k_pz_val = safe_float(r.get("K_п.з,i", 0))

        k_obn_str = f"{k_obn:.1f} (ПС соответствует)" if k_obn > 0 else "0.0 (ПС не соответствует)"
        k_soue_str = f"{k_soue:.1f} (СОУЭ соответствует)" if k_soue > 0 else "0.0 (СОУЭ не соответствует)"
        k_pdz_str = f"{k_pdz:.1f} (ПДЗ соответствует)" if k_pdz > 0 else "0.0 (ПДЗ не соответствует)"

        _add_para(f"Сценарий {si}:")
        _add_para(f"  Kобн,{si} = {k_obn_str}")
        _add_para(f"  Kсоуэ,{si} = {k_soue_str}")
        _add_para(f"  Kпдз,{si} = {k_pdz_str}")
        formula_str = (
            f"  Kп.з,{si} = 1 - (1 - {k_obn:.1f}×{k_soue:.1f}) × (1 - {k_obn:.1f}×{k_pdz:.1f}) "
            f"= 1 - (1 - {k_obn*k_soue:.4f}) × (1 - {k_obn*k_pdz:.4f}) = {k_pz_val:.4f}"
        )
        _add_para(formula_str)
        kpz_rows.append([si, f"{k_obn:.1f}", f"{k_soue:.1f}", f"{k_pdz:.1f}", f"{k_pz_val:.4f}"])

    doc.add_paragraph()
    _add_table(["Сценарий i", "Kобн", "Kсоуэ", "Kпдз", "Kп.з"], kpz_rows)

    #  Раздел 3: P_пр
    doc.add_paragraph()
    _add_heading("3. Расчёт вероятности присутствия — формула (5)", level=1)

    for _, r in df_scen_calc.iterrows():
        si = int(r.get("Сценарий i", 0))
        t_pr = safe_float(r.get("t_пр,i (ч/сут)", 0))
        p_pr = safe_float(r.get("P_пр,i", 0))
        _add_para(f"Pпр,{si} = tпр,{si} / 24 = {t_pr:.1f} / 24 = {p_pr:.4f}")

    # ── Раздел 4: P_э
    doc.add_paragraph()
    _add_heading("4. Расчёт вероятности эвакуации — формула (6)", level=1)

    for _, r in df_rows_calc.iterrows():
        row_id = int(r.get("ID", 0))
        si = int(r.get("Сценарий i", 0))
        grp = r.get("Группа j", "")
        t_bl = safe_float(r.get("t_бл,i (мин)", 0))
        t_p = safe_float(r.get("t_р,i,j (мин)", 0))
        t_ne = safe_float(r.get("t_н.э,i,j (мин)", 0))
        t_ck = safe_float(r.get("t_ск,i,j (мин)", 0))
        p_e = safe_float(r.get("P_э,i,j", 0))
        border = 0.8 * t_bl

        _add_para(f"ID {row_id} — Сценарий {si}, Группа «{grp}»:")
        _add_para(f"  0.8 × tбл = 0.8 × {t_bl:.2f} = {border:.2f} мин")
        _add_para(f"  tр + tн.э = {t_p:.2f} + {t_ne:.2f} = {t_p + t_ne:.2f} мин")
        _add_para(f"  tск = {t_ck:.2f} мин")

        if t_ck > 6.0:
            _add_para(f"  tск > 6 мин → Pэ = 0.000")
        elif (t_p + t_ne) <= border:
            _add_para(
                f"  {t_p + t_ne:.2f} ≤ {border:.2f} и tск = {t_ck:.2f} ≤ 6 → Pэ = 0.999"
            )
        elif t_p < border < (t_p + t_ne):
            _add_para(
                f"  Промежуточная ветвь: "
                f"Pэ = 0.999 × ({border:.2f} - {t_p:.2f}) / {t_ne:.2f} = {p_e:.4f}"
            )
        else:
            _add_para(f"  tр ≥ 0.8·tбл → Pэ = 0.000")

    #  Раздел 5: R_i,j 
    doc.add_paragraph()
    _add_heading("5. Расчёт ИПР по группам — формула (4)", level=1)

    rij_table_rows = []
    for _, r in df_rows_calc.iterrows():
        row_id = int(r.get("ID", 0))
        si = int(r.get("Сценарий i", 0))
        grp = r.get("Группа j", "")
        q_p = safe_float(r.get("Q_п,i (год⁻¹)", 0))
        k_ap = safe_float(r.get("K_ап,i", 0))
        p_pr = safe_float(r.get("P_пр,i", 0))
        p_e = safe_float(r.get("P_э,i,j", 0))
        k_pz_val = safe_float(r.get("K_п.з,i", 0))
        r_val = safe_float(r.get("R_i,j", 0))

        _add_para(
            f"R{si},{row_id} = {q_p:.2e} × (1 - {k_ap:.2f}) × {p_pr:.4f} × "
            f"(1 - {p_e:.4f}) × (1 - {k_pz_val:.4f}) = {r_val:.3e}"
        )
        rij_table_rows.append([row_id, si, grp, f"{r_val:.3e}"])

    doc.add_paragraph()
    _add_table(["ID", "Сценарий i", "Группа j", "Ri,j"], rij_table_rows)

    #  Раздел 6: R_i, R 
    doc.add_paragraph()
    _add_heading("6. Определение расчётной величины ИПР — формулы (2)–(3)", level=1)

    for _, r in df_agg.iterrows():
        si = r.get("Сценарий i", "")
        r_i = safe_float(r.get("R_i = max_j(R_i,j)", 0))
        if str(si) == "ИТОГО (R = max)":
            continue
        r_vals = df_rows_calc.loc[df_rows_calc["Сценарий i"].astype(str) == str(si), "R_i,j"]
        vals_str = ", ".join(f"{v:.3e}" for v in r_vals.tolist())
        _add_para(f"R{si} = max{{{vals_str}}} = {r_i:.3e} год⁻¹")

    _add_para(f"R = max{{Ri}} = {r_total:.3e} год⁻¹", bold=True)

    #  Раздел 7: Двери
    if use_fire_doors:
        doc.add_paragraph()
        _add_heading("7. Учёт противопожарных дверей — формула (8)", level=1)
        r_open_val = r_total
        r_closed_val = 0.0
        _add_para(
            f"R = {P_DOOR_OPEN} × R(откр) + {P_DOOR_CLOSED} × R(закр) = "
            f"{P_DOOR_OPEN} × {r_open_val:.3e} + {P_DOOR_CLOSED} × {r_closed_val:.3e} = "
            f"{r_final:.3e} год⁻¹"
        )

    #тРаздел 8: Заключение
    sec_num = 8 if use_fire_doors else 7
    doc.add_paragraph()
    _add_heading(f"{sec_num}. Заключение", level=1)

    passed = r_final <= R_NORM
    verdict = "ВЫПОЛНЕНО" if passed else "НЕ ВЫПОЛНЕНО"

    _add_para(f"R = {r_final:.3e} год⁻¹", bold=True)
    _add_para(f"Rнорм = {R_NORM:.1e} год⁻¹")
    _add_para(
        f"R {'≤' if passed else '>'} Rнорм — условие формулы (1) {verdict}",
        bold=True,
    )

    if passed:
        _add_para(
            "Расчётная величина индивидуального пожарного риска не превышает "
            "нормативного значения. Пожарный риск ДОПУСТИМЫЙ."
        )
    else:
        _add_para(
            "Расчётная величина индивидуального пожарного риска ПРЕВЫШАЕТ "
            "нормативное значение. Требуется разработка мероприятий по снижению пожарного риска."
        )

    #  Сохранение в bytes
    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.read()
