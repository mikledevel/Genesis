"""scripts/make_report.py reads what real clients send: ';'-separated cp1251 CSV (Russian Excel), UTF-8 with BOM, and .xlsx."""
import datetime
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
_spec = importlib.util.spec_from_file_location("make_report_cli", os.path.join(ROOT, "scripts", "make_report.py"))
cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cli)


def _frame(n=150):
    r = np.random.default_rng(2)
    d0 = datetime.date(2025, 3, 1)
    return pd.DataFrame({"Дата": [(d0 + datetime.timedelta(days=i)) for i in range(n)],
                         "Продажи": (100 + 20 * np.sin(np.arange(n) * 2 * np.pi / 7) + r.normal(0, 3, n)).round(1)})


def test_semicolon_cp1251_csv_with_russian_headers_and_dotted_dates(tmp_path):
    df = _frame()
    out = df.assign(Дата=[d.strftime("%d.%m.%Y") for d in df["Дата"]])
    path = tmp_path / "r.csv"
    out.to_csv(path, sep=";", index=False, encoding="cp1251")
    got = cli.read_table(str(path))
    assert list(got.columns) == ["Дата", "Продажи"] and len(got) == len(df) and got["Дата"].iloc[0] == "01.03.2025"


def test_utf8_with_bom_and_comma(tmp_path):
    path = tmp_path / "b.csv"
    _frame().to_csv(path, index=False, encoding="utf-8-sig")
    assert list(cli.read_table(str(path)).columns) == ["Дата", "Продажи"]               # no '\ufeffДата'


def test_xlsx_first_sheet(tmp_path):
    df = _frame()
    path = tmp_path / "x.xlsx"
    with pd.ExcelWriter(path) as xw:
        df.to_excel(xw, sheet_name="Продажи", index=False)
        pd.DataFrame({"x": [1]}).to_excel(xw, sheet_name="Другое", index=False)
    got = cli.read_table(str(path))
    assert list(got.columns) == ["Дата", "Продажи"] and len(got) == len(df)


def test_old_xls_gets_a_clear_message(tmp_path):
    p = tmp_path / "old.xls"
    p.write_bytes(b"x")
    with pytest.raises(SystemExit, match="xlsx or CSV"):
        cli.read_table(str(p))


def test_end_to_end_from_an_excel_file(tmp_path, monkeypatch, capsys):
    path = tmp_path / "wb.xlsx"
    _frame().to_excel(path, index=False)
    out = tmp_path / "report.html"
    monkeypatch.setattr(sys, "argv", ["make_report.py", str(path), "--client", "WB shop", "--lang", "ru", "--horizon", "7", "--out", str(out)])
    assert cli.main() == 0
    html = out.read_text(encoding="utf-8")
    assert "WB shop" in html and "Насколько этому можно верить" in html and "Продажи" in html
