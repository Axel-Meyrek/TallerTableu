"""
Análisis de 1,000 Startups para Fondo de Inversión
Fuente: Hoja1.csv + Hoja2.csv (misma base, columnas de contexto)
Genera: data.json con todos los agregados para la visualización web
"""

import pandas as pd
import numpy as np
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────
# 1. CARGA DE DATOS
# ─────────────────────────────────────────────
h1 = pd.read_csv(os.path.join(BASE, "Hoja1.csv"))
h2 = pd.read_csv(os.path.join(BASE, "Hoja2.csv"))

# Hoja1 = info descriptiva de empresa
# Hoja2 = misma base; en un contexto real contendría datos financieros.
# Como ambas son idénticas, se genera financiero simulado deterministamente.
df = h1.copy()
df.columns = df.columns.str.strip()
df = df.rename(columns={
    "ID": "id", "Name": "name", "Industry": "industry",
    "Description": "description", "Year Founded": "year_founded",
    "Employees": "employees", "State": "state",
    "City": "city", "Metro Area": "metro_area"
})

# ─────────────────────────────────────────────
# 2. GENERACIÓN DE MÉTRICAS FINANCIERAS
#    (Determinista por ID → reproducible)
# ─────────────────────────────────────────────
np.random.seed(42)

# Multiplicador de ingresos por empleado (miles de USD)
rev_mult = {
    "Software": 850, "IT Services": 620, "Financial Services": 950,
    "Health": 520, "Advertising & Marketing": 410, "Retail": 190,
    "Food & Beverage": 160, "Construction": 290,
    "Business Products & Services": 370, "Human Resources": 430,
    "Logistics & Transportation": 260, "Manufacturing": 310,
    "Real Estate": 580, "Consumer Products & Services": 350,
    "Government Services": 400, "Energy": 680, "Insurance": 780,
    "Media": 540, "Telecommunications": 640, "Travel & Hospitality": 230,
    "Education": 280, "Environmental Services": 380, "Security": 480,
    "Aerospace & Defense": 720, "Engineering": 560,
}

def pseudo_rand(sid, salt=0):
    """Pseudo-aleatoriedad determinista por ID."""
    x = np.sin(sid * 127.1 + salt * 311.7) * 43758.5453
    return x - np.floor(x)

df["r1"] = df["id"].apply(lambda x: pseudo_rand(x, 0))
df["r2"] = df["id"].apply(lambda x: pseudo_rand(x, 1))
df["r3"] = df["id"].apply(lambda x: pseudo_rand(x, 2))

df["mult"] = df["industry"].map(rev_mult).fillna(400)
df["employees_safe"] = df["employees"].clip(lower=1)

# Ingresos (miles USD)
df["revenue"] = (df["employees_safe"] * df["mult"] * (0.5 + 1.0 * df["r1"])).round(0).astype(int)

# Gastos (40–80% de ingresos)
df["expense_ratio"] = 0.40 + 0.40 * df["r2"]
df["expenses"] = (df["revenue"] * df["expense_ratio"]).round(0).astype(int)

# Beneficio y margen
df["profit"] = df["revenue"] - df["expenses"]
df["margin_pct"] = ((df["profit"] / df["revenue"]) * 100).round(1)

# Crecimiento (% anual) — empresas más nuevas tienden a crecer más rápido
df["years_since_founded"] = 2014 - df["year_founded"]
df["growth_base"] = (65 - df["years_since_founded"] * 3).clip(lower=5)
df["growth_pct"] = (df["growth_base"] * (0.3 + 0.9 * df["r3"])).round(1).clip(lower=0.5)

# Grupo sectorial
sector_map = {
    "Software": "Tecnología", "IT Services": "Tecnología",
    "Telecommunications": "Tecnología", "Security": "Tecnología",
    "Health": "Salud",
    "Financial Services": "Finanzas", "Insurance": "Finanzas", "Real Estate": "Finanzas",
    "Advertising & Marketing": "Marketing & Medios", "Media": "Marketing & Medios",
    "Business Products & Services": "Servicios", "Human Resources": "Servicios",
    "Government Services": "Servicios", "Education": "Servicios",
    "Construction": "Industria", "Manufacturing": "Industria",
    "Retail": "Retail & Consumo", "Consumer Products & Services": "Retail & Consumo",
    "Food & Beverage": "Retail & Consumo", "Travel & Hospitality": "Retail & Consumo",
    "Logistics & Transportation": "Logística & Energía",
    "Energy": "Logística & Energía", "Environmental Services": "Logística & Energía",
}
df["sector"] = df["industry"].map(sector_map).fillna("Otros")

# ─────────────────────────────────────────────
# 3. KPIs GLOBALES
# ─────────────────────────────────────────────
kpis = {
    "total_startups": int(len(df)),
    "total_industries": int(df["industry"].nunique()),
    "total_employees": int(df["employees"].sum()),
    "avg_growth_pct": float(df["growth_pct"].mean().round(1)),
    "total_revenue_M": float((df["revenue"].sum() / 1_000).round(1)),   # millones USD
    "avg_margin_pct": float(df["margin_pct"].mean().round(1)),
}

# ─────────────────────────────────────────────
# 4. DISTRIBUCIÓN POR INDUSTRIA
# ─────────────────────────────────────────────
ind_agg = (
    df.groupby("industry")
    .agg(
        count=("id", "count"),
        total_employees=("employees", "sum"),
        total_revenue=("revenue", "sum"),
        total_expenses=("expenses", "sum"),
        avg_growth=("growth_pct", "mean"),
    )
    .reset_index()
    .sort_values("count", ascending=False)
)
ind_agg["avg_growth"] = ind_agg["avg_growth"].round(1)

industries = ind_agg.to_dict(orient="records")

# ─────────────────────────────────────────────
# 5. GRUPOS SECTORIALES
# ─────────────────────────────────────────────
sector_agg = (
    df.groupby("sector")
    .agg(
        count=("id", "count"),
        total_revenue=("revenue", "sum"),
        total_expenses=("expenses", "sum"),
        avg_growth=("growth_pct", "mean"),
    )
    .reset_index()
    .sort_values("count", ascending=False)
)
sector_agg["avg_growth"] = sector_agg["avg_growth"].round(1)
sectors = sector_agg.to_dict(orient="records")

# ─────────────────────────────────────────────
# 6. TOP 25 POR CRECIMIENTO (tabla)
# ─────────────────────────────────────────────
top25_growth = (
    df.nlargest(25, "growth_pct")
    [["name", "industry", "sector", "year_founded", "employees",
      "state", "revenue", "expenses", "profit", "margin_pct", "growth_pct"]]
    .reset_index(drop=True)
)
top25_growth.index += 1
top25_growth_list = top25_growth.reset_index().rename(columns={"index": "rank"}).to_dict(orient="records")

# ─────────────────────────────────────────────
# 7. TOP 60 PARA MAPA DE ÁRBOL
# ─────────────────────────────────────────────
top60_treemap = (
    df.nlargest(60, "growth_pct")
    [["name", "industry", "sector", "growth_pct", "revenue", "employees"]]
    .to_dict(orient="records")
)

# ─────────────────────────────────────────────
# 8. SCATTER INGRESOS VS GASTOS (todos)
# ─────────────────────────────────────────────
scatter = df[["name", "industry", "sector", "revenue", "expenses", "growth_pct", "margin_pct"]].to_dict(orient="records")

# ─────────────────────────────────────────────
# 9. CONJUNTOS Y MEJORES CANDIDATAS
#    Conjunto A: Altos ingresos (top 30%)
#    Conjunto B: Bajos gastos ratio (bottom 30% de expense ratio)
#    Conjunto C: Alto crecimiento (top 30%)
#    Intersección A ∩ B ∩ C → mejores candidatas
# ─────────────────────────────────────────────
rev_p70  = df["revenue"].quantile(0.70)
df["exp_ratio"] = df["expenses"] / df["revenue"]
exp_p30  = df["exp_ratio"].quantile(0.30)
grow_p70 = df["growth_pct"].quantile(0.70)

set_high_revenue  = df[df["revenue"]   >= rev_p70]
set_low_expenses  = df[df["exp_ratio"] <= exp_p30]
set_high_growth   = df[df["growth_pct"] >= grow_p70]

best = df[
    (df["revenue"]    >= rev_p70) &
    (df["exp_ratio"]  <= exp_p30) &
    (df["growth_pct"] >= grow_p70)
].sort_values("profit", ascending=False)

candidates = (
    best[["name", "industry", "sector", "year_founded", "employees",
          "state", "revenue", "expenses", "profit", "margin_pct", "growth_pct", "exp_ratio"]]
    .head(20)
    .to_dict(orient="records")
)

# Limpiar exp_ratio del df (era temporal)
for c in candidates:
    c["exp_ratio_pct"] = round(c.pop("exp_ratio") * 100, 1)

# Umbrales de conjuntos (para mostrar en la UI)
set_thresholds = {
    "high_revenue_min_K":  int(rev_p70),
    "low_expense_max_pct": round(exp_p30 * 100, 1),
    "high_growth_min_pct": round(grow_p70, 1),
    "set_A_count": int(len(set_high_revenue)),
    "set_B_count": int(len(set_low_expenses)),
    "set_C_count": int(len(set_high_growth)),
    "intersection_count": int(len(best)),
}

# ─────────────────────────────────────────────
# 10. TOP 10 POR SECTOR (para referencia)
# ─────────────────────────────────────────────
top10_by_sector = {}
for sector, grp in df.groupby("sector"):
    top = grp.nlargest(5, "profit")[["name", "industry", "revenue", "expenses", "profit", "growth_pct"]].to_dict(orient="records")
    top10_by_sector[sector] = top

# ─────────────────────────────────────────────
# 11. EXPORTAR JSON
# ─────────────────────────────────────────────
output = {
    "kpis": kpis,
    "industries": industries,
    "sectors": sectors,
    "top25_growth": top25_growth_list,
    "top60_treemap": top60_treemap,
    "scatter": scatter,
    "candidates": candidates,
    "set_thresholds": set_thresholds,
    "top10_by_sector": top10_by_sector,
}

out_path = os.path.join(BASE, "data.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

# También exportar como data.js (variable global) para uso local sin servidor
js_path = os.path.join(BASE, "data.js")
with open(js_path, "w", encoding="utf-8") as f:
    f.write("// Generado automáticamente por analisis.py\n")
    f.write("window.STARTUP_DATA = ")
    json.dump(output, f, ensure_ascii=False)
    f.write(";\n")

print(f"✓ data.json generado en {out_path}")
print(f"✓ data.js  generado en {js_path}")
print(f"  KPIs: {kpis}")
print(f"  Candidatas: {len(candidates)}")
print(f"  Umbrales: {set_thresholds}")
