"""
Package Import
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import warnings
import argparse

"""
Project Setup
"""
warnings.simplefilter(action="ignore", category=FutureWarning)

assets = [
    "SPY",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]
data = pd.DataFrame()

# Fetch the data for each stock and concatenate it to the `data` DataFrame
for asset in assets:
    raw = yf.download(asset, start="2012-01-01", end="2024-04-01")
    raw["Symbol"] = asset
    data = pd.concat([data, raw], axis=0)

# Initialize df and df_returns
Bdf = portfolio_data = data.pivot_table(
    index="Date", columns="Symbol", values="Adj Close"
)
df = Bdf.loc["2019-01-01":"2024-04-01"]

