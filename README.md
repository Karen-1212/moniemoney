This code constructs a dollar-neutral relative-PE long-short pairs strategy on the S&P 500 Consumer Discretionary Sector. 

First SnP 500 Industry Classification Data from Kaggle (updated as of July 2026) is used to screen for GICS consumer discretionary stocks, and to assign them GICS subsectors.

Then the daily trailing-twelve month(ttm) EPS was caclulated using the Yfinance package. 

Then the adjusted close price from yfinance was divided by the calculated ttm EPS to yield the daily PE.

Appropriate pairs to be traded were chosen by restricting to the same GICS sub-industry, screening for significant PE co-movement, and then remaining candidates were ranked by a mean-reversion score (correlation, ADF, half-life, excursion/reversion stats). 

The Top 10 pairs by mean reversion score were chosen.

These pairs were then monitored and the following trading rule was applied: Enter when |z| ≥ 1.2 (short the rich leg / long the cheap leg on relative PE); exit when |z| ≤ 0.75. Where z is the z-score of the relative-PE spread for a pair.

Sector capital (100,000 USD) is split equally among open trades and rebalanced on entry/exit. 


Complete Selection criteria for pairs can be found in src/pair_select.py / selection_thresholds.json

Min overlapping days: 924
Min correlation: 0.65
Max ADF t-stat: −2.20
Half-life range: 20–250 days
Z entry / exit: 1.2 / 0.75
Max pairs per ticker: 2
Max concurrent trades: 10


The model is trained on data from January 2022 to December 2025 and tested on data from January 2025 to July 2026.



TO RUN THE CODE:

1. SETUP

python3 -m venv .venv
source .venv/bin/activate # Windows: .venv\Scripts\activate
pip install -r requirements.txt


2. REBUILD WITH CURRENT DATA

python3 scripts/run_pipeline.py --force \
 && python3 scripts/select_top_pairs.py \
 && python3 scripts/portfolio_backtest.py \
 && python3 scripts/cd_pe_factsheet.py \
 && python3 scripts/cd_pe_factsheet_pdf.py


REPO STRUCTURE:

members.csv: Sector tickers + sub-industry
close.csv: Daily adjusted closes
eps_365.csv: Trailing EPS panel
pe.csv: Daily PE panel
significant_pe_pairs.csv:Statistically screened pairs
top10_significant_pe_pairs.csv: Traded pair set
selection_thresholds.json: Locked selection / trading thresholds
portfolio_backtest.xlsx: Return graph, trades, summary
…Factsheet.xlsx: Presentation Excel

DISCLAIMER:
 This repository is for research and educational use. It is not investment advice. Past simulated performance is not indicative of
future results. 
