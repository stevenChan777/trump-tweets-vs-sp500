Trump Tweets vs S&P 500

Statistical analysis of whether Donald Trump’s Twitter activity influenced S&P 500 returns.

📌 Overview

This project investigates whether Donald Trump’s tweet activity (favorites and retweets) had measurable effects on short-term S&P 500 index returns during his first presidential term (2017–2021).

The analysis applies multiple statistical tests to evaluate both linear and non-linear relationships, and tests multiple lag periods (t+1, t+3, t+5, t+10 days) to determine whether market reactions occurred immediately or with delay.

⚡ Motivation

Political leaders’ social media activity can generate significant market attention. For hedge funds and financial analysts, even small predictive signals from tweet engagement could inform trading strategies or hedging decisions.

This project asks:

Do tweet engagement metrics (favorites, retweets) correlate with S&P 500 returns?

Can they serve as predictive signals for short-term trading?

🔑 Features

Data Cleaning & Merging: Combined Trump’s tweet data with S&P 500 daily returns.

Lag Analysis: Examined impacts at t+1, t+3, t+5, t+10 days.

Statistical Testing:

Correlation analysis

Normality checks

Independent t-tests

Mann–Whitney U tests

One-way ANOVA + Tukey HSD

Chi-square tests

Linear regression modeling

Trading Viability Assessment: Tested if any statistically significant results could form the basis for strategies.

🛠️ Tech Stack

Language: Python (Pandas, NumPy, Matplotlib, SciPy, Statsmodels)

Data Sources:

S&P 500 Historical Data

Trump Twitter Archive

📊 Dataset

S&P 500: Daily close prices & returns (2017/01/20 – 2021/01/20)

Tweets: Daily aggregated favorites & retweets from Trump’s account (same period)

Merged dataset aligns trading days with daily tweet engagement.

🚀 Results

No strong or consistent evidence that tweet engagement predicts S&P 500 returns.

A single significant effect at t+3 lag was detected (p < 0.05), but effect size was small and not robust.

Retweets ≈ perfectly correlated with favorites → no added predictive value.

Regression R² ≈ 0 → engagement metrics explain virtually none of the variance in returns.

🏆 Key Takeaways

No consistent relationship between Trump’s tweet engagement and S&P 500 returns.

Single significant result at t+3 likely due to chance, not a persistent pattern.

Engagement counts alone are not reliable trading signals.

Future work: analyze tweet content, sentiment, and context for stronger predictive insights.
