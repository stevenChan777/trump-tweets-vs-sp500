import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import time
from scipy.stats import normaltest, ttest_ind, mannwhitneyu, f_oneway, chi2_contingency, linregress
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def load_clean_spx(sp500):
    spx_df = pd.read_csv(sp500, thousands=',', keep_default_na=True,
                         na_values=['', 'NA', 'NaN', 'null'])
    spx_df['Date'] = pd.to_datetime(spx_df['Date'], errors='coerce')
    spx_df = spx_df.sort_values(by='Date').reset_index(drop=True)
    spx_df['Close'] = pd.to_numeric(spx_df['Close'])

    # percentage change from previous day close value to the next day close value
    spx_df['Return'] = spx_df['Close'].pct_change()
    spx_df = spx_df[['Date', 'Close', 'Return']]

    return spx_df

def load_clean_trump_tweets(tweet_csv):
  
    tweets_df = pd.read_csv(tweet_csv, keep_default_na=True,
                            na_values=['', 'NA', 'NaN', 'null'])
    tweets_df = tweets_df[tweets_df['isRetweet'] != 't']
    tweets_df['date'] = pd.to_datetime(tweets_df['date'], errors='coerce')
    tweets_df['date'] = tweets_df['date'].dt.tz_localize(None)
    tweets_df = tweets_df.sort_values(by='date').reset_index(drop=True)
    mask  = (tweets_df['date'] >= '2017-01-20') & (tweets_df['date'] <= '2021-01-20')
    tweets_df = tweets_df.loc[mask]

    return tweets_df

def aggregate_tweets_daily(tweets_df):
    tweets_df['date'] = tweets_df['date'].dt.normalize()
    tweets_df = tweets_df.groupby('date')[['favorites', 'retweets']].sum().reset_index()
    tweets_df = tweets_df.rename(columns={'date': 'Date'})
    return tweets_df

def merge_spx_and_tweets(spx_df, tweet_daily_df):
    merged = pd.merge(spx_df, tweet_daily_df, on='Date', how='left').sort_values('Date').reset_index(drop=True)
    tweet_cols = merged.columns.difference(['Date', 'Close', 'Return'])
    merged[tweet_cols] = merged[tweet_cols].fillna(0)
    merged['Return_t+1'] = merged['Return'].shift(-1)
    merged = merged[~((merged['favorites'] == 0) & (merged['retweets'] == 0))]
    return merged

def run_analysis(merged, L):
    target = f'Return_t+{L}'

    # =============== Correlations (Return, Return_t+L) ===============
    corr = merged[['Return', target, 'favorites', 'retweets']].corr().round(3)
    print(f"\n[Correlation matrix] (lag = t+{L})\n", corr)

    plt.figure()
    corr_matrix = plt.imshow(corr.values, interpolation='nearest')
    plt.xticks(range(corr.shape[1]), corr.columns, rotation=45, ha='right')
    plt.yticks(range(corr.shape[0]), corr.index)
    plt.title(f'Correlation Matrix (t+{L})')
    plt.colorbar(corr_matrix, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(f'corr_matrix_t+{L}.png')
    plt.close()

    # =============== Normality  ===============
    for tgt in ['Return', target]:
        arr = merged[tgt].dropna().values
        stat, p = normaltest(arr)
        print(f"[Normality] {tgt}: stat={stat:.3f}, p={p:.4f}")


    # ===============  High vs Low activity (favorites) ===============
    #find top 10% of days (by favorites count)
    thr = merged['favorites'].quantile(0.90)
    hi = merged.loc[merged['favorites'] >= thr, target].dropna()
    lo = merged.loc[merged['favorites'] <  thr, target].dropna()

    t_stat, t_p = ttest_ind(hi, lo, equal_var=False)
    mw_stat, mw_p = mannwhitneyu(hi, lo, alternative='two-sided')
    print(f"\n[High vs Low favorites → {target}] t-test: t={t_stat:.3f}, p={t_p:.4f} (N_hi={len(hi)}, N_lo={len(lo)})")
    print(f"[High vs Low favorites → {target}] Mann–Whitney: U={mw_stat:.3f}, p={mw_p:.4f}")

    plt.figure(); plt.hist(hi, bins=20, alpha=0.8)
    plt.title(f'{target} — High Favorites (Top 10%)')
    plt.xlabel(target)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f'hist_high_fav_{target}.png'); plt.close()

    plt.figure(); plt.hist(lo, bins=20, alpha=0.8)
    plt.title(f'{target} — Other Days')
    plt.xlabel(target)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f'hist_low_fav_{target}.png')
    plt.close()

    # ===============  ANOVA + Tukey (favorites terciles) ===============
    qs = merged['favorites'].quantile([1/3, 2/3]).values
    labels = pd.cut(merged['favorites'], bins=[-np.inf, qs[0], qs[1], np.inf], labels=['Low','Med','High'])
    g_low  = merged.loc[labels=='Low',  target].dropna()
    g_med  = merged.loc[labels=='Med',  target].dropna()
    g_high = merged.loc[labels=='High', target].dropna()
    f_stat, f_p = f_oneway(g_low, g_med, g_high)
    print(f"\n[ANOVA] {target} across Low/Med/High favorites: F={f_stat:.3f}, p={f_p:.4f}")
    tukey_df = pd.DataFrame({target: merged[target], 'group': labels}).dropna()
    tukey = pairwise_tukeyhsd(endog=tukey_df[target].values,
                                groups=tukey_df['group'].values, alpha=0.05)
    print("\n[Tukey HSD]\n", tukey)

    # ===============  Chi-square ===============
    posneg = np.where(merged[target] >= 0, 'Positive', 'Negative')
    highlow = np.where(merged['favorites'] >= thr, 'High', 'Low')
    ct = pd.crosstab(posneg, highlow)
    chi2, chi_p, dof, exp = chi2_contingency(ct)
    print(f"\n[Chi-square: sign({target}) vs High/Low favorites]")
    print(ct)
    print(f"chi2={chi2:.3f}, p={chi_p:.4f}, dof={dof}")

    # ===============  linregress  ===============
    x = merged['favorites'].values
    y = merged[target].values
    mask = ~np.isnan(x) & ~np.isnan(y)
    slope, intercept, r_value, p_value, std_err = linregress(x[mask], y[mask])
    print(f"\n[linregress] {target} ~ favorites: slope={slope:.6e}, R^2={(r_value**2):.4f}, p={p_value:.4f}")
    x_line = np.linspace(x[mask].min(), x[mask].max(), 200)
    y_line = intercept + slope * x_line
    plt.figure()
    plt.scatter(x[mask], y[mask], s=10)
    plt.plot(x_line, y_line)
    plt.title(f'{target} vs Favorites (linregress fit)')
    plt.xlabel('Favorites')
    plt.ylabel(target)
    plt.tight_layout()
    plt.savefig(f'scatter_linreg_favorites_{target}.png'); plt.close()

def ensure_lags(merged, lags):
    for L in lags:
        col = f'Return_t+{L}'
        if col not in merged.columns:
            merged[col] = merged['Return'].shift(-L)
    return merged

def main(sp500, trump_tweets):
    spx = load_clean_spx(sp500)
    tweet = load_clean_trump_tweets(trump_tweets)
    tweet_daily = aggregate_tweets_daily(tweet)
    merged = merge_spx_and_tweets(spx, tweet_daily)

    lags = [1, 3, 5, 10]
    merged = ensure_lags(merged, lags)

    for L in lags:
        run_analysis(merged, L)


    #testing the load and clean df
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None) 
    #print(spx)
    #print(tweet)
    #print(tweet_daily)
    #print(merged)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])