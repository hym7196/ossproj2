import pandas as pd

data_df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')

for year in range(2015, 2019):
    year_data = data_df[data_df['year'] == year]

    print(f"Top 10 players in {year}")

    # Top 10 players in hits (H)
    top_10_hits = year_data.sort_values(by='H', ascending=False).head(10)
    print(f"hits: {', '.join(top_10_hits['batter_name'].tolist())}")

    # Top 10 players in batting average (avg)
    top_10_avg = year_data.sort_values(by='avg', ascending=False).head(10)
    print(f"batting average: {', '.join(top_10_avg['batter_name'].tolist())}")

    # Top 10 players in homerun (HR)
    top_10_hr = year_data.sort_values(by='HR', ascending=False).head(10)
    print(f"homerun: {', '.join(top_10_hr['batter_name'].tolist())}")

    # Top 10 players in on-base percentage (OBP)
    top_10_obp = year_data.sort_values(by='OBP', ascending=False).head(10)
    print(f"on-base percentage: {', '.join(top_10_obp['batter_name'].tolist())}")

    print()

year_2018_data = data_df[data_df['year'] == 2018]

top_player_by_position = year_2018_data.groupby('cp')['war'].idxmax().apply(lambda x: year_2018_data.loc[x])
top_players_str = ', '.join([f"{row['cp']}: {row['batter_name']}" for _, row in top_player_by_position.iterrows()])
print(f"Player with the highest war by position in 2018: {top_players_str}\n")

correlations = data_df[['salary', 'R', 'H', 'HR', 'RBI', 'SB', 'war', 'avg', 'OBP', 'SLG']].corr()

correlations = correlations.drop('salary')

highest_correlation = correlations['salary'].idxmax()
highest_correlation_value = correlations.loc[highest_correlation, 'salary']

print(f"The attribute with the highest correlation with salary is '{highest_correlation}' with a correlation value of {highest_correlation_value:.2f}")

