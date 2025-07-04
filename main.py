import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

pd.options.display.float_format = '{:,.2f}'.format
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

data = pd.read_csv('data/cost_revenue_dirty.csv')

def run_challenge(description, func):
    print(f"\nChallenge: {description}")
    input("Press ENTER to see the result...\n")
    func()

# Challenge 1: Explore data shape, samples, duplicates, and data types
def explore_data():
    print("Shape:", data.shape)
    print("Sample 5 rows:\n", data.sample(5))
    print("Last 5 rows:\n", data.tail())
    print(f"Any NaN values? {data.isna().values.any()}")
    print(f"Any duplicates? {data.duplicated().values.any()}")
    duplicates = data[data.duplicated()]
    print(f"Number of duplicates: {len(duplicates)}")
    data.info()

# Challenge 2: Clean monetary columns and convert dates
def clean_data():
    chars_to_remove = [',', '$']
    columns_to_clean = ['USD_Production_Budget', 'USD_Worldwide_Gross', 'USD_Domestic_Gross']
    for col in columns_to_clean:
        for char in chars_to_remove:
            data[col] = data[col].astype(str).str.replace(char, "", regex=False)
        data[col] = pd.to_numeric(data[col])
    data['Release_Date'] = pd.to_datetime(data['Release_Date'])

# Challenge 3: Generate descriptive statistics and investigate outliers
def descriptive_stats():
    print(data.describe())
    print(data[data.USD_Production_Budget == 1100.00])
    print(data[data.USD_Production_Budget == 425000000.00])

# Challenge 4: Analyze films with zero domestic or worldwide gross
def zero_revenue_analysis():
    zero_domestic = data[data.USD_Domestic_Gross == 0]
    print(f'Number of films that grossed $0 domestically: {len(zero_domestic)}')
    print(zero_domestic.sort_values('USD_Production_Budget', ascending=False))
    zero_worldwide = data[data.USD_Worldwide_Gross == 0]
    print(f'Number of films that grossed $0 worldwide: {len(zero_worldwide)}')
    print(zero_worldwide.sort_values('USD_Production_Budget', ascending=False))

# Challenge 5: Filter films zero domestic gross but non-zero worldwide gross
def filter_multiple_conditions():
    international_releases = data.loc[(data.USD_Domestic_Gross == 0) & (data.USD_Worldwide_Gross != 0)]
    print(f'Number of international releases: {len(international_releases)}')
    print(international_releases.head())
    international_releases_query = data.query('USD_Domestic_Gross == 0 and USD_Worldwide_Gross != 0')
    print(f'Number of international releases (query): {len(international_releases_query)}')
    print(international_releases_query.tail())

# Challenge 6: Identify unreleased films as of scrape date and clean data
def unreleased_films():
    scrape_date = pd.Timestamp('2018-05-01')
    future_releases = data[data.Release_Date >= scrape_date]
    print(f'Number of unreleased movies: {len(future_releases)}')
    print(future_releases)
    global data_clean
    data_clean = data.drop(future_releases.index)
    print(f"Number of rows dropped: {data.shape[0] - data_clean.shape[0]}")

# Challenge 7: Calculate fraction of money-losing films by comparing budget and worldwide gross
def money_losing_films():
    money_losing = data_clean.loc[data_clean.USD_Production_Budget > data_clean.USD_Worldwide_Gross]
    print(f"Fraction money losing (loc): {len(money_losing)/len(data_clean):.4f}")
    money_losing_query = data_clean.query('USD_Production_Budget > USD_Worldwide_Gross')
    print(f"Fraction money losing (query): {money_losing_query.shape[0]/data_clean.shape[0]:.4f}")

# Challenge 8: Plot scatterplot of production budget vs worldwide gross
def scatterplots_basic():
    plt.figure(figsize=(8,4), dpi=200)
    ax = sns.scatterplot(data=data_clean,
                         x='USD_Production_Budget',
                         y='USD_Worldwide_Gross')
    ax.set(ylim=(0, 3_000_000_000),
           xlim=(0, 450_000_000),
           ylabel='Revenue in $ billions',
           xlabel='Budget in $100 millions')
    plt.show()

# Challenge 9: Plot scatterplot with color and size mapped to worldwide gross
def scatterplots_colored_sized():
    plt.figure(figsize=(8,4), dpi=200)
    ax = sns.scatterplot(data=data_clean,
                         x='USD_Production_Budget',
                         y='USD_Worldwide_Gross',
                         hue='USD_Worldwide_Gross',
                         size='USD_Worldwide_Gross')
    ax.set(ylim=(0, 3_000_000_000),
           xlim=(0, 450_000_000),
           ylabel='Revenue in $ billions',
           xlabel='Budget in $100 millions')
    plt.show()

# Challenge 10: Scatterplot with seaborn style context
def scatterplot_styled():
    plt.figure(figsize=(8,4), dpi=200)
    with sns.axes_style('darkgrid'):
        ax = sns.scatterplot(data=data_clean,
                             x='USD_Production_Budget',
                             y='USD_Worldwide_Gross',
                             hue='USD_Worldwide_Gross',
                             size='USD_Worldwide_Gross')
        ax.set(ylim=(0, 3_000_000_000),
               xlim=(0, 450_000_000),
               ylabel='Revenue in $ billions',
               xlabel='Budget in $100 millions')
    plt.show()

# Challenge 11: Bubble chart showing releases over time with budget and revenue
def bubble_chart_over_time():
    plt.figure(figsize=(8,4), dpi=200)
    with sns.axes_style("darkgrid"):
        ax = sns.scatterplot(data=data_clean,
                             x='Release_Date',
                             y='USD_Production_Budget',
                             hue='USD_Worldwide_Gross',
                             size='USD_Worldwide_Gross')
        ax.set(ylim=(0, 450_000_000),
               xlim=(data_clean.Release_Date.min(), data_clean.Release_Date.max()),
               xlabel='Year',
               ylabel='Budget in $100 millions')
    plt.show()

# Challenge 12: Add a Decade column for release decade
def add_decade_column():
    dt_index = pd.DatetimeIndex(data_clean.Release_Date)
    years = dt_index.year
    decades = (years // 10) * 10
    data_clean['Decade'] = decades
    print(data_clean[['Release_Date', 'Decade']].head())

# Challenge 13: Split data into old films (<=1960s) and new films (>1960s)
def split_old_new_films():
    global old_films, new_films
    old_films = data_clean[data_clean.Decade <= 1960]
    new_films = data_clean[data_clean.Decade > 1960]
    print(f"Old films count: {len(old_films)}")
    print(f"New films count: {len(new_films)}")
    print(old_films.describe())
    print(old_films.sort_values('USD_Production_Budget', ascending=False).head())

# Challenge 14: Regression plot for old films
def regression_old_films():
    sns.regplot(data=old_films, x='USD_Production_Budget', y='USD_Worldwide_Gross')
    plt.show()

# Challenge 15: Styled regression plot for old films
def regression_old_films_styled():
    plt.figure(figsize=(8,4), dpi=200)
    with sns.axes_style("whitegrid"):
        sns.regplot(data=old_films,
                    x='USD_Production_Budget',
                    y='USD_Worldwide_Gross',
                    scatter_kws={'alpha': 0.4},
                    line_kws={'color': 'black'})
    plt.show()

# Challenge 16: Regression plot for new films with styling
def regression_new_films():
    plt.figure(figsize=(8,4), dpi=200)
    with sns.axes_style('darkgrid'):
        ax = sns.regplot(data=new_films,
                         x='USD_Production_Budget',
                         y='USD_Worldwide_Gross',
                         color='#2f4b7c',
                         scatter_kws={'alpha': 0.3},
                         line_kws={'color': '#ff7c43'})
        ax.set(ylim=(0, 3_000_000_000),
               xlim=(0, 450_000_000),
               ylabel='Revenue in $ billions',
               xlabel='Budget in $100 millions')
    plt.show()

# Challenge 17: Fit linear regression for new films and print stats
def linear_regression_new_films():
    regression = LinearRegression()
    X = pd.DataFrame(new_films, columns=['USD_Production_Budget'])
    y = pd.DataFrame(new_films, columns=['USD_Worldwide_Gross'])
    regression.fit(X, y)
    print(f"The intercept is: {regression.intercept_[0]:,.2f}")
    print(f"The slope coefficient is: {regression.coef_[0][0]:,.6f}")
    print(f"The r-squared is: {regression.score(X, y):.4f}")

# ------------------ Run Challenges ------------------

run_challenge("Explore data shape, samples, duplicates, and data types", explore_data)

run_challenge("Clean monetary columns and convert dates", clean_data)

run_challenge("Generate descriptive statistics and investigate outliers", descriptive_stats)

run_challenge("Analyze films with zero domestic or worldwide gross", zero_revenue_analysis)

run_challenge("Filter films zero domestic gross but non-zero worldwide gross", filter_multiple_conditions)

run_challenge("Identify unreleased films as of scrape date and clean data", unreleased_films)

run_challenge("Calculate fraction of money-losing films by comparing budget and worldwide gross", money_losing_films)

run_challenge("Plot scatterplot of production budget vs worldwide gross", scatterplots_basic)

run_challenge("Plot scatterplot with color and size mapped to worldwide gross", scatterplots_colored_sized)

run_challenge("Scatterplot with seaborn style context", scatterplot_styled)

run_challenge("Bubble chart showing releases over time with budget and revenue", bubble_chart_over_time)

run_challenge("Add a Decade column for release decade", add_decade_column)

run_challenge("Split data into old films (<=1960s) and new films (>1960s)", split_old_new_films)

run_challenge("Regression plot for old films", regression_old_films)

run_challenge("Styled regression plot for old films", regression_old_films_styled)

run_challenge("Regression plot for new films with styling", regression_new_films)

run_challenge("Fit linear regression for new films and print stats", linear_regression_new_films)

