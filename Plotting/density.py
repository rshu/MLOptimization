import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# https://towardsdatascience.com/histograms-and-density-plots-in-python-f6bda88f5ac0

# Read in data and examine first 10 rows
flights = pd.read_csv('./data/formatted_flights.csv')

# Density Plot and Histogram of all arrival delays
sns.distplot(flights['arr_delay'], hist=True, kde=True,
             bins=int(180 / 5), color='darkblue',
             hist_kws={'edgecolor': 'black'},
             kde_kws={'linewidth': 3})

plt.title('Density Plot and Histogram of Arrival Delays')
plt.xlabel('Delay (min)')
plt.ylabel('Density')
plt.show()

# List of five airlines to plot
airlines = ['United Air Lines Inc.', 'JetBlue Airways', 'ExpressJet Airlines Inc.',
            'Delta Air Lines Inc.', 'American Airlines Inc.']

# Iterate through the five airlines
for airline in airlines:
    # Subset to the airline
    subset = flights[flights['name'] == airline]

    # Draw the density plot
    sns.distplot(subset['arr_delay'], hist=False, kde=True,
                 kde_kws={'linewidth': 3},
                 label=airline)

# Plot formatting
plt.legend(prop={'size': 12}, title='Airline')
plt.title('Density Plot with Multiple Airlines')
plt.xlabel('Delay (min)')
plt.ylabel('Density')
plt.show()

# **********shaded density plot**********
for airline in ['United Air Lines Inc.', 'Alaska Airlines Inc.']:
    subset = flights[flights['name'] == airline]

    sns.distplot(subset['arr_delay'], hist=False, kde=True,
                 kde_kws={'shade': True, 'linewidth': 3}, label=airline)

plt.legend(prop={'size': 12}, title='Airline')
plt.title('Shaded Density Plot of Arrival Delays')
plt.xlabel('Delay (min)')
plt.ylabel('Density')
plt.show()

# **********rug plots**********
# Subset to Alaska Airlines
subset = flights[flights['name'] == 'Alaska Airlines Inc.']

# Density Plot with Rug Plot
sns.distplot(subset['arr_delay'], hist=False, kde=True, rug=True,
             color='darkblue',
             kde_kws={'linewidth': 3},
             rug_kws={'color': 'black'})

# Plot formatting
plt.title('Density Plot with Rug Plot for Alaska Airlines')
plt.xlabel('Delay (min)')
plt.ylabel('Density')
plt.show()
