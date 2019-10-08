import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read in data and examine first 10 rows
flights = pd.read_csv('./data/formatted_flights.csv')
print(flights.head(10))

# matplotlib histogram
plt.hist(flights['arr_delay'], color='blue', edgecolor='black',
         bins=int(180 / 5))

# seaborn histogram
sns.distplot(flights['arr_delay'], hist=True, kde=False,
             bins=int(180 / 5), color='blue',
             hist_kws={'edgecolor': 'black'})

# Add labels
plt.title('Histogram of Arrival Delays')
plt.xlabel('Delay (min)')
plt.ylabel('Flights')
plt.show()

# Show 4 different binwidths
for i, binwidth in enumerate([1, 5, 10, 15]):
    # Set up the plot
    ax = plt.subplot(2, 2, i + 1)

    # Draw the plot
    ax.hist(flights['arr_delay'], bins=int(180 / binwidth),
            color='blue', edgecolor='black')

    # Title and labels
    ax.set_title('Histogram with Binwidth = %d' % binwidth, size=12)
    ax.set_xlabel('Delay (min)', size=12)
    ax.set_ylabel('Flights', size=12)

plt.tight_layout()
plt.show()

# **********Side by side histograms**********
# Make a separate list for each airline
x1 = list(flights[flights['name'] == 'United Air Lines Inc.']['arr_delay'])
x2 = list(flights[flights['name'] == 'JetBlue Airways']['arr_delay'])
x3 = list(flights[flights['name'] == 'ExpressJet Airlines Inc.']['arr_delay'])
x4 = list(flights[flights['name'] == 'Delta Air Lines Inc.']['arr_delay'])
x5 = list(flights[flights['name'] == 'American Airlines Inc.']['arr_delay'])

# Assign colors for each airline and the names
colors = ['#E69F00', '#56B4E9', '#F0E442', '#009E73', '#D55E00']
names = ['United Air Lines Inc.', 'JetBlue Airways', 'ExpressJet Airlines Inc.',
         'Delta Air Lines Inc.', 'American Airlines Inc.']

# Make the histogram using a list of lists
# Normalize the flights and assign colors and names
plt.hist([x1, x2, x3, x4, x5], bins=int(180 / 15), density=True,
         color=colors, label=names)

# Plot formatting
plt.legend()
plt.xlabel('Delay (min)')
plt.ylabel('Normalized Flights')
plt.title('Side-by-Side Histogram with Multiple Airlines')
plt.show()

# **********stack bars**********
# Stacked histogram with multiple airlines
plt.hist([x1, x2, x3, x4, x5], bins=int(180 / 15), stacked=True,
         density=True, color=colors, label=names)

# Plot formatting
plt.legend()
plt.xlabel('Delay (min)')
plt.ylabel('Normalized Flights')
plt.title('Side-by-Side Histogram with Multiple Airlines')
plt.show()
