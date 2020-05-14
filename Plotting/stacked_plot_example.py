import pandas as pd
import matplotlib.pyplot as plt

# we load the data with pandas
df = pd.read_csv('./data/ch03-energy-production.csv')

# we give names for the columns that we want to load
# different types of energy have been orderd by total production values
columns = ['Coal', 'Natural Gas (Dry)', 'Crude Oil', 'Nuclear Electric Power', 'Biomass Energy',
           'Hydroelectric Power', 'Natural Gas Plant Liquids', 'Wind Energy', 'Geothermal Energy',
           'Solar/PV Energy']

# we define some specific colors to plot each type of energy produced
colors = ['darkslategray', 'powderblue', 'darkmagenta', 'lightgreen', 'sienna',
          'royalblue', 'mistyrose', 'lavender', 'tomato', 'gold']

# Let's create the figure
plt.figure(figsize=(12, 8))

# T: the transpose operator
# df['Year']: (42,), df[columns].values.T (10, 42)
# stackplot creates a list of polygons
polys = plt.stackplot(df['Year'], df[columns].values.T, colors=colors)

# the legend is not yer supported with stackplot
# we will add it manually
rectangles = []

for poly in polys:
    # plt.Rectangle creates the legend's rectangles
    rectangles.append(plt.Rectangle((0, 0), 1, 1, fc=poly.get_facecolor()[0]))

# loc specify the location of legend, 3 means lower left
legend = plt.legend(rectangles, columns, loc=3)

frame = legend.get_frame()
frame.set_color('white')

# we add some information to the plot
plt.title('Primary Energy Production by Source', fontsize=16)
plt.xlabel('Year', fontsize=16)
plt.ylabel('Production (Quad BTU)', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlim(1973, 2014)

plt.show()
