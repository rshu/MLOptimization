from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np

# llcrnrlon: lower-left corner longitude
# llcrnrlat: lower-left corner latitude
# urcrnrlon: upper-right corner longitude
# urcrnrlat: upper-right corner latitude
map = Basemap(projection='merc',
              resolution='h',
              area_thresh=100,
              llcrnrlon=-180, llcrnrlat=15,
              urcrnrlon=-59, urcrnrlat=70)

shapeinfo = map.readshapefile('cities', 'cities')

x, y = zip(*map.cities)

# build a list of US cities
city_names = []
for each in map.cities_info:
    if each['COUNTRY'] != 'US':
        city_names.append("")
    else:
        city_names.append(each['NAME'])

map.drawcoastlines()
map.drawcountries()
map.fillcontinents(color='coral', lake_color='aqua')
map.drawmapboundary(fill_color='aqua')
map.drawmeridians(np.arange(0, 360, 30))
map.drawparallels(np.arange(-90, 90, 30))

# draw city markers
map.scatter(x, y, 25, marker='o', zorder=10)

# plot labels at City coords.
for city_label, city_x, city_y in zip(city_names, x, y):
    plt.text(city_x, city_y, city_label)

plt.title('Cities in USA')

plt.show()
