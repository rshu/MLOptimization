import numpy as np

bounds=[(50, 150), (1, 20), (2, 20), (2, 50), (0.01, 1), (1, 10)]
popsize = 20

dimensions = len(bounds)
pop = np.random.rand(popsize, dimensions)

min_b, max_b = np.asarray(bounds).T
diff = np.fabs(min_b - max_b)
pop_denorm = min_b + pop * diff

# print(pop_denorm)
# convert from float to integer
pop_denorm_convert = pop_denorm.tolist()

print(pop_denorm_convert)
print()

result_list = []
temp_list = []

for index in pop_denorm_convert:
    print(index)
    print("...")
    # print(index[0])
    temp_list.append(np.int_(np.round_(index[0])))
    temp_list.append(np.int_(np.round_(index[1])))
    temp_list.append(np.int_(np.round_(index[2])))
    temp_list.append(np.int_(np.round_(index[3])))
    temp_list.append(float('%.2f' % index[4]))
    temp_list.append(np.int(np.round_(index[5])))
    result_list.append(temp_list)
    temp_list = []

print(result_list)
