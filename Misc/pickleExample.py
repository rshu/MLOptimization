import pickle

# take user input to take the amount of data
number_of_data = int(input('Enter the number of data : '))
data = []

# take input of the data
for i in range(number_of_data):
    raw = input('Enter data ' + str(i) + ' : ')
    data.append(raw)

# open a file, where you ant to store the data
file = open('important.p', 'wb')

# pickle.dump() function to store the object data to the file
pickle.dump(data, file)

# close the file
file.close()

# open a file, where you stored the pickled data
file = open('important.p', 'rb')

# dump information to that file
data = pickle.load(file)

# close the file
file.close()

print('Showing the pickled data:')

cnt = 0
for item in data:
    print('The data ', cnt, ' is : ', item)
    cnt += 1

