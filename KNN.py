from sklearn import datasets
from sklearn.model_selection import train_test_split
import math


#Lets load in our test dataset!
iris = datasets.load_iris()


X = iris.data
Y = iris.target


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)



#This is our distance metric
def euclidean_distance(point_1,point_2):
    total = 0
    for i in range(0,len(point_2)):
        total += (point_1[i] - point_2[i])**2

    return math.sqrt(total)


'''
k - Number of neighbors
X_train - Input training data
y_train - Associated labels for training data
X_test - The datapoint you would like labeled with out knn model
'''
def knn(k, X_train, y_train, X_test):
    #Find K neighbors
    distance_list = []
    for i in range(0,len(X_train)):

        distance = euclidean_distance(X_train[i], X_test)

        distance_list.append((distance , y_train[i]))

    distance_list.sort()

    K_neighbors = distance_list[0:k]
    K=3
    #Format of K_neighbors [(distance, class), (distance, class) ...]
    # Find K neighbors

    #Voting to decide on output class
    class_dict = {}
    for data_point in K_neighbors:

        class_label = data_point[1]


        if class_label in class_dict:
            class_dict[class_label] += 1

        else:
            class_dict[class_label] = 1

    print(class_dict)
    return max(class_dict, key=class_dict.get)

#Iterate throu Everything to Check
for i in range(0,100):
    print(X_test[i])
    print(y_test[i])
    print(knn(6,X_train, y_train, X_test[i]))






