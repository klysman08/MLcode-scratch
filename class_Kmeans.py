# %%
from scipy.spatial import distance
import statistics
import matplotlib.pyplot as plt
import numpy as np

class KNN:
    def __init__(self, k=3):
        self.k = k
        self.x_train = None
        self.y_train = None
    
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
    
    def predict(self, x_test):
        distances = []
        x1 = x_test
        for x2 in self.x_train:
            dist = distance.euclidean(x1, x2)
            distances.append(dist)
        
        indices = []
        cl = []
        for i in range(self.k):
            ind = np.argmin(distances)
            distances[ind] = np.max(distances)
            indices.append(ind)
            cl.append(self.y_train[ind])
        
        print("Classes:", cl)
        classification = statistics.mode(cl)
        return classification
    
    def plot(self, x_test):
        plt.scatter(self.x_train[:,0], self.x_train[:,1], c=self.y_train, s=150, marker='o', edgecolor='black')
        plt.plot(x_test[0], x_test[1], marker='s', markersize=15, color="black")
        plt.xlim(0.2, 1.6)
        plt.ylim(0, 1.6)
        plt.savefig('knn.eps')
        plt.show(True)

# Example usage
if __name__ == "__main__":
    # Create KNN instance
    knn = KNN(k=3)
    
    # Training data
    x_train = np.array([[1,0.5], [0.8,0.8], [1.2,1.4], [0.6,0.4], [0.4,1.2], [1.5,1]])
    y_train = np.array(['white', 'gray', 'white', 'gray', 'gray', 'white'], dtype='str')
    
    # Fit the model
    knn.fit(x_train, y_train)
    
    # Test data
    x_test = np.array([1,1])
    
    # Make prediction
    cl = knn.predict(x_test)
    print("Classification:", cl)
    
    # Plot results
    knn.plot(x_test)


# %%
