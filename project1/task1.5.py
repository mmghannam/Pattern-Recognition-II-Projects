import project1.strategy_kmeans as sk
import project1.Data as d
import project1.Data_New as dnew
import numpy as np
from scipy.spatial import distance_matrix
import scipy.linalg as la
import matplotlib.pyplot as plt
import project1.strategy_convex_hull as ch
from scipy.spatial import distance
import datetime

def evaluate(var,error,strategy,varname):
    plt.title("Variation with " + varname + " with " + strategy)
    plt.xlabel(varname)
    plt.ylabel("Error")
    plt.plot(var,error)
    plt.savefig("Variation with " + varname + " with " + strategy + ".pdf",
                papertype=None, format='pdf', transparent=False,
                bbox_inches='tight', pad_inches=0.1)
    plt.clf()

def plot_points(predictions,truth,k,n):
    plt.title("Predicted Points vs the ground truth")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(predictions[:,0],predictions[:,1],marker = "o")
    plt.scatter(truth[:,0],truth[:,1],marker="x")
    plt.savefig("Plotted Points with " +str(k)+" and "+str(n)+" points.pdf",papertype=None, format='pdf', transparent=False,
                bbox_inches='tight', pad_inches=0.1)
    plt.clf()

if __name__ == '__main__':
    data = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    k_Values = [25,50,100]
    dimension = [2,3,4]
    total_points= [100,500,1000]
    allError = []
    X_orig = np.load('faceMatrix.npy').astype('float')
    size = X_orig.shape
    X = X_orig
    #Zero centered required
    #Normalizing the data
    mean = np.mean(X_orig,axis=1)
    for i in range(size[1]):
        X[:,i] = X_orig[:,i] - mean
    C = np.cov(X)
    evals, U = np.linalg.eig(C)
    H = U[:,0:6].T @ X
    Data_input = H.T
    for i in k_Values:
        prediction_indices = ch.strategy_convex_hull(Data_input, i)
        output = np.zeros((361,i))
        for j in range(i):
            output[:,j] = X_orig[:,prediction_indices[j]]
        sumPredictions = distance.cdist(output.T / 255, output.T / 255, 'cityblock').sum() / (25 * 25)
        allError.append(sumPredictions)
        fig = plt.figure(figsize=(int(i/5),5))
        col = 5
        row = int(i/5)
        for j in range(0, col * row  ):
            img = output[:,j].reshape(19,19)
            fig.add_subplot(row, col, j+1)
            plt.imshow(img)
        plt.show()
        plt.clf()
    evaluate(total_points, allError, "convex_hull_strategy", "k values")
    allError.clear()
    for i in k_Values:
        predictions = sk.strategy_kmeans(X_orig.T, i)
        sumPredictions = distance.cdist(predictions / 255, predictions / 255, 'cityblock').sum() / (25 * 25)
        allError.append(sumPredictions)
        fig = plt.figure(figsize=(int(i / 5), 5))
        col = 5
        row = int(i / 5)
        for j in range(0, col * row):
            img = predictions[j].reshape(19, 19)
            fig.add_subplot(row, col, j + 1)
            plt.imshow(img)
        plt.show()
        plt.clf()
    evaluate(total_points, allError, "convex_hull_strategy", "k values")
    allError.clear()

    ##Validation done with own data
    #For kmeans Strategy
    #For variation with k
    for i in k_Values:
        truth, data = d.Data(k=i, dim=2, n=500)
        predictions = (sk.strategy_kmeans(data, i))
        print(i)
        sumPredictions = distance_matrix(predictions, predictions).sum()
        sumTruth = distance_matrix(truth, truth).sum()
        error = sumTruth - sumPredictions
        plot_points(predictions, truth, i, n=500)
        allError.append(error)
    evaluate(k_Values, allError, "k_means_strategy", "k values")
    allError.clear()
    #For variation with dim
    for i in dimension:
        truth, data = d.Data(k=25,dim=i,n=100)
        predictions = (sk.strategy_kmeans(data,25))
        sumPredictions = distance_matrix(predictions, predictions).sum()
        sumTruth = distance_matrix(truth, truth).sum()
        error = sumTruth - sumPredictions
        if i==2:
            plot_points(predictions, truth, k=25, n=100)
        allError.append(error)
    evaluate(dimension,allError,"kmeans_strategy","dimension values")
    allError.clear()
    #For variation with number of points:
    for i in total_points:
        truth, data = d.Data(k=25,dim=2,n=i)
        predictions = (sk.strategy_kmeans(data,25))
        sumPredictions = distance_matrix(predictions, predictions).sum()
        sumTruth = distance_matrix(truth, truth).sum()
        plot_points(predictions, truth, 25, i)
        error = sumTruth - sumPredictions
        allError.append(error)
    evaluate(total_points,allError,"kmeans_strategy","number of points")
    allError.clear()

    #For convolution hull

    for i in total_points:
        truth, data = dnew.Data(k=3,dim=2,n=i)
        predictions = (ch.strategy_convex_hull(data,3))
        sumPredictions = distance_matrix(predictions, predictions).sum()
        sumTruth = distance_matrix(truth, truth).sum()
        plot_points(predictions, truth, 3, i)
        error = sumTruth - sumPredictions
        allError.append(error)
    evaluate(total_points,allError,"convex hull_strategy","number of points")
    allError.clear()