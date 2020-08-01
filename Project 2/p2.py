"""MATH96012 Project 2
Alexander John Pinches CID:01201653
"""

import numpy as np
import matplotlib.pyplot as plt
from m1 import lrmodel as lr #assumes that p2_dev.f90 has been compiled with: f2py -c p2_dev.f90 -m m1
# May also use scipy, scikit-learn, and time modules as needed
import scipy.optimize as op
from sklearn.neural_network import MLPClassifier

plt.style.use("ggplot")

def clrmodelWrapperCost(fvec,n,d):
    return lr.clrmodel(fvec,d,n)[0]
def clrmodelWrapperCostGrad(fvec,n,d):
    return lr.clrmodel(fvec,d,n)[1]

def mlrmodelWrapperCost(fvec,n,d,m):
    return lr.mlrmodel(fvec,n,d,m)[0]
def mlrmodelWrapperCostGrad(fvec,n,d,m):
    return lr.mlrmodel(fvec,n,d,m)[1]

def read_data(tsize=15000):
    """Read in image and label data from data.csv.
    The full image data is stored in a 784 x 20000 matrix, X
    and the corresponding labels are stored in a 20000 element array, y.
    The final 20000-tsize images and labels are stored in X_test and y_test, respectively.
    X,y,X_test, and y_test are all returned by the function.
    You are not required to use this function.
    """
    print("Reading data...") #may take 1-2 minutes
    Data=np.loadtxt('data.csv',delimiter=',')
    Data =Data.T
    X,y = Data[:-1,:]/255.,Data[-1,:].astype(int) #rescale the image, convert the labels to integers between 0 and M-1)
    Data = None

    # Extract testing data
    X_test = X[:,tsize:]
    y_test = y[tsize:]
    print("processed dataset")
    return X,y,X_test,y_test
#----------------------------

def clr_test(X,y,X_test,y_test,bnd=1.0,l=0.0,input=(None)):
    """Train CLR model with input images and labels (i.e. use data in X and y), then compute and return testing error in test_error
    using X_test, y_test. The fitting parameters obtained via training should be returned in the 1-d array, fvec_f
    X: training image data, should be 784 x d with 1<=d<=15000
    y: training image labels, should contain d elements
    X_test,y_test: should be set as in read_data above
    bnd: Constraint parameter for optimization problem
    l: l2-penalty parameter for optimization problem
    input: tuple, set if and as needed
    """
    n = X.shape[0]
    y = y%2
    
    fvec = np.random.randn(n+1)*0.1 #initial fitting parameters

    #Add code to train CLR model and evaluate testing test_error
    # set d and type
    y = y.astype(int)
    X= X.astype(np.double)
    d = X.shape[1]
    y_test = y_test%2
    # set variables in fortran
    lr.clrmodel.lr_x = X.tolist()
    lr.clrmodel.lr_y = y.tolist()
    lr.clrmodel.lr_lambda = l

    # set bounds
    bound = [(-bnd,bnd)]*n + [(-np.inf,np.inf)]

    
    
    # minimise
    fvec_f = op.minimize(clrmodelWrapperCost,fvec,(n,d),method='L-BFGS-B',jac=clrmodelWrapperCostGrad,bounds=bound).x #Modify to store final fitting parameters after training
    # predictions
    z = np.zeros((2,X_test.shape[1]))
    z[1] = fvec_f[:n]@X_test + fvec_f[-1]
    # errors
    errors = z.argmax(axis=0) != y_test



        
    test_error = errors.sum()/y_test.shape[0] #Modify to store testing error; see neural network notes for further details on definition of testing error
    output = lr.clrmodel(fvec_f,d,n) #output tuple, modify as needed
    return fvec_f,test_error,output
#--------------------------------------------

def mlr_test(X,y,X_test,y_test,m=3,bnd=1.0,l=0.0,input=(None)):
    """Train MLR model with input images and labels (i.e. use data in X and y), then compute and return testing error (in test_error)
    using X_test, y_test. The fitting parameters obtained via training should be returned in the 1-d array, fvec_f
    X: training image data, should be 784 x d with 1<=d<=15000
    y: training image labels, should contain d elements
    X_test,y_test: should be set as in read_data above
    m: number of classes
    bnd: Constraint parameter for optimization problem
    l: l2-penalty parameter for optimization problem
    input: tuple, set if and as needed
    """
    n = X.shape[0]
    y = y%m
    

    fvec = np.random.randn((m-1)*(n+1))*0.1 #initial fitting parameters

    #Add code to train MLR model and evaluate testing error, test_error
    y_test = y_test%m # modulo
    d = X.shape[1] # get d
    # set types
    y = y.astype(int)
    X= X.astype(np.double)

    # set in fortran
    lr.mlrmodel.lr_x = X.tolist()
    lr.mlrmodel.lr_y = y.tolist()
    lr.mlrmodel.lr_lambda = l

    # set bounds
    bound = [(-bnd,bnd)]*n*(m-1)+[(-np.inf,np.inf)]*(m-1)
    

    # minimise
    fvec_f = op.minimize(mlrmodelWrapperCost,fvec,(n,d,m),method='L-BFGS-B',jac=mlrmodelWrapperCostGrad,bounds=bound).x #Modify to store final fitting parameters after training
    # predict
    z = np.zeros((m,X_test.shape[1]))
    w = np.zeros((m-1,n))
    for i1 in range(n):
        j1 = i1*(m-1)
        w[:,i1] = fvec_f[j1:j1+m-1] #weight matrix
    b = fvec_f[(m-1)*n:] #bias vector
    z = ((w@X_test).T + b).T
    # return error
    errors = z.argmax(axis=0) != y_test


    test_error = errors.sum()/X_test.shape[1] #Modify to store testing error; see neural network notes for further details on definition of testing error
    output = (lr.mlrmodel(fvec_f,n,d,m)) #output tuple, modify as needed
    return fvec_f,test_error,output
#--------------------------------------------

def lr_compare():
    """ Analyze performance of MLR and neural network models
    on image classification problem
    Add input variables and modify return statement as needed.
    Should be called from name==main section below
    """
    # read in data
    X,y,X_test,y_test = read_data() 
   
    errorNN = []
    errorLR = []
    # lr error
    errorLR.append(clr_test(X,y,X_test,y_test)[1])
    for i in range(4,11,2):
        errorLR.append(mlr_test(X,y,X_test,y_test,m=i)[1])
    # nn error
    model = MLPClassifier()
    for i in range(2,11,2):
        model.fit(X.T,y%i)
        errorNN.append(1 - model.score(X_test.T,y_test%i))
    # m
    m = [i for i in range(2,11,2)]
    randomGuess = [(i-1)/i for i in m]
    # plot
    plt.figure()
    plt.plot(m,errorNN)
    plt.plot(m,errorLR)
    plt.plot(m,randomGuess)
    plt.legend(["MLPClassifier()","mlr_test()/clrtest()","Random guess"])
    plt.title("Comparison of LR and NN error for varying m. \n Alexander Pinches")
    plt.xlabel("m")
    plt.ylabel("error")
    plt.savefig("graph.png")
    plt.figure()
    plt.plot(m,errorNN)
    plt.title("NN error from the MLPClassifier() class for varying m. \n Alexander Pinches")
    plt.xlabel("m")
    plt.ylabel("error")
    plt.savefig("graphNN.png")
    plt.figure()
    plt.plot(m,errorLR)
    plt.title("LR error from the mlr_test() and clr_test() for m=2 functions for varying m. \n Alexander Pinches")
    plt.xlabel("m")
    plt.ylabel("error")
    plt.savefig("graphLR.png")
    return (errorNN,errorLR)
#--------------------------------------------

def display_image(X):
    """Displays image corresponding to input array of image data"""
    n2 = X.size
    n = np.sqrt(n2).astype(int) #Input array X is assumed to correspond to an n x n image matrix, M
    M = X.reshape(n,n)
    plt.figure()
    plt.imshow(M)
    return None
#--------------------------------------------
#--------------------------------------------


if __name__ == '__main__':
    #The code here should call analyze and generate the
    #figures that you are submitting with your code
    output = lr_compare()
    