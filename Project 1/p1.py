"""MATH96012 2019 Project 1
Alexander John Pinches 
CID: 01201653
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as ds
import scipy.optimize as op
#--------------------------------
plt.style.use('ggplot') # plot style
plt.rcParams["figure.figsize"] = [8,8] # figure size

def simulate1(N=64,L=8,s0=0.2,r0=1,A=0.2,Nt=100):
    """Part1: Simulate bacterial colony dynamics
    Input:
    N: number of particles
    L: length of side of square domain
    s0: speed of particles
    r0: particles within distance r0 of particle i influence direction of motion
    of particle i
    A: amplitude of noise
    dt: time step
    Nt: number of time steps

    Output:
    X,Y: position of all N particles at Nt+1 times
    alpha: alignment parameter at Nt+1 times

    Do not modify input or return statement without instructor's permission.

    Add brief description of approach to problem here:

    First I tought about what can we caculate and store such as R and index later.
    Then I thought about what it was possible to vectorise over and looked in scipy
    for a way to calculate the distances in compiled c code. There's no way
    at least I could think of to vecorise over t as when applyng the limits this affects everything
    downstream so we have to do it in a for loop. As we know the size of all the outputs
    we create empty arrays to store them. Wedont have to store all the thetas each time we could have two
    variables and use .copy() but assignment is faster than copying but this would use less memory.
    Much memory isn't used though.


    """
    #Set initial condition
    phi_init = np.random.rand(N)*(2*np.pi)
    r_init = np.sqrt(np.random.rand(N))
    Xinit,Yinit = r_init*np.cos(phi_init),r_init*np.sin(phi_init)
    Xinit+=L/2
    Yinit+=L/2

    theta = np.column_stack((np.random.rand(N)*(2*np.pi),np.zeros((N,Nt)))) #initial directions of motion and 0s for future values
    #---------------------
    X = np.column_stack((Xinit,np.zeros((N,Nt)))) # create array to store X with initial X
    Y = np.column_stack((Yinit,np.zeros((N,Nt)))) # create array to store Y with initial Y

    R = np.random.rand(N,Nt)*(2*np.pi) # calculate all random numbers

    alpha = np.zeros(Nt+1) # empty array to store alpha
    alpha[0] = np.abs(np.sum(np.exp(1j*theta[:,0])))/N # put initial alpha in a list
    for t in range(0,Nt): # over t
        distances = ds.squareform(ds.pdist(np.stack((X[:,t],Y[:,t]),axis=-1))) <= r0 # get array of points within r0 of each point
        
        # calculate theta
        theta[:,t+1] = np.angle(np.dot(np.exp(1j*theta[:,t]),distances)+A*np.exp(1j*R[:,t])*np.sum(distances,axis=0))

        # calculate X and Y and apply bounds
        X[:,t+1] = X[:,t] + s0*np.cos(theta[:,t+1])
        X[X[:,t+1]>L,t+1] = X[X[:,t+1]>L,t+1] - L
        X[X[:,t+1]<0,t+1] = L + X[X[:,t+1]<0,t+1]
        
        Y[:,t+1] = Y[:,t] + s0*np.sin(theta[:,t+1])
        Y[Y[:,t+1]>L,t+1] = Y[Y[:,t+1]>L,t+1] - L
        Y[Y[:,t+1]<0,t+1] = L + Y[Y[:,t+1]<0,t+1]

        # add alpha value at t+1
        alpha[t+1] = np.abs(np.sum(np.exp(1j*theta[:,t+1])))/N

    return X,Y,alpha


def analyze(A,Nt,diffTest,display=False,**kwargs):
    """Part 2: Add input variables and modify return as needed

    **kwargs passed to simulate2 
    """
    out = {} # output dict

    fig = plt.figure()
    # subplots
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    # x
    x = np.linspace(0,Nt+1,Nt+1)
    # arrays to store results
    alphaMean16 = np.zeros(len(A))
    alphaMean32 = np.zeros(len(A))
    alphaVar16 = np.zeros(len(A))
    alphaVar32 = np.zeros(len(A))
    for i in range(0,len(A)):
        # calculate alphas
        alpha16=simulate2(A=A[i],Nt=Nt,N=16,**kwargs)[2]
        alpha32=simulate2(A=A[i],Nt=Nt,N=32,**kwargs)[2]
        # add to subplots
        ax1.plot(alpha16)
        ax2.plot(alpha32)
        # store var and mean
        alphaVar16[i] = np.var(alpha16[50:])
        alphaVar32[i] = np.var(alpha32[50:])
        alphaMean16[i] = np.mean(alpha16[50:])
        alphaMean32[i] = np.mean(alpha32[50:])
    # set titles and labels
    ax1.set_title("N=16")
    ax2.set_title("N=32")
    ax1.set_ylabel(r'$\alpha$')
    ax2.set_ylabel(r'$\alpha$')
    ax2.set_xlabel('t')

    # add to output dict
    out["alphaMean16"] = alphaMean16
    out["alphaMean32"] = alphaMean32
    out["alphaVar16"] = alphaVar16
    out["alphaVar32"] = alphaVar32

    # add legend
    fig.legend(A)
    plt.savefig("alphaTime.png") # save

    # new figure
    plt.figure()
    # plot series
    plt.plot(A,alphaMean16,color="blue",label="N=16")
    plt.plot(A,alphaMean16 + 2*np.sqrt(alphaVar16),color="blue",linestyle="dashed",label="_")
    plt.plot(A,alphaMean16 - 2*np.sqrt(alphaVar16),color="blue",linestyle="dashed",label="_")
    plt.plot(A,alphaMean32,color="red",label="N=32")
    plt.plot(A,alphaMean32 + 2*np.sqrt(alphaVar32),color="red",linestyle="dashed",label="_")
    plt.plot(A,alphaMean32 - 2*np.sqrt(alphaVar32),color="red",linestyle="dashed",label="_")
    # legend
    plt.legend()
    # title and labels
    plt.title(r"$\alpha$ Vs A")
    plt.xlabel("A")
    plt.ylabel(r"$\alpha$")
    plt.savefig("alphaMean.png") # save

    # new figure
    plt.figure()
    # plot
    plt.plot(A,alphaVar16)
    plt.plot(A,alphaVar32)
    # legend
    plt.legend(["N=16","N=32"])
    # title and labels
    plt.title(r"$\mathrm{Var}[\alpha$] Vs A")
    plt.xlabel("A")
    plt.ylabel(r"$\mathrm{\alpha}$")
    plt.savefig("alphaVar.png") # save

    # results array
    y = np.zeros((len(diffTest),Nt-49))
    for i in range(0,len(diffTest)): # get alpha
        y[i,:] = simulate2(A=diffTest[i],L=4,N=32,Nt=1000)[2][50:]
    y = np.var(y,axis=1) # calculate variance
    
    Astar = diffTest[y==y.max()] # get the max

    out["Astar"] = Astar # add to output dictionary

    # new figure
    plt.figure()
    # plot
    plt.plot(diffTest,y)
    plt.plot(Astar,y.max(),marker="o")
    # add title and labels
    plt.title(r"$\mathrm{Var}[\alpha]$ Vs A")
    plt.xlabel("A")
    plt.ylabel(r"$\mathrm{Var}[\alpha]$")
    plt.savefig("rate.png") #save

    # dependence
    A2 = np.linspace(0.2,Astar-0.025)
    x = 1 - A2/Astar
    alphaMean = np.zeros(len(A2))
    alphaVar = np.zeros(len(A2)) 
    
    for i in range(0,len(A2)):
        alpha = simulate2(A=A2[i],Nt=Nt,N=32,**kwargs)[2][50:]
        alphaMean[i] = np.mean(alpha)
        alphaVar[i] = np.var(alpha)

    # new fig
    plt.figure()
    # plot
    plt.plot(x,alphaMean,color="red")
    plt.plot(x,alphaMean + 2*np.sqrt(alphaVar),color="blue",linestyle="dashed")
    plt.plot(x,alphaMean - 2*np.sqrt(alphaVar),color="blue",linestyle="dashed")
    # title and labels
    plt.title(r"$\alpha$ Vs $1 - \frac{A}{A^{*}}$")
    plt.xlabel(r"$1 - \frac{A}{A^{*}}$")
    plt.ylabel(r"$\alpha$")
    plt.savefig("alphaDependence.png") # save

    # new fig
    plt.figure()
    # plot
    plt.plot(x,alphaVar)
    # title and labels
    plt.title(r"$\mathrm{Var}[\alpha]$ Vs $1 - \frac{A}{A^{*}}$")
    plt.xlabel(r"$1 - \frac{A}{A^{*}}$")
    plt.ylabel(r"$\mathrm{Var}[\alpha$]")
    plt.savefig("alphaDependenceVar.png")

    # add to out dict
    out["1-A/Astar"] = x
    out["AstarMean"] = alphaMean
    out["AstarVar"] = alphaVar

    if display: # display
        plt.show()

    


    return out


def simulate2(A,N,Nt,L=4,**kwargs):
    """Part 2: Simulation code for Part 2, add input variables and simulation
    code needed for simulations required by analyze
    """

    return simulate1(A=A,N=N,L=L,Nt=Nt,**kwargs)

#--------------------------------------------------------------
if __name__ == '__main__':
    #The code here should call analyze and
    #generate the figures that you are submitting with your
    #discussion.
   
    output_a = analyze([0.2,0.4,0.6,0.8],1000,np.linspace(0.2,0.8,1201))
