"""MATH 96012 Project 3
Alexander John Pinches CID:01201653
Contains four functions:
    simulate2: Simulate bacterial dynamics over m trials. Return: all positions at final time
        and alpha at nt+1 times averaged across the m trials.
    performance: To be completed -- analyze and assess performance of python, fortran, and fortran+openmp simulation codes
    correlation: To be completed -- compute and analyze correlation function, C(tau)
    visualize: To be completed -- generate animation illustrating "non-trivial" particle dynamics
"""
import numpy as np
import matplotlib.pyplot as plt
from m1 import bmotion as bm #assumes that p3_dev.f90 has been compiled with: f2py --f90flags='-fopenmp' -c p3_dev.f90 -m m1 -lgomp
import scipy.spatial.distance as scd
#May also use scipy and time modules as needed
import time
import matplotlib.animation as animation

plt.style.use("ggplot")

def simulate2(M=10,N=64,L=8,s0=0.2,r0=1,A=0,Nt=100):
    """Simulate bacterial colony dynamics
    Input:
    M: Number of simulations
    N: number of particles
    L: length of side of square domain
    s0: speed of particles
    r0: particles within distance r0 of particle i influence direction of motion
    of particle i
    A: amplitude of noise
    Nt: number of time steps

    Output:
    X,Y: position of all N particles at Nt+1 times
    alpha: alignment parameter at Nt+1 times averaged across M simulation

    Do not modify input or return statement without instructor's permission.

    Add brief description of approach of differences from simulate1 here:
    This code carries out M simulations at a time with partial vectorization
    across the M samples.
    """
    #Set initial condition
    phi_init = np.random.rand(M,N)*(2*np.pi)
    r_init = np.sqrt(np.random.rand(M,N))
    Xinit,Yinit = r_init*np.cos(phi_init),r_init*np.sin(phi_init)
    Xinit+=L/2
    Yinit+=L/2
    #---------------------

    #Initialize variables
    P = np.zeros((M,N,2)) #positions
    P[:,:,0],P[:,:,1] = Xinit,Yinit
    alpha = np.zeros((M,Nt+1)) #alignment parameter
    S = np.zeros((M,N),dtype=complex) #phases
    T = np.random.rand(M,N)*(2*np.pi) #direction of motion
    n = np.zeros((M,N)) #number of neighbors
    E = np.zeros((M,N,Nt+1),dtype=complex)
    d = np.zeros((M,N,N))
    dtemp = np.zeros((M,N*(N-1)//2))
    AexpR = np.random.rand(M,N,Nt)*(2*np.pi)
    AexpR = A*np.exp(1j*AexpR)

    r0sq = r0**2
    E[:,:,0] = np.exp(1j*T)

    #Time marching-----------
    for i in range(Nt):
        for j in range(M):
            dtemp[j,:] = scd.pdist(P[j,:,:],metric='sqeuclidean')

        dtemp2 = dtemp<=r0sq
        for j in range(M):
            d[j,:,:] = scd.squareform(dtemp2[j,:])
        n = d.sum(axis=2) + 1
        S = E[:,:,i] + n*AexpR[:,:,i]

        for j in range(M):
            S[j,:] += d[j,:,:].dot(E[j,:,i])

        T = np.angle(S)

        #Update X,Y
        P[:,:,0] = P[:,:,0] + s0*np.cos(T)
        P[:,:,1] = P[:,:,1] + s0*np.sin(T)

        #Enforce periodic boundary conditions
        P = P%L

        E[:,:,i+1] = np.exp(1j*T)
    #----------------------

    #Compute order parameter
    alpha = (1/(N*M))*np.sum(np.abs(E.sum(axis=1)),axis=0)

    return P[:,:,0],P[:,:,1],alpha


def performance(input_p=(None),display=False):
    """Assess performance of simulate2, simulate2_f90, and simulate2_omp
    Modify the contents of the tuple, input, as needed
    When display is True, figures equivalent to those
    you are submitting should be displayed
    input_p=(s0,A,L,r0,num_threads,M,N,Nt)
    int :: s0,A,L,r0,num_threads
    list of ints :: M,N,Nt
    """
    # set parameters in fortran module
    bm.bm_s0=input_p[0]
    bm.bm_a=input_p[1]
    bm.bm_l=input_p[2]
    bm.bm_r0=input_p[3]
    bm.numthreads=input_p[4]
    # extract values to time for
    m = input_p[5]
    n = input_p[6]
    nt = input_p[7]
    # create dictionaries to store results
    mTimes={"F90+OMP":[None]*len(m),"F90":[None]*len(m),"PYTHON":[None]*len(m)}
    nTimes={"F90+OMP":[None]*len(n),"F90":[None]*len(n),"PYTHON":[None]*len(n)}
    ntTimes={"F90+OMP":[None]*len(n),"F90":[None]*len(nt),"PYTHON":[None]*len(nt)}
    # time for m
    for v,i in enumerate(m): 
        print(f"Timing for M={i}...")   
        mTimes["F90+OMP"][v]=timer(bm.simulate2_omp,m=i,n=100,nt=100)
        mTimes["F90"][v]=timer(bm.simulate2_f90,m=i,n=100,nt=100)
        mTimes["PYTHON"][v]=timer(simulate2,M=i,N=100,Nt=100,A=input_p[1])
    # time for n
    for v,i in enumerate(n):  
        print(f"Timing for N={i}...")  
        nTimes["F90+OMP"][v]=timer(bm.simulate2_omp,n=i,m=100,nt=100)
        nTimes["F90"][v]=timer(bm.simulate2_f90,n=i,m=100,nt=100)
        nTimes["PYTHON"][v]=timer(simulate2,N=i,M=100,Nt=100,A=input_p[1]) 
    # time for nt
    for v,i in enumerate(nt):
        print(f"Timing for Nt={i}...")    
        ntTimes["F90+OMP"][v]=timer(bm.simulate2_omp,n=100,m=100,nt=i)
        ntTimes["F90"][v]=timer(bm.simulate2_f90,n=100,m=100,nt=i)
        ntTimes["PYTHON"][v]=timer(simulate2,N=100,M=100,Nt=i,A=input_p[1])   
    # increasing m figure       
    plt.figure()
    plt.plot(m,mTimes["F90+OMP"],label="F90+OMP")
    plt.plot(m,mTimes["F90"],label="F90")
    plt.plot(m,mTimes["PYTHON"],label="PYTHON")
    plt.legend()
    plt.title("Run time for increasing M and N=100 and Nt=100")
    plt.xlabel("M")
    plt.ylabel("Time (s)")
    plt.savefig("m_time_comparison.png")
    # increasing N figure
    plt.figure()
    plt.plot(n,nTimes["F90+OMP"],label="F90+OMP")
    plt.plot(n,nTimes["F90"],label="F90")
    plt.plot(n,nTimes["PYTHON"],label="PYTHON")
    plt.legend()
    plt.title("Run time for increasing N and M=100 and Nt=100")
    plt.xlabel("N")
    plt.ylabel("Time (s)")
    plt.savefig("n_time_comparison.png")
    # increasin nt figure
    plt.figure()
    plt.plot(nt,ntTimes["F90+OMP"],label="F90+OMP")
    plt.plot(nt,ntTimes["F90"],label="F90")
    plt.plot(nt,ntTimes["PYTHON"],label="PYTHON")
    plt.legend()
    plt.title("Run time for increasing Nt and N=100 and M=100")
    plt.xlabel("Nt")
    plt.ylabel("Time (s)")
    plt.savefig("nt_time_comparison.png")
    
    if display: # if display==True show
        plt.show()
    
    return mTimes,nTimes,ntTimes #Modify as needed

def timer(func,repeats=1,**kwargs):
    """
    func = function to time
    repeats = repeats to average over
    **kwargs = kwargs to be passed to func
    """
    l=[None]*repeats # store results
    for i in range(repeats): # time for repeats
        start = time.time()
        func(**kwargs)
        end = time.time()
        l[i] = end - start
    return np.mean(l) # reaturn mean time
    
    
def correlation(input_c=(None),display=False):
    """Compute and analyze temporal correlation function, C(tau)
    Modify the contents of the tuple, input, as needed
    When display is True, figures equivalent to those
    you are submitting should be displayed
    input_c=(s0,A,L,r0,num_threads,a,M)
    int :: s0,A,L,r0,num_threads
    """
    # set fortran module variables
    bm.bm_s0=input_c[0]
    bm.bm_a=input_c[1]
    bm.bm_l=input_c[2]
    bm.bm_r0=input_c[3]
    bm.numthreads=input_c[4]
    # set a,b and m
    a= input_c[5]
    b = 200 +a
    m = input_c[6]
    # compute correlation for tau 0 to 80
    c=bm.correlation(a,b,m)
    # plot
    x=np.linspace(0,80,81)
    plt.figure()
    plt.plot(x,c)
    plt.title(r"$C(\tau)$")
    plt.xlabel(r'$ \tau $')
    plt.ylabel(r"$C(\tau)$")
    plt.savefig("corr.png")
    if display: # if display==True show
        plt.show()
    return c #Modify as needed

    
def updatefig(i,x,y,line):
    """Updates figure each time function is called
    and returns new figure 'axes'
    """
    line.set_ydata(y[:,i])
    line.set_xdata(x[:,i])
    return line,    


def visualize(n=128,nt=500,s0=0.1,A=0.65,l=4,r0=1):
    """Generate an animation illustrating particle dynamics
    """
    # set variables in fortran
    bm.bm_s0=s0
    bm.bm_a=A
    bm.bm_l=l
    bm.bm_r0=r0
    bm.numthreads=2
    x,y = bm.allpositions(n=n,nt=nt) # faster to generate all values and index
    fig, ax=plt.subplots()
    # create plot
    line,=ax.plot(x[:,0],y[:,0],linewidth=0,markersize=8,marker='o')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"Animation for N={n}, s0={s0}, A={A}, l={l}, r0={r0}")
    # set limits to l
    ax.set_xlim(0,l)
    ax.set_ylim(0,l)
    # animate
    ani = animation.FuncAnimation(fig, updatefig,frames=30, repeat=False,blit=True,fargs=(x,y,line))
    ani.save("p3movie.mp4",writer="ffmpeg")
    return x,y #Modify as needed


if __name__ == '__main__':
    #Modify the code here so that it calls performance analyze and
    # generates the figures that you are submitting with your code
    input_p = (0.2,0.64,8,1,2,[100,200,300,400,500],[100,200,300,400,500],[100,250,500,750,1000])
    output_p = performance(input_p) #modify as needed

    input_c = (0.1,0.625,16,1,2,100,200)
    output_c = correlation(input_c)
    
    visualize()
