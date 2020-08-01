!Final project part 2
!This file contains 1 module, 1 main program and 4 subroutines:
! params: module contain problem parameters and useful constants
! crickets: main program which reads in parameters from data.in
! 	calls simulate, and writes the computed results to output files
! simulate: subroutine for simulating coupled oscillator model
! 	using explicit-Euler time-marching and distributed-memory
! 	parallelization
! RHS: subroutine called by simulate, generates right-hand side
!		of oscillator model equations
! MPE_DECOMP1D: subroutine called by simulate and used to assign
!		oscillators to processes
! random_normal: subroutine called by main program and used to generate
!		natural frequencies, w

!-------------------------------------------------------------
module params
	implicit none
	real(kind=8), parameter :: pi = acos(-1.d0)
	complex(kind=8), parameter :: ii=cmplx(0.0,1.0) !ii = sqrt(-1)
    integer :: ntotal !total number of oscillators,
	real(kind=8) :: c,mu,sigma !coupling coefficient, mean, std for computing omega
    integer :: nlocal_min,ysize
	save
end module params
!-------------------------------

program crickets
    use mpi
    use params
    implicit none
    integer :: i1,j1
    integer :: nt !number of time steps
    real(kind=8) :: dt!time step
    integer :: myid, numprocs, ierr, istart, iend
    real(kind=8), allocatable, dimension(:) :: f0,w,f ! initial phases, frequencies, final phases
    real(kind=8), allocatable, dimension(:) :: r !synchronization parameter

 ! Initialize MPI
    call MPI_INIT(ierr)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, numprocs, ierr)

!gather input
    open(unit=10,file='data.in')
        read(10,*) ntotal !total number of oscillators
        read(10,*) nt !number of time steps
        read(10,*) dt !size of time step
        read(10,*) c ! coupling parameter
        read(10,*) sigma !standard deviation for omega calculation
    close(10)

    allocate(f0(ntotal),f(ntotal),w(ntotal),r(nt))


!generate initial phases
    call random_number(f0)
    f0 = f0*2.d0*pi


!generate frequencies
    mu = 1.d0
    call random_normal(ntotal,w)
    w = sigma*w+mu

!compute min(nlocal)
		nlocal_min = ntotal
		do i1 = 0,numprocs-1
			call mpe_decomp1d(ntotal,numprocs,i1,istart,iend)
			nlocal_min = min(iend-istart+1,nlocal_min)
		end do


!compute solution
    call simulate(MPI_COMM_WORLD,numprocs,ntotal,0.d0,f0,w,dt,nt,f,r)
    

!output solution (after collecting solution onto process 0 in simulate)
     call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierr)
     if (myid==0) then
        open(unit=11,file='theta.dat')
        do i1=1,ntotal
            write(11,*) f(i1)
        end do
        close(11)

        open(unit=12,file='R.dat')
        do i1=1,nt
	    		write(12,*) r(i1)
				end do
				close(12)
    	end if
    !can be loaded in python, e.g. theta=np.loadtxt('theta.dat')

    call MPI_FINALIZE(ierr)
end program crickets



subroutine simulate(comm,numprocs,n,t0,y0,w,dt,nt,y,r)
    !explicit Euler method, parallelized with mpi
    !input:
    !comm: MPI communicator
    !numprocs: total number of processes
    !n: number of oscillators
    !t0: initial time
    !y0: initial phases of oscillators
    !w: array of frequencies, omega_i
    !dt: time step
    !nt: number of time steps
    !output: y, final solution
    !r: synchronization parameter at each time step
    use mpi
    use params
    implicit none
    integer, intent (in) :: n,nt
    real(kind=8), dimension(n), intent(in) :: y0,w
    real(kind=8), intent(in) :: t0,dt
    real(kind=8), dimension(n), intent(out) :: y
		real(kind=8), dimension(nt), intent(out) :: r
    real(kind=8) :: t
    integer :: i1,k,istart,iend
    integer :: comm,myid,ierr,numprocs
		integer, allocatable, dimension(:) :: seed,ai
		real(kind=8), allocatable, dimension(:) ::  temp
		integer :: nseed,time
		!add other variables as needed
        real(kind=8),allocatable,dimension(:) ::  ylocal,f,z,Rpart
        integer :: sender, receiver, request,root
        integer, dimension(MPI_STATUS_SIZE) :: status
        complex(kind=8), dimension(nt) :: rcomp
        complex(kind=8) :: rlocal
    call MPI_COMM_RANK(comm, myid, ierr)
    print *, 'start simulate, myid=',myid

    !set initial conditions
    y = y0
    t = t0
    !generate decomposition and allocate sub-domain variables
    
    call mpe_decomp1d(size(y),numprocs,myid,istart,iend)
    print *, 'istart,iend,threadID=',istart,iend,myid

		!Set coupling ranges, ai
		allocate(ai(iend-istart+1),temp(iend-istart+1))
		call random_seed(size=nseed)
		call system_clock(time)
		allocate(seed(nseed))
		seed = myid!+time !remove the "+time" to generate same ai each run
		call random_seed(put=seed)
		call random_number(temp)
		ai = 1 + FLOOR((nlocal_min-1)*temp)
        

        !add code as needed
        deallocate(temp)
        ysize = iend-istart+1
        allocate(f(3*ysize-2),ylocal(ysize),z(ysize),Rpart(ysize))
        ylocal = y(istart:iend)
        if ((myid.NE.0).AND.(myid.NE.(numprocs-1))) then 
            f = y(istart-ysize+1:iend+ysize-1)
        else if (myid.EQ.0) then 
            f(:ysize-1) = y(n-ysize+2:)
            f(ysize:) = y(:2*ysize-1)
        else 
            f(:2*ysize-1) = y(n-2*ysize+2:)
            f(2*ysize:) = y(:ysize-1)
        end if



    !time marching
    do k = 1,nt
        
        call sumSin(f,ai,z)
        call RHS(n,t,w(nlocal_min:nlocal_min+ysize),z,Rpart)!add code here

        ylocal= ylocal + dt*Rpart !ylocal must be declared and defined, Rpart must be declared, and should be returned by RHS
        f(ysize:2*ysize-1) = ylocal

        if (myid<numprocs-1) then
            receiver = myid+1
        else
            receiver = 0
        end if
    
        if (myid>0) then
            sender = myid-1
        else
            sender = numprocs-1
        end if
    
        call MPI_ISEND(ylocal(2:ysize),1,MPI_DOUBLE_PRECISION,receiver,0,MPI_COMM_WORLD,request,ierr)
        call MPI_RECV(f(:ysize-1),1,MPI_DOUBLE_PRECISION,sender,MPI_ANY_TAG,MPI_COMM_WORLD,status,ierr)
    
        !call MPI_BARRIER(MPI_COMM_WORLD,ierr)
        if (myid<numprocs-1) then
            sender = myid+1
        else
            sender = 0
        end if
    
        if (myid>0) then
            receiver = myid-1
        else
            receiver = numprocs-1
        end if
    
        call MPI_ISEND(ylocal(:ysize-1),1,MPI_DOUBLE_PRECISION,receiver,0,MPI_COMM_WORLD,request,ierr)
        call MPI_RECV(f(2*ysize-1:),1,MPI_DOUBLE_PRECISION,sender,MPI_ANY_TAG,MPI_COMM_WORLD,status,ierr)
    
        !call MPI_BARRIER(MPI_COMM_WORLD,ierr)



        rlocal = sum(exp(ii*ylocal))
        call MPI_REDUCE(rcomp(k),rlocal,1,MPI_DOUBLE_COMPLEX,MPI_SUM,0,MPI_COMM_WORLD,ierr)
        !call MPI_BARRIER(MPI_COMM_WORLD,ierr)

    end do
    if (myid==0) r=abs(rcomp)/n
    
    print *, 'before collection',myid, maxval(abs(ylocal))
    
    !collect ylocal from each processor into y on myid=0
    call MPI_GATHER(ylocal,ysize,MPI_DOUBLE_PRECISION,y,ysize,MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierr)
  
    !if (myid==0) print *, 'finished',y
    if (myid==0) print *, 'finished',maxval(abs(y))
   
end subroutine simulate

!-------------------------
subroutine RHS(nn,t,w,f,Rpart)
    !called by simulate
    !Rpart = (1/dt)*(f(t+dt)-f(t))
    use params
    implicit none
    integer, intent(in) :: nn
    real(kind=8), intent(in) :: t
!dimensions of variables below must be added
    real(kind=8), dimension(ysize), intent(in) :: w
    real(kind=8), dimension(3*ysize-2), intent(in) :: f
    real(kind=8), dimension(ysize), intent(out) :: Rpart
    
    
    
        
    Rpart = w - f(ysize:2*ysize-1)*c/nn
!Add code to compute rhs


end subroutine RHS

subroutine sumSin(z,ai,outsum) 
    use params
    implicit NONE

    real(kind=8), dimension(3*ysize-2),intent(in) :: z
    integer, dimension(ysize),intent(in) :: ai
    real(kind=8), dimension(ysize),intent(out) :: outsum
    integer :: i1,j1
    outsum=0.d0
    do i1 = ysize,2*ysize -1
        do j1 = i1 - ai(i1-ysize+1),i1 + ai(i1-ysize+1)
            
            outsum(i1-ysize+1) = outsum(i1-ysize+1) + sin(z(i1)-z(j1))  
            
            
        end do
    end do 

    
end subroutine sumSin



!--------------------------------------------------------------------
!  (C) 2001 by Argonne National Laboratory.
!      See COPYRIGHT in online MPE documentation.
!  This file contains a routine for producing a decomposition of a 1-d array
!  when given a number of processors.  It may be used in "direct" product
!  decomposition.  The values returned assume a "global" domain in [1:n]
!
subroutine MPE_DECOMP1D( n, numprocs, myid, s, e )
    implicit none
    integer :: n, numprocs, myid, s, e
    integer :: nlocal
    integer :: deficit

    nlocal  = n / numprocs
    s       = myid * nlocal + 1
    deficit = mod(n,numprocs)
    s       = s + min(myid,deficit)
    if (myid .lt. deficit) then
        nlocal = nlocal + 1
    endif
    e = s + nlocal - 1
    if (e .gt. n .or. myid .eq. numprocs-1) e = n

end subroutine MPE_DECOMP1D

!--------------------------------------------------------------------

subroutine random_normal(n,rn)

! Adapted from the following Fortran 77 code
!      ALGORITHM 712, COLLECTED ALGORITHMS FROM ACM.
!      THIS WORK PUBLISHED IN TRANSACTIONS ON MATHEMATICAL SOFTWARE,
!      VOL. 18, NO. 4, DECEMBER, 1992, PP. 434-435.

!  The function random_normal() returns a normally distributed pseudo-random
!  number with zero mean and unit variance.

!  The algorithm uses the ratio of uniforms method of A.J. Kinderman
!  and J.F. Monahan augmented with quadratic bounding curves.

IMPLICIT NONE
integer, intent(in) :: n
real(kind=8), intent(out) :: rn(n)
!     Local variables
integer :: i1
REAL(kind=8)     :: s = 0.449871, t = -0.386595, a = 0.19600, b = 0.25472,           &
            r1 = 0.27597, r2 = 0.27846, u, v, x, y, q

!     Generate P = (u,v) uniform in rectangle enclosing acceptance region
do i1=1,n

DO
  CALL RANDOM_NUMBER(u)
  CALL RANDOM_NUMBER(v)
  v = 1.7156d0 * (v - 0.5d0)

!     Evaluate the quadratic form
  x = u - s
  y = ABS(v) - t
  q = x**2 + y*(a*y - b*x)

!     Accept P if inside inner ellipse
  IF (q < r1) EXIT
!     Reject P if outside outer ellipse
  IF (q > r2) CYCLE
!     Reject P if outside acceptance region
  IF (v**2 < -4.d0*LOG(u)*u**2) EXIT
END DO

!     Return ratio of P's coordinates as the normal deviate
rn(i1) = v/u
end do
RETURN


END subroutine random_normal
