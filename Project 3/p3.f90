!MATH 96012 Project 3
! Alexander John Pinches CID:01201653
!This module contains four module variables and two subroutines;
!one of these routines must be developed for this assignment.
!Module variables--
! bm_l, bm_s0, bm_r0, bm_a: the parameters l, s0, r0, and a in the particle dynamics model
! numthreads: The number of threads that should be used in parallel regions within simulate2_omp
!
!Module routines---
! simulate2_f90: Simulate particle dynamics over m trials. Return: all x-y positions at final time
! and alpha at nt+1 times averaged across the m trials.
! simulate2_omp: Same input/output functionality as simulate2.f90 but parallelized with OpenMP

module bmotion
  implicit none
  integer :: numthreads
  real(kind=8) :: bm_l,bm_s0,bm_r0,bm_a
  real(kind=8), parameter :: pi = acos(-1.d0)
  complex(kind=8), parameter :: ii = complex(0.d0,1.d0)
contains

!Compute m particle dynamics simulations using the parameters,bm_l,bm_s0,bm_r0,bm_a.
!Input:
!m: number of simulations
!n: number of particles
!nt: number of time steps
!Output:
! x: x-positions at final time step for all m trials
! y: y-positions at final time step for all m trials
! alpha_ave: alignment parameter at each time step (including initial condition)
! averaged across the m trials
subroutine simulate2_f90(m,n,nt,x,y,alpha_ave)
  implicit none
  integer, intent(in) :: m,n,nt
  real(kind=8), dimension(m,n), intent(out) :: x,y
  real(kind=8), dimension(nt+1), intent(out) :: alpha_ave
  integer :: i1,j1,k1
  real(kind=8), dimension(m,n) :: nn !neighbors
  real(kind=8) :: r0sq !r0^2
  real(kind=8), dimension(m,n) :: phi_init,r_init,theta !used for initial conditions
  real(kind=8), dimension(m,n,n) :: dist2 !distance squared
  real(kind=8), allocatable, dimension(:,:,:) :: temp
  complex(kind=8), dimension(m,n) :: phase
  complex(kind=8), dimension(m,n,nt+1) :: exp_theta,AexpR




!---Set initial condition and initialize variables----
  allocate(temp(m,n,nt+1))
  call random_number(phi_init)
  call random_number(r_init)
  call random_number(theta)
  call random_number(temp)
  phi_init = phi_init*(2.d0*pi)
  r_init = sqrt(r_init)
  theta = theta*(2.d0*pi) !initial direction of motion
  x = r_init*cos(phi_init)+0.5d0*bm_l !initial positions
  y = r_init*sin(phi_init)+0.5d0*bm_l

  alpha_ave=0.d0
  r0sq = bm_r0*bm_r0
  exp_theta(:,:,1) = exp(ii*theta)
  AexpR = bm_a*exp(ii*temp*2.d0*pi) !noise term
  deallocate(temp)
  nn = 0

!----------------------------------------------
  !Time marching
  do i1 = 2,nt+1

    phase=0.d0
    dist2 = 0.d0

    !Compute distances
    do j1=1,n-1
      do k1 = j1+1,n
        dist2(:,j1,k1) = (x(:,j1)-x(:,k1))**2 + (y(:,j1)-y(:,k1))**2
        where (dist2(:,j1,k1)>r0sq)
          dist2(:,j1,k1)=0
        elsewhere
          dist2(:,j1,k1)=1
        end where
       dist2(:,k1,j1) =dist2(:,j1,k1)
      end do
    end do

    nn = sum(dist2,dim=3)+1

    !Update phase
    phase =  exp_theta(:,:,i1-1) +nn*AexpR(:,:,i1-1)
    do j1=1,m
      phase(j1,:) = phase(j1,:) + matmul(dist2(j1,:,:),exp_theta(j1,:,i1-1))
    end do

    !Update Theta
    theta = atan2(aimag(phase),real(phase))

    !Update X,Y
    x = x + bm_s0*cos(theta)
    y = y + bm_s0*sin(theta)

    x = modulo(x,bm_l)
    y = modulo(y,bm_l)

    exp_theta(:,:,i1) = exp(ii*theta)

  end do

  alpha_ave = (1.d0/dble(m*n))*sum(abs(sum(exp_theta,dim=2)),dim=1)


end subroutine simulate2_f90


!Same functionality as simulate2_f90, but parallelized with OpenMP
!Parallel regions should use numthreads threads.
!Compute m particle dynamics simulations using the parameters,bm_l,bm_s0,bm_r0,bm_a.
!Same functionality as simulate2_f90, but parallelized with OpenMP
!Parallel regions should use numthreads threads.
!Input:
!m: number of simulations
!n: number of particles
!nt: number of time steps
!Output:
! x: x-positions at final time step for all m trials
! y: y-positions at final time step for all m trials
! alpha_ave: alignment parameter at each time step (including initial condition)
! averaged across the m trials

! We assume n and m are large so any loop over either of these we should seek to parallise
! with everything inside the loop vectorised. Any sums can also be decomposed to loops. We 
! know we can't parallise over nt and we arent assuming this is large for the main loop so we parallise the do loops inside this. 
! For the nested loop in finding distances we make sure we arent having threads writing to the
! same or wrong parts of the array by making k1 private. Also using reduction for sums and calling 
! parallel do as few times as possible as forking takes time.
subroutine simulate2_omp(m,n,nt,x,y,alpha_ave)
  use omp_lib
  implicit none
  integer, intent(in) :: m,n,nt
  real(kind=8), dimension(m,n), intent(out) :: x,y
  real(kind=8), dimension(nt+1), intent(out) :: alpha_ave
  integer :: i1,j1,k1
  real(kind=8), dimension(m,n) :: phi_init,r_init,theta !used for initial conditions
  !Note: use allocatable arrays where possible to avoid OpenMP memory issues
  real(kind=8), allocatable, dimension(:,:,:) :: dist2 !distance squared
  real(kind=8), allocatable, dimension(:,:,:) :: temp
  complex(kind=8), allocatable,dimension(:,:) :: phase,temp2
  complex(kind=8), allocatable,dimension(:,:,:) :: exp_theta,AexpR
  real(kind=8), allocatable,dimension(:,:) :: nn
  real(kind=8) :: r0sq
!---Set initial condition and initialize variables (does not need to be parallelized)----
  call random_number(phi_init)
  call random_number(r_init)
  call random_number(theta)
  phi_init = phi_init*(2.d0*pi)
  r_init = sqrt(r_init)
  theta = theta*(2.d0*pi) !initial direction of motion
  x = r_init*cos(phi_init)+0.5d0*bm_l !initial positions
  y = r_init*sin(phi_init)+0.5d0*bm_l
!-------------------------------------------------
  

 !$ call omp_set_num_threads(numthreads)
  
  r0sq = bm_r0*bm_r0
  allocate(exp_theta(m,n,nt+1))
  !$OMP parallel do
  do i1=1,m
    exp_theta(i1,:,1)=exp(ii*theta(i1,:))
  end do
  !$OMP end parallel do
  allocate(temp(m,n,nt+1))
  call random_number(temp)
  allocate(AexpR(m,n,nt+1))
  !$OMP parallel do
  do i1=1,m
    AexpR(i1,:,:) = bm_a*exp(ii*temp(i1,:,:)*2.d0*pi)
  end do
  !$OMP end parallel do
  
  deallocate(temp)
  allocate(dist2(m,n,n))
  allocate(phase(m,n))
  allocate(nn(m,n))
!----------------------------------------------
  !Time marching
  
  do i1 = 2,nt+1
    !Compute distances
    
    dist2 = 0.d0
    !$OMP parallel do private(k1)
    do j1=1,n-1
      do k1 = j1+1,n
        dist2(:,k1,j1) = (x(:,j1)-x(:,k1))**2 + (y(:,j1)-y(:,k1))**2
        where (dist2(:,k1,j1)>r0sq)
          dist2(:,k1,j1)=0.d0
        elsewhere
          dist2(:,k1,j1)=1.d0
        end where
      end do
    end do
    !$OMP end parallel do
    
   
    
    
    nn = 1.d0
    !$OMP parallel do reduction(+:nn)
    do k1=1,n
      nn = nn + dist2(:,:,k1) 
    end do
    !$OMP end parallel do
    
    !Update phase
    
    phase=0.d0
    !$OMP parallel do 
    do j1=1,m
      phase(j1,:) = exp_theta(j1,:,i1-1) +nn(j1,:)*AexpR(j1,:,i1-1) + matmul(dist2(j1,:,:),exp_theta(j1,:,i1-1))
      theta(j1,:) = atan2(aimag(phase(j1,:)),real(phase(j1,:)))
      x(j1,:)=modulo(x(j1,:) + bm_s0*cos(theta(j1,:)),bm_l)
      y(j1,:)=modulo(y(j1,:) + bm_s0*sin(theta(j1,:)),bm_l)
      exp_theta(j1,:,i1) = exp(ii*theta(j1,:))
      
    end do
    !$OMP end parallel do
    
    
    
  end do
  ! deallocate
  deallocate(nn)
  deallocate(phase)
  deallocate(AexpR)
  ! calculate alpha
  allocate(temp2(m,nt+1))
  temp2=0.d0
  !$OMP parallel do reduction(+:temp2)
  do i1=1,n
    temp2 = temp2 + exp_theta(:,i1,:)
  end do
  !$OMP end parallel do 
  deallocate(exp_theta)
  alpha_ave=0.d0
  !$OMP parallel do reduction(+:alpha_ave)
  do i1=1,m
    alpha_ave = alpha_ave + (1.d0/dble(m*n))*abs(temp2(i1,:))
  end do
  !$OMP end parallel do 
  

end subroutine simulate2_omp

subroutine allPositions(n,nt,x,y)
  use omp_lib
  implicit none
  integer :: n,nt
  real(kind=8), dimension(n,nt+1), intent(out) :: x,y
  integer :: i1,j1,k1
  real(kind=8), allocatable, dimension(:) :: phi_init,r_init,theta !used for initial conditions
  !Note: use allocatable arrays where possible to avoid OpenMP memory issues
  real(kind=8), allocatable, dimension(:,:) :: dist2,temp !distance squared
  complex(kind=8), allocatable,dimension(:) :: phase
  complex(kind=8), allocatable,dimension(:,:) :: exp_theta,AexpR
  real(kind=8), allocatable,dimension(:) :: nn
  real(kind=8) :: r0sq
!---Set initial condition and initialize variables (does not need to be parallelized)----
  allocate(phi_init(n))
  allocate(r_init(n))
  allocate(theta(n))
  call random_number(phi_init)
  call random_number(r_init)
  call random_number(theta)
  phi_init = phi_init*(2.d0*pi)
  r_init = sqrt(r_init)
  theta = theta*(2.d0*pi) !initial direction of motion
  x(:,1) = r_init*cos(phi_init)+0.5d0*bm_l !initial positions
  y(:,1) = r_init*sin(phi_init)+0.5d0*bm_l
  deallocate(r_init)
  deallocate(phi_init)
!-------------------------------------------------
  !$ call omp_set_num_threads(numthreads)
  
  r0sq = bm_r0*bm_r0
  allocate(exp_theta(n,nt+1))
  !$OMP parallel do
  do i1=1,n
    exp_theta(i1,1)=exp(ii*theta(i1))
  end do
  !$OMP end parallel do
  allocate(temp(n,nt+1))
  call random_number(temp)
  allocate(AexpR(n,nt+1))

  !$OMP parallel do
  do i1=1,n
    AexpR(i1,:) = bm_a*exp(ii*temp(i1,:)*2.d0*pi)
  end do
  !$OMP end parallel do

  deallocate(temp)
  allocate(dist2(n,n))
  allocate(phase(n))
  allocate(nn(n))
!----------------------------------------------
  !Time marching
  
  do i1 = 2,nt+1
    !Compute distances
    
    dist2 = 0.d0
    !$OMP parallel do private(k1)
    do j1=1,n-1
      do k1 = j1+1,n
        dist2(k1,j1) = (x(j1,i1-1)-x(k1,i1-1))**2 + (y(j1,i1-1)-y(k1,i1-1))**2
        if (dist2(k1,j1)>r0sq) then 
          dist2(k1,j1)=0.d0
        else 
          dist2(k1,j1)=1.d0
        end if
      end do
    end do
    !$OMP end parallel do
    

    
    
    nn = 1.d0
    !$OMP parallel do reduction(+:nn)
    do k1=1,n
      nn = nn + dist2(:,k1) 
    end do
    !$OMP end parallel do
    
    !Update phase

    phase=0.d0
    !$OMP parallel do
    do j1=1,n
      phase(j1) = exp_theta(j1,i1-1) +nn(j1)*AexpR(j1,i1-1)
    end do
    !$OMP end parallel do

    phase = phase + matmul(dist2,exp_theta(:,i1-1))
    ! calculate new positions
    !$OMP parallel do
    do j1=1,n
      theta(j1) = atan2(aimag(phase(j1)),real(phase(j1)))
      x(j1,i1)=modulo(x(j1,i1-1) + bm_s0*cos(theta(j1)),bm_l)
      y(j1,i1)=modulo(y(j1,i1-1) + bm_s0*sin(theta(j1)),bm_l)
      exp_theta(j1,i1) = exp(ii*theta(j1))
    end do
    !$OMP end parallel do
    
    
    
    
  end do
 
end subroutine allPositions

subroutine correlation(a,b,m,c)
  use omp_lib
  implicit none
  integer :: a,b,m,nt,i1
  real(kind=8), dimension(81), intent(out) :: c
  real(kind=8), allocatable, dimension(:) :: alpha_ave
  real(kind=8) :: last
  real(kind=8), allocatable,dimension(:,:) :: x,y

  nt= a + b + 80
  allocate(alpha_ave(nt+1))
  allocate(x(m,400))
  allocate(y(m,400))
  ! get alphas using simulate2_omp
  call simulate2_omp(m=m,n=400,nt=nt,x=x,y=y,alpha_ave=alpha_ave)
  last=0.d0
  !$ call omp_set_num_threads(numthreads)
  ! calculate last term
  !$OMP parallel do reduction(+:last)
  do i1=a,b
    last = last + (alpha_ave(i1)/dble(b-a))**2
  end do
  !$OMP end parallel do
  ! calculate c for each tau
  !$OMP parallel do
  do i1 = 1,81
    c(i1) = SUM(alpha_ave(i1+a-1:b+i1-1)*alpha_ave(a:b))/dble(b-a) + last
  end do
  !$OMP end parallel do
end subroutine correlation
end module bmotion
