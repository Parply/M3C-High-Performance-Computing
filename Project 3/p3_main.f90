! This is a main program which can be used with the bmotion module
! if and as you wish as you develop your codes.
! It reads simulation parameters from a text file, data.in, which must be created.
!
! The subroutine simulate2_f90 is called, and output is written to the text files,
! x.txt,y.txt, and alpha_ave.txt which can be read in Python using np.loadtxt (see below)
!
! You should not submit this code with your assignment.
! To compile: gfortran -fopenmp -O3 -o main.exe p3_dev.f90 p3_main.f90
program p3_main
  use bmotion
  implicit none
  integer :: m,n,nt,i1
  real(kind=8), allocatable, dimension(:) :: alpha_ave
  real(kind=8), allocatable, dimension(:,:) :: x,y

  !Read in problem parameters from text file, data.in
  open(unit=11,file='data.in')
  read(11,*) m !number of trials
  read(11,*) n !number of particles
  read(11,*) nt !number of time steps
  read(11,*) bm_l
  read(11,*) bm_s0
  read(11,*) bm_r0
  read(11,*) bm_a
  read(11,*) numthreads !not used below
  close(11)

  allocate(alpha_ave(nt+1),x(m,n),y(m,n))

  call simulate2_omp(m,n,nt,x,y,alpha_ave)


  !load in python using alpha_ave = np.loadtxt('alpha_ave.txt')
  open(unit=12,file='alpha_ave.txt')
  do i1=1,nt+1
    write(12,*) alpha_ave(i1)
  end do
  close(12)

  !load in python using: x = np.loadtxt('x.txt')

  open(unit=13,file='x.txt')
  open(unit=14,file='y.txt')


  do i1=1,m
      write(13,'(10000E28.16)') x(i1,:)
      write(14,'(10000E28.16)') y(i1,:)
  end do

  close(13)
  close(14)

end program p3_main
