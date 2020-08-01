! This is a main program which can be used with the lrmodel module
! if and as you wish as you develop your codes.
! It reads the full set of 20000 images and labels from data.csv
! into the variable, p, which is then split into the
! images (xfull) and labels (yfull)
! Note that here the intial fitting parameters are sampled from
! an uniform distribution while a normal distribution should be used
! in your python code.
! You should not submit this code with your assignment.
! To compile: gfortran -o main.exe hw2_dev.f90 hw2_main.f90

program hw2main
  use lrmodel !brings in module variables x and y as well as the module subroutines
  implicit none
  integer, parameter :: n=784,dfull=20000, dtest=5000
  integer:: d=100,m=2
  integer :: i1, p(n+1,dfull)
  real(kind=8) :: xfull(n,dfull),xtest(n,dtest)
  real(kind=8), allocatable, dimension(:) :: fvec0,fvec !array of fitting parameters
  integer :: yfull(dfull),ytest(dtest) !Labels

  !read raw data from data.csv and store in p
  open(unit=12,file='data.csv')
  do i1=1,dfull
    read(12,*) p(:,i1)
    if (mod(i1,1000)==0) print *, 'read in image #', i1
  end do
  close(12)

  open(unit=13,file='p.dat')
  do i1=1,n
    write(13,*) p(i1,1)
  end do
  close(13)

  !Rearrange into input data, x,  and labels, y
  xfull = p(1:n,:)/255.d0
  yfull = p(n+1,:)
  yfull = mod(yfull,2)
  print *, 'yfull(1:4)',yfull(1:4) ! output first few labels
  xtest = xfull(:,dfull-dtest+1:dfull) ! set up test data (not used below)
  ytest = yfull(dfull-dtest+1:dfull)

  !Set, d training images---------------
  call data_init(n,d) !allocate module variables x and y
  lr_x = xfull(:,1:d) !set module variables
  lr_y = yfull(1:d)
  allocate(fvec0((m-1)*(n+1)),fvec((m-1)*(n+1)))
  call random_number(fvec0) !set initial fitting parameters
  fvec0 = fvec0-0.5d0


  deallocate(lr_x,lr_y)
end program hw2main
