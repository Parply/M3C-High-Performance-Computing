!MATH96012 Project 2
! Alexander John Pinches CID:01201653
!This module contains two module variables and three subroutines;
!two of these routines must be developed for this assignment.
!Module variables--
! lr_x: training images, typically n x d with n=784 and d<=15000
! lr_y: labels for training images, d-element array containing 0s and 1s
!   corresponding to images of even and odd integers, respectively.
!lr_lambda: l2-penalty parameter, should be set to be >=0.
!Module routines---
! data_init: allocate lr_x and lr_y using input variables n and d. May be used if/as needed.
! clrmodel: compute cost function and gradient using classical logistic
!   regression model (CLR) with lr_x, lr_y, and
!   fitting parameters provided as input
! mlrmodel: compute cost function and gradient using MLR model with m classes
!   and with lr_x, lr_y, and fitting parameters provided as input
    

module lrmodel
  implicit none
  real(kind=8), allocatable, dimension(:,:) :: lr_x
  integer, allocatable, dimension(:) :: lr_y
  real(kind=8) :: lr_lambda !penalty parameter

    contains
    

!---allocate lr_x and lr_y deallocating first if needed (used by p2_main)--
! ---Use if needed---
subroutine data_init(n,d)
  implicit none
  integer, intent(in) :: n,d
  if (allocated(lr_x)) deallocate(lr_x)
  if (allocated(lr_y)) deallocate(lr_y)
  allocate(lr_x(n,d),lr_y(d))
end subroutine data_init


!Compute cost function and its gradient for CLR model
!for d images (in lr_x) and d labels (in lr_y) along with the
!fitting parameters provided as input in fvec.
!The weight vector, w, corresponds to fvec(1:n) and
!the bias, b, is stored in fvec(n+1)
!Similarly, the elements of dc/dw should be stored in cgrad(1:n)
!and dc/db should be stored in cgrad(n+1)
!Note: lr_x and lr_y must be allocated and set before calling this subroutine.
subroutine clrmodel(fvec,n,d,c,cgrad)
  implicit none
  integer, intent(in) :: n,d !training data sizes
  real(kind=8), dimension(n+1), intent(in) :: fvec !fitting parameters
  real(kind=8), intent(out) :: c !cost
  real(kind=8), dimension(n+1), intent(out) :: cgrad !gradient of cost
  !Declare other variables as needed
    real(kind=8), dimension(2,d):: z
    real(kind=8), dimension(2,d) :: a
    real(kind=8), dimension(d) :: sum_z

  !Add code to compute c and cgrad
    
    z(1,:) = 0.0d0

    z(2,:) = MATMUL(fvec(:n),lr_x) + fvec(n+1)
    sum_z = SUM(EXP(z),DIM=1)
		  

    a = EXP(z)/RESHAPE(sum_z,SHAPE=[2,d],PAD=sum_z)


	  
    cgrad(n+1) = SUM( - (1.0d0-lr_y)*(1.0d0 + a(1,:)) + lr_y* a(1,:))


    cgrad = - (1.0d0-lr_y)*MATMUL(lr_x,1.0d0-a(1,:)) + lr_y*MATMUL(lr_x,a(1,:))+2.0d0*lr_lambda*fvec(:n)

	 
    c = -SUM(lr_y*LOG(a(1,:)+1.0d-12) + (1.0d0 - lr_y)*LOG(1.0d0-a(1,:)+1.0d-12))  + lr_lambda*SUM(fvec(:n)**2)
end subroutine clrmodel


!!Compute cost function and its gradient for MLR model
!for d images (in lr_x) and d labels (in lr_y) along with the
!fitting parameters provided as input in fvec. The labels are integers
! between 0 and m-1.
!fvec contains the elements of the weight matrix w and the bias vector, b
! Code has been provided below to "unpack" fvec
!The elements of dc/dw and dc/db should be stored in cgrad
!and should be "packed" in the same order that fvec was unpacked.
!Note: lr_x and lr_y must be allocated and set before calling this subroutine.
subroutine mlrmodel(fvec,n,d,m,c,cgrad)
  implicit none
  integer, intent(in) :: n,d,m !training data sizes and number of classes
  real(kind=8), dimension((m-1)*(n+1)), intent(in) :: fvec !fitting parameters
  real(kind=8), intent(out) :: c !cost
  real(kind=8), dimension((m-1)*(n+1)), intent(out) :: cgrad !gradient of cost
  integer :: i1,j1
  real(kind=8), dimension(m-1,n) :: w
  real(kind=8), dimension(m-1) :: b
  !Declare other variables as needed
    real(kind=8), dimension(m,d):: z
    real(kind=8), dimension(m,d) :: a
    real(kind=8), dimension(d) :: sum_z
    if (.not. allocated(lr_x)) allocate(lr_x(n,d))
    if (.not. allocated(lr_y)) allocate(lr_y(d))
  !unpack fitting parameters (use if needed)
  do i1=1,n
    j1 = (i1-1)*(m-1)+1
    w(:,i1) = fvec(j1:j1+m-2) !weight matrix
  end do
  b = fvec((m-1)*n+1:(m-1)*(n+1)) !bias vector


  !Add code to compute c and cgrad
  !Add code to compute c and cgrad
    !call data_init(n,d)
    z(1,:) = 0.0d0
    z(2:,:) = MATMUL(w,lr_x) + RESHAPE(b,[m-1,d],PAD=b)
    sum_z = SUM(EXP(z),DIM=1)
    a = EXP(z)/RESHAPE(sum_z,SHAPE=[m,d],PAD=sum_z)

    do i1=1,m-1
        do j1 =1,n
            if (lr_y(j1).EQ.i1) then
            cgrad(i1-1 +(m-1)*n) = cgrad(i1 +(m-1)*n)+SUM(a(i1+1,:)-1.0d0)
            else 
            cgrad(i1-1 +(m-1)*n) = cgrad(i1 +(m-1)*n)+SUM(a(lr_y(j1)+1,:))
            end if
        end do
    end do
    do i1 = 1,m-1
        do j1=1,n
            if (lr_y(j1).EQ.i1) then
            cgrad((m-1)*(j1-1)+i1) = cgrad((m-1)*(j1-1)+i1) - SUM(lr_x(j1,:)*(1.0d0-a(i1+1,:))) + 2.0d0*lr_lambda*w(i1,j1)
            else
            cgrad((m-1)*(j1-1)+i1) = cgrad((m-1)*(j1-1)+i1) + SUM(lr_x(j1,:)*a(lr_y(j1)+1,:)) + 2.0d0*lr_lambda*w(i1,j1)
            end if
        end do
    end do
	   
    c = - SUM(SUM(LOG(a+1.0d-12),DIM=1)) + lr_lambda*SUM(SUM(w**2,DIM=1))


end subroutine mlrmodel



end module lrmodel
