!Final project part 1
!Alexander Pinches 01201653
!Module for flow simulations of liquid through tube
!This module contains a few module variables (see comments below)
!and four subroutines:
!jacobi: Uses jacobi iteration to compute solution
! to flow through tube
!sgisolve: To be completed. Use sgi method to
! compute flow through tube
!mvec: To be completed; matrix-vector multiplication z = Ay
!mtvec: To be completed; matrix-vector multiplication z = A^T y
module flow
    implicit none
    real(kind=8), parameter :: pi = acos(-1.d0)
    integer :: numthreads !number of threads used in parallel regions
    integer :: fl_kmax=10000 !max number of iterations
    real(kind=8) :: fl_tol=0.00000001d0 !convergence criterion
    real(kind=8), allocatable, dimension(:) :: fl_deltaw !|max change in w| each iteration
    real(kind=8) :: fl_s0=0.1d0 !deformation magnitude

contains
!-----------------------------------------------------
!Solve 2-d tube flow problem with Jacobi iteration
subroutine jacobi(n,w)
    !input  n: number of grid points (n+2 x n+2) grid
    !output w: final velocity field
    !Should also compute fl_deltaw(k): max(|w^k - w^k-1|)
    !A number of module variables can be set in advance.

    integer, intent(in) :: n
    real(kind=8), dimension(0:n+1,0:n+1), intent(out) :: w
    integer :: i1,j1,k1
    real(kind=8) :: del_r,del_t,del_r2,del_t2
    real(kind=8), dimension(0:n+1) :: s_bc,fac_bc
    real(kind=8), dimension(0:n+1,0:n+1) :: r,r2,t,RHS,w0,wnew,fac,fac2,facp,facm

    if (allocated(fl_deltaw)) then
      deallocate(fl_deltaw)
    end if
    allocate(fl_deltaw(fl_kmax))


    !grid--------------
    del_t = 0.5d0*pi/dble(n+1)
    del_r = 1.d0/dble(n+1)
    del_r2 = del_r**2
    del_t2 = del_t**2


    do i1=0,n+1
        r(i1,:) = i1*del_r
    end do

    do j1=0,n+1
        t(:,j1) = j1*del_t
    end do
    !-------------------

    !Update-equation factors------
    r2 = r**2
    fac = 0.5d0/(r2*del_t2 + del_r2)
    facp = r2*del_t2*fac*(1.d0+0.5d0*del_r/r) !alpha_p/gamma
    facm = r2*del_t2*fac*(1.d0-0.5d0*del_r/r) !alpha_m/gamma
    fac2 = del_r2 * fac !beta/gamma
    RHS = fac*(r2*del_r2*del_t2) !1/gamma
    !----------------------------

    !set initial condition/boundary deformation
    w0 = (1.d0-r2)/4.d0
    w = w0
    wnew = w0
    s_bc = fl_s0*exp(-10.d0*((t(0,:)-pi/2.d0)**2))/del_r
    fac_bc = s_bc/(1.d0+s_bc)


    !Jacobi iteration
    do k1=1,fl_kmax
        wnew(1:n,1:n) = RHS(1:n,1:n) + w(2:n+1,1:n)*facp(1:n,1:n) + w(0:n-1,1:n)*facm(1:n,1:n) + &
                                         (w(1:n,0:n-1) + w(1:n,2:n+1))*fac2(1:n,1:n)

        !Apply boundary conditions
        wnew(:,0) = wnew(:,1) !theta=0
        wnew(:,n+1) = wnew(:,n) !theta=pi/2
        wnew(0,:) = wnew(1,:) !r=0
        wnew(n+1,:) = wnew(n,:)*fac_bc !r=1s

        fl_deltaw(k1) = maxval(abs(wnew-w)) !compute relative error

        w=wnew    !update variable
        if (fl_deltaw(k1)<fl_tol) exit !check convergence criterion
        if (mod(k1,1000)==0) print *, k1,fl_deltaw(k1)
    end do

    print *, 'k,error=',k1,fl_deltaw(min(k1,fl_kmax))

end subroutine jacobi
!-----------------------------------------------------

!Solve 2-d tube flow problem with sgi method
subroutine sgisolve(n,w)
    !input  n: number of grid points (n+2 x n+2) grid
    !output w: final velocity field stored in a column vector
    !Should also compute fl_deltaw(k): max(|w^k - w^k-1|)
    !A number of module variables can be set in advance.
    use omp_lib 
    integer, intent(in) :: n
    real(kind=8), dimension((n+2)*(n+2)), intent(out) :: w
    real(kind=8) :: del_t,del_r
    !add other variables as needed
    real(kind=8), dimension(5*(n*n)+8*n+8) :: values
    integer, dimension(5*(n*n)+8*n+8) :: rows,columns
    real(kind=8), dimension((n+2)*(n+2)) :: b,z,xold,d,eold,enew,md
    real(kind=8), dimension(n+2) :: t,s_bc,fac_bc,r,r2,fac,fac2,facp,facm
    integer :: i1
    real(kind=8) :: k,mu,eolddot,del_r2,del_t2
    
    if (allocated(fl_deltaw)) then
        deallocate(fl_deltaw)
    end if
    allocate(fl_deltaw(fl_kmax))
    !$ call omp_set_num_threads(numthreads)
    !grid spacings------------
    del_t = 0.5d0*pi/dble(n+1)
    del_r = 1.d0/dble(n+1)
    del_r2 = del_r**2
    del_t2 = del_t**2

    do i1=1,n+2
        r(i1) = (i1-1)*del_r
        t(i1) = (i1-1)*del_t
    end do

    
    r2 = r**2
    fac = 0.5d0/(r2*del_t2 + del_r2)
    facp = fac*(1.d0/del_r2 + 0.5d0/(r*del_r)) !alpha_p/gamma
    facm = fac*(1.d0/del_r2 - 1.d0/(r*del_r)) !alpha_m/gamma
    fac2 = fac/(del_r2*r2) !beta/gamma

    s_bc = fl_s0*(exp(-10.d0*((t-pi/2.d0)**2)+exp(-10.d0*(t+pi/2.d0)**2)))/del_r
    fac_bc = s_bc/(1.d0+s_bc)
    b = 0.d0
    do i1 = 1,n
        b((i1*(n+2)+1):((i1+1)*(n+2)-1)) = -fac(2:(n+1))
    end do
    
    call asparseform(n,fac2,facp,facm,fac_bc,values,rows,columns)
    call mtvec_omp(n,values,rows,columns,b,d)
    
    eold = d
    xold = 0.d0
    
    do i1= 1,fl_kmax
        
        call mvec_omp(n,values,rows,columns,d,z)
        call mtvec_omp(n,values,rows,columns,z,md)
        eolddot = SUM(eold**2)

        k = eolddot/SUM(d*md)
        w = xold + k*d
        enew = eold - k*md
        
        mu = SUM(enew**2)/eolddot
        d = enew + mu*d


        fl_deltaw(i1) = maxval(abs(w-xold))
        if (fl_deltaw(i1).LE.fl_tol) exit 
        if (mod(i1,1000)==0) print *, i1,fl_deltaw(i1)
        xold = w
        eold = enew
       
        
    end do
    
    print *, 'k,error=',i1,fl_deltaw(min(i1,fl_kmax))


    
end subroutine sgisolve

!sparce a matrix
subroutine asparseform(n,fac2,facp,facm,fac_bc,values,rows,columns)
    !input n: grid is (n+2) x (n+2)
    ! fac,fac2,facp,facm,fac_bc: arrays that appear in
    !   discretized equations
    !output values,rows,columns: As values and their row and column index
    implicit none
    integer, intent(in) :: n
    real(kind=8), dimension(n+2), intent(in) :: fac2,facp,facm,fac_bc
    real(kind=8), dimension(5*(n*n)+8*n+8), intent(out) :: values
    integer, dimension(5*(n*n)+8*n+8), intent(out) :: rows,columns
    real(kind=8) :: delta_r,delta_tht
    real(kind=8), dimension(n+2) :: sOverR,sOverR1
    integer :: i1,j1,counter

    
    sOverR = fac_bc/(1.d0-fac_bc)
    sOverR1 = sOverR + 1.d0
    delta_r = dble(n+1)
    delta_tht = dble(n+1)*2.d0/pi
    counter = 1




    do i1 = 1,n+2
        values(counter) = - delta_r
        columns(counter) = i1
        rows(counter) = i1
        counter = counter + 1
        values(counter) = delta_r
        columns(counter) = i1+n+2
        rows(counter) = i1
        counter = counter + 1
    end do
    do i1=1,n
        values(counter) = delta_tht
        rows(counter) = i1*(n+2) + 1
        columns(counter) = i1*(n+2) + 2
        counter = counter + 1
        values(counter) = -delta_tht
        rows(counter) = i1*(n+2) + 1
        columns(counter) = i1*(n+2) + 1
        counter = counter + 1
        values(counter) = delta_tht
        rows(counter) = (i1+1)*(n+2) 
        columns(counter) = (i1+1)*(n+2)
        counter = counter + 1
        values(counter) = -delta_tht
        rows(counter) = (i1+1)*(n+2) 
        columns(counter) = (i1+1)*(n+2) - 1
        counter = counter + 1
    end do
    do i1 = 1,n
        do j1 = 1,n
            values(counter) = facm(i1+1)
            rows(counter) = i1*(n+2) + j1 + 1
            columns(counter) = (i1-1)*(n+2) + j1 + 1 
            counter = counter+1
            values(counter) = fac2(i1+1)
            rows(counter) = i1*(n+2) + j1 + 1
            columns(counter) = i1*(n+2) + j1 
            counter = counter+1
            values(counter) = -1.d0
            rows(counter) = i1*(n+2) + j1 + 1
            columns(counter) = i1*(n+2) + j1 + 1
            counter = counter+1
            values(counter) = fac2(i1+1)
            rows(counter) = i1*(n+2) + j1 + 1
            columns(counter) = i1*(n+2) + j1 + 2
            counter = counter+1
            values(counter) = facp(i1+1)
            rows(counter) = i1*(n+2) + j1 + 1
            columns(counter) = (i1+1)*(n+2) + j1 + 1
            counter = counter+1
            
        end do
    end do
    do i1 = 1,n+2
        values(counter) = sOverR1(i1)
        rows(counter) = (n+2)*(n+1) + i1 
        columns(counter) = (n+2)*(n+1) + i1 
        counter = counter + 1
        values(counter) = -sOverR(i1)
        rows(counter) = (n+2)*(n+1) + i1
        columns(counter) = (n+2)*n + i1 
        counter = counter + 1
    end do
    
end subroutine asparseform

!Compute matrix-vector multiplication, z = Ay
subroutine mvec(n,fac,fac2,facp,facm,fac_bc,y,z)
    !input n: grid is (n+2) x (n+2)
    ! fac,fac2,facp,facm,fac_bc: arrays that appear in
    !   discretized equations
    ! y: vector multipled by A
    !output z: result of multiplication Ay
    implicit none
    integer, intent(in) :: n
    real(kind=8), dimension(n+2), intent(in) :: fac,fac2,facp,facm,fac_bc
    real(kind=8), dimension((n+2)*(n+2)), intent(in) :: y
    real(kind=8), dimension((n+2)*(n+2)), intent(out) :: z
    !add other variables as needed
    real(kind=8), dimension(5*(n*n)+8*n+8) :: values
    integer, dimension(5*(n*n)+8*n+8) :: rows,columns
    integer :: i1

    call asparseform(n,fac2,facp,facm,fac_bc,values,rows,columns)

    z = 0.d0
    do i1 = 1, 5*(n*n)+8*n+8
        z(rows(i1)) = z(rows(i1)) + y(columns(i1))*values(i1)
    end do
    

    
    


end subroutine mvec

!omp mvec
subroutine mvec_omp(n,values,rows,columns,y,z)
    !input n: grid is (n+2) x (n+2)
    ! values,rows,columns: sparce form of A
    ! y: vector multipled by A
    !output z: result of multiplication Ay
    implicit none
    integer, intent(in) :: n
    real(kind=8), dimension((n+2)*(n+2)), intent(in) :: y
    real(kind=8), dimension(5*(n*n)+8*n+8), intent(in) :: values
    integer, dimension(5*(n*n)+8*n+8), intent(in) :: rows,columns
    real(kind=8), dimension((n+2)*(n+2)), intent(out) :: z
    integer :: i1

    

    z = 0.d0
    !$ call omp_set_num_threads(numthreads)
    !$OMP parallel do reduction(+:z)
    do i1 = 1, 5*(n*n)+8*n+8
        z(rows(i1)) = z(rows(i1)) + y(columns(i1))*values(i1)
    end do
    !$OMP end parallel do
    

    
    


end subroutine mvec_omp

!Omp Compute matrix-vector multiplication, z = A^T y
subroutine mtvec_omp(n,values,rows,columns,y,z)
    !input n: grid is (n+2) x (n+2)
    ! values,rows,columns: Of A in sparce form
    ! y: vector multipled by A^T
    !output z: result of multiplication A^T y
    implicit none
    integer, intent(in) :: n
    real(kind=8), dimension((n+2)*(n+2)), intent(in) :: y
    real(kind=8), dimension(5*(n*n)+8*n+8), intent(in) :: values
    integer, dimension(5*(n*n)+8*n+8), intent(in) :: rows,columns
    real(kind=8), dimension((n+2)*(n+2)), intent(out) :: z
    integer :: i1


    z = 0.d0
    !$ call omp_set_num_threads(numthreads)
    !$OMP parallel do reduction(+:z)
    do i1 = 1, 5*(n*n)+8*n+8
        z(columns(i1)) = z(columns(i1)) + y(rows(i1))*values(i1)
    end do
    !$OMP end parallel do

end subroutine mtvec_omp

!Compute matrix-vector multiplication, z = A^T y
subroutine mtvec(n,fac,fac2,facp,facm,fac_bc,y,z)
    !input n: grid is (n+2) x (n+2)
    ! fac,fac2,facp,facm,fac_bc: arrays that appear in
    !   discretized equations
    ! y: vector multipled by A^T
    !output z: result of multiplication A^T y
    implicit none
    integer, intent(in) :: n
    real(kind=8), dimension(n+2), intent(in) :: fac,fac2,facp,facm,fac_bc
    real(kind=8), dimension((n+2)*(n+2)), intent(in) :: y
    real(kind=8), dimension((n+2)*(n+2)), intent(out) :: z
    !add other variables as needed
    real(kind=8), dimension(5*(n*n)+8*n+8) :: values
    integer, dimension(5*(n*n)+8*n+8) :: rows,columns
    integer :: i1

    call asparseform(n,fac2,facp,facm,fac_bc,values,rows,columns)

    z = 0.d0

    do i1 = 1, 5*(n*n)+8*n+8
        z(columns(i1)) = z(columns(i1)) + y(rows(i1))*values(i1)
    end do


end subroutine mtvec







end module flow
