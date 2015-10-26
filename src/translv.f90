SUBROUTINE translv

!-----------------------------------------------------------------------
!
! Solution driver. Contains the time and outer loops. Calls for outer
! iteration work. Checks convergence and handles eventual output.
!
!-----------------------------------------------------------------------

  USE global_module, ONLY: i_knd, r_knd, ounit, zero, half, one, two

  USE plib_module, ONLY: glmax, comm_snap, iproc, root, thread_num,    &
    ichunk, do_nested

  USE geom_module, ONLY: geom_alloc, geom_dealloc, dinv, param_calc,   &
    nx, ny_gl, nz_gl, diag_setup, hi, hj, hk, dx, dy, dz

  USE sn_module, ONLY: nang, noct, mu, eta, xi, cmom, ec, w, nmom, lma

  USE data_module, ONLY: ng, v, vdelt, mat, sigt, siga, slgg, src_opt, &
    qim, nmat, qi

  USE control_module, ONLY: nsteps, timedep, dt, oitm, otrdone,        &
    control_alloc, control_dealloc, dfmxo, it_det, iitm, epsi, tolr

  USE utils_module, ONLY: print_error, stop_run

  USE solvar_module, ONLY: solvar_alloc, ptr_in, ptr_out, t_xs, a_xs,  &
    s_xs, flux, fluxm, q2grp

  USE expxs_module, ONLY: expxs_reg, expxs_slgg

  USE outer_module, ONLY: outer

  USE time_module, ONLY: tslv, wtime, tgrind, tparam,                  &
    ext_copy_time, ext_sweep_time, ext_reduc_time

  IMPLICIT NONE
!_______________________________________________________________________
!
! Local variables
!_______________________________________________________________________

  CHARACTER(LEN=1) :: star='*'

  CHARACTER(LEN=64) :: error

  INTEGER(i_knd) :: cy, otno, ierr, g, i, tot_iits, cy_iits, out_iits, o

  REAL(r_knd) :: sf, time, t1, t2, t3, t4, t5, t6, t7, tmp, ft_start, ft_end

  REAL(r_knd), DIMENSION(:,:,:,:,:,:), POINTER :: ptr_tmp
!_______________________________________________________________________
!
!   Local memory for the EXT sweep result
!_______________________________________________________________________

    REAL(r_knd) :: ext_first_copy_tic, ext_first_copy_toc
    REAL(r_knd) :: ext_update_tic, ext_update_toc
    REAL(r_knd), DIMENSION(:,:,:,:,:,:), POINTER :: ext_angular_flux
    REAL(r_knd), DIMENSION(:,:,:,:), POINTER :: scalar_flux
    REAL(r_knd), DIMENSION(:,:,:,:,:), POINTER :: scalar_flux_moments
    ALLOCATE( ext_angular_flux(nang,nx,ny_gl,nz_gl,noct,ng) )
    ALLOCATE( scalar_flux(nx,ny_gl,nz_gl,ng) )
    ALLOCATE( scalar_flux_moments((cmom-1),nx,ny_gl,nz_gl,ng) )
!_______________________________________________________________________
!
! Call for data allocations. Some allocations depend on the problem
! type being requested.
!_______________________________________________________________________

  CALL wtime ( t1 )

  ierr = 0
  error = ' '

  CALL geom_alloc ( nang, ng, ierr )
  CALL glmax ( ierr, comm_snap )
  IF ( ierr /= 0 ) THEN
    error = '***ERROR: GEOM_ALLOC: Allocation error of sweep parameters'
    CALL print_error ( ounit, error )
    CALL stop_run ( 3, 0, 0 )
  END IF

  CALL solvar_alloc ( ierr )
  CALL glmax ( ierr, comm_snap )
  IF ( ierr /= 0 ) THEN
    error = '***ERROR: SOLVAR_ALLOC: Allocation error of solution ' // &
            'arrays'
    CALL print_error ( ounit, error )
    CALL stop_run ( 3, 1, 0 )
  END IF

  CALL control_alloc ( ng, ierr )
  CALL glmax ( ierr, comm_snap )
  IF ( ierr /= 0 ) THEN
    error = '***ERROR: CONTROL_ALLOC: Allocation error of control ' // &
      'arrays'
    CALL print_error ( ounit, error )
    CALL stop_run ( 3, 2, 0 )
  END IF
!_______________________________________________________________________
!
! Call for setup of the mini-KBA diagonal map
!_______________________________________________________________________

  CALL diag_setup ( do_nested, ichunk, ierr )
  CALL glmax ( ierr, comm_snap )
  IF ( ierr /= 0 ) THEN
    error = '***ERROR: DIAG_SETUP: Allocation error of diag type array'
    CALL print_error ( ounit, error )
    CALL stop_run ( 3, 3, 0 )
  END IF

  CALL wtime ( t2 )
  tparam = tparam + t2 - t1

!_______________________________________________________________________
!
!   Initialise the problem parameters and allocate memory
!_______________________________________________________________________

  CALL wtime ( ext_first_copy_tic )

  CALL ext_initialise_parameters ( nx, ny_gl, nz_gl, ng, nang, noct, &
      cmom, nmom, ichunk, dx, dy, dz, dt, nmat, nsteps, oitm, iitm, epsi, tolr )

  !CALL ext_initialise_memory ( mu, eta, xi, ec, w, v, sigt, mat, &
  !    qi, slgg, lma, q2grp)

  CALL wtime ( ext_first_copy_toc )

  WRITE ( *, 212 ) ( ext_first_copy_toc-ext_first_copy_tic )

!_______________________________________________________________________
!
!   Run the iterations
!_______________________________________________________________________

  CALL ext_solve( mu, eta, xi, ec, w, v, sigt, mat, qi, slgg, lma)

!_______________________________________________________________________
!
! The time loop solves the problem for nsteps. If static, there is
! only one step, and it does not have any time-absorption or -source
! terms. Set the pointers to angular flux arrays. Set time to one for
! static for proper multiplication in octsweep.
!_______________________________________________________________________

  IF ( iproc == root ) WRITE( ounit, 201) ( star, i = 1, 80 )

  tot_iits = 0

  CALL wtime ( ft_start )

  time_loop: DO cy = 1, nsteps

    CALL wtime ( t3 )

    vdelt = zero
    time = one
    IF ( timedep == 1 ) THEN
      IF ( iproc == root ) WRITE( ounit, 202 ) ( star, i = 1, 30 ), cy
      vdelt = two / ( dt * v )
      time = dt * ( REAL( cy, r_knd ) - half )
    END IF

    IF ( cy > 1 ) THEN
      ptr_tmp => ptr_out
      ptr_out => ptr_in
      ptr_in  => ptr_tmp
    END IF
!_______________________________________________________________________
!
!   Scale the manufactured source for time
!_______________________________________________________________________

    IF ( src_opt == 3 ) THEN
      IF ( cy == 1 ) THEN
        qim = time*qim
      ELSE
        sf = REAL( 2*cy - 1, r_knd ) / REAL( 2*cy-3, r_knd )
        qim = qim*sf
      END IF
    END IF
!_______________________________________________________________________
!
!   Zero out flux arrays. Use threads when available.
!_______________________________________________________________________

  !$OMP PARALLEL DO SCHEDULE(DYNAMIC,1) DEFAULT(SHARED) PRIVATE(g)
    DO g = 1, ng
      flux(:,:,:,g)    = zero
      fluxm(:,:,:,:,g) = zero
    END DO
  !$OMP END PARALLEL DO
!_______________________________________________________________________
!
!   Using Jacobi iterations in energy, and the work in the outer loop
!   will be parallelized with threads.
!_______________________________________________________________________

    otrdone = .FALSE.

    cy_iits = 0

    IF ( iproc==root .AND. it_det==0 ) WRITE( ounit, 203 )

    CALL wtime ( t4 )
    tparam = tparam + t4 - t3

    outer_loop: DO otno = 1, oitm

      CALL wtime ( t5 )

      IF ( iproc==root .AND. it_det==1 ) THEN
        WRITE( ounit, 204 ) ( star, i = 1, 20 ), otno
      END IF
!_______________________________________________________________________
!
!   Prepare some cross sections: total, in-group scattering, absorption.
!   Keep in the time loop for better consistency with PARTISN. Set up
!   geometric sweep parameters. Parallelize group loop with threads.
!_______________________________________________________________________

  !$OMP PARALLEL DO SCHEDULE(DYNAMIC,1) DEFAULT(SHARED) PRIVATE(g)
      DO g = 1, ng
        CALL expxs_reg ( sigt(:,g), mat, t_xs(:,:,:,g) )
        CALL expxs_reg ( siga(:,g), mat, a_xs(:,:,:,g) )
        CALL expxs_slgg ( slgg(:,:,g,g), mat, s_xs(:,:,:,:,g) )
        CALL param_calc ( ichunk, nang, mu, eta, xi, t_xs(:,:,:,g),    &
          vdelt(g), dinv(:,:,:,:,g) )
      END DO
  !$OMP END PARALLEL DO

!_______________________________________________________________________
!
!     Perform an outer iteration. Add up inners. Check convergence.
!_______________________________________________________________________

      CALL wtime ( t6 )
      tparam = tparam + t6 - t5

      CALL outer ( out_iits )

      cy_iits = cy_iits + out_iits

      IF ( iproc == root ) WRITE( ounit, 205 ) otno, dfmxo, out_iits

      IF ( otrdone ) EXIT outer_loop

    END DO outer_loop
!_______________________________________________________________________
!
!   Print the time cycle details. Add time cycle iterations.
!_______________________________________________________________________

    IF ( timedep == 1 ) THEN
      IF ( otrdone ) THEN
        IF ( iproc == root ) WRITE( ounit, 206 ) cy, time, otno, cy_iits
      ELSE
        IF ( iproc == root ) WRITE( ounit, 207 ) cy, time, otno, cy_iits
      END IF
    ELSE
      IF ( otrdone ) THEN
        IF ( iproc == root ) WRITE( ounit, 208 ) otno, cy_iits
      ELSE
        IF ( iproc == root ) WRITE( ounit, 209 ) otno, cy_iits
      END IF
    END IF

    tot_iits = tot_iits + cy_iits

    IF ( .NOT. otrdone ) EXIT time_loop

  END DO time_loop

  IF ( timedep==1 .AND. iproc == root ) THEN
    WRITE( ounit, 210 ) ( star, i = 1, 30 ), tot_iits
  END IF
  IF ( iproc == root ) WRITE( ounit, 211 ) ( star, i = 1, 80 )

  CALL wtime ( ft_end )

  CALL wtime ( t7 )
  tslv = t7 - t1
  tmp = REAL( nx, r_knd ) * REAL( ny_gl, r_knd ) * REAL( nz_gl, r_knd )&
        * REAL( nang, r_knd ) * REAL( noct, r_knd )                    &
        * REAL( tot_iits, r_knd )
  tgrind = tslv*1.0E9_r_knd / tmp

  WRITE ( *, 217 ) ( t7-t1 )
  WRITE ( *, 218 ) ( ft_end-ft_start )


!_______________________________________________________________________
!
!   Check angular flux difference
!_______________________________________________________________________

  CALL ext_get_transpose_output_flux ( ext_angular_flux )
  PRINT *
  PRINT *, "Checking angular flux"
  PRINT *, "Max difference out:", MAXVAL ( ABS ( ptr_out-ext_angular_flux ) )
  PRINT *, "Max difference in:", MAXVAL ( ABS ( ptr_in-ext_angular_flux ) )
  PRINT *

  PRINT *, "OCL SUM", SUM ( ext_angular_flux )
  PRINT *, "ORIG SUM", SUM ( ptr_out )


!_______________________________________________________________________
!
!   Check that scalar and scalar moment fluxes match
!_______________________________________________________________________

  PRINT *
  PRINT *, "Checking scalar and moments match"
  PRINT *, "Tolerance", 100.0*epsi
  PRINT *

  CALL ext_get_transpose_scalar_flux( scalar_flux )

  IF ( ALL ( ABS ( scalar_flux - flux ) < 100.0*epsi ) ) THEN
    PRINT *, "Scalar flux matched"
  ELSE
    PRINT *, "Scalar flux did not match"
  END IF
  PRINT *, "Max error:", MAXVAL ( ABS ( scalar_flux - flux ) )

  PRINT *

  CALL ext_get_transpose_scalar_moments( scalar_flux_moments )
  IF ( ALL ( ABS ( scalar_flux_moments - fluxm ) < 100.0*epsi ) ) THEN
    PRINT *, "Scalar flux moments matched"
  ELSE
    PRINT *, "Scalar flux moments did not match"
  END IF
  PRINT *, "Max error:", MAXVAL ( ABS ( scalar_flux_moments - fluxm ) )
  PRINT *

  DEALLOCATE ( ext_angular_flux )
  DEALLOCATE ( scalar_flux )
  DEALLOCATE ( scalar_flux_moments )


!_______________________________________________________________________

  201 FORMAT( 10X, 'Iteration Monitor', /, 80A )
  202 FORMAT( /, 1X, 30A, /, 2X, 'Time Cycle ', I3 )
  203 FORMAT( 2X, 'Outer' )
  204 FORMAT( 1X, 20A, /, 2X, 'Outer ', I3 )
  205 FORMAT( 2X, I3, 4X, 'Dfmxo=', ES11.4, 4X, 'No. Inners=', I5 )
  206 FORMAT( /, 2X, 'Cycle=', I4, 4X, 'Time=', ES11.4, 4X, 'No. ',    &
              'Outers=', I4, 4X, 'No. Inners=', I5 )
  207 FORMAT( /, 2X, '***UNCONVERGED*** Stopping Iterations!!', /, 2X, &
             'Cycle=', I4, 4X, 'Time=', ES11.4, 4X, 'No. Outers=', I4, &
             4X, 'No. Inners=', I5, / )
  208 FORMAT( /, 2X, 'No. Outers=', I4, 4X, 'No. Inners=', I5 )
  209 FORMAT( /, 2X, '***UNCONVERGED*** Stopping Iterations!!', /, 2X, &
              'No. Outers=', I4, 4X, 'No. Inners=', I5, / )
  210 FORMAT( /, 1X, 30A, /, 2X, 'Total inners for all time steps, '   &
              'outers = ', I6 )
  211 FORMAT( /, 80A, / )

!_______________________________________________________________________
  212 FORMAT( 'EXT buffer init time: ', F10.3, 's' )
  213 FORMAT( 'Time spent copying updated source: ', F10.3, 's')
  214 FORMAT( 'EXT sweeps + scalar reduction: ', F10.3, 's')
  215 FORMAT( 'EXT flux reduction time: ', F10.3, 's')
  216 FORMAT( 'EXT grind time (for resident sweep): ', F10.3, 'ns')
  217 FORMAT( 'Original time: ', F10.3, 's')
  218 FORMAT( 'Fortran Runtime: ', F10.3, 's')

!_______________________________________________________________________
!_______________________________________________________________________

END SUBROUTINE translv
