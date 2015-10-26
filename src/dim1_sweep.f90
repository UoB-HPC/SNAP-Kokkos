!-----------------------------------------------------------------------
!
! MODULE: dim1_sweep_module
!> @brief
!> This module contains the 1D mesh sweep logic.
!
!-----------------------------------------------------------------------

MODULE dim1_sweep_module

  USE global_module, ONLY: i_knd, r_knd, zero, two, one, half

  USE geom_module, ONLY: nx, hi

  USE sn_module, ONLY: cmom, nang, mu, w

  USE data_module, ONLY: src_opt, qim

  USE control_module, ONLY: fixup, tolr

  IMPLICIT NONE

  PUBLIC :: dim1_sweep

  SAVE
!_______________________________________________________________________
!
! Module variable
!
! fmin        - min scalar flux. Dummy for now, not used elsewhere.
! fmax        - max scalar flux. Dummy for now, not used elsewhere.
!
!_______________________________________________________________________

  REAL(r_knd) :: fmin=zero, fmax=zero


  CONTAINS


  SUBROUTINE dim1_sweep ( id, d1, d2, d3, d4, oct, g, psii, qtot, ec,  &
    vdelt, ptr_in, ptr_out, dinv, flux, fluxm, wmu, flkx, t_xs )

!-----------------------------------------------------------------------
!
! 1-D slab mesh sweeper.
!
!-----------------------------------------------------------------------

    INTEGER(i_knd), INTENT(IN) :: id, d1, d2, d3, d4, oct, g

    REAL(r_knd), INTENT(IN) :: vdelt

    REAL(r_knd), DIMENSION(nang), INTENT(IN) :: wmu

    REAL(r_knd), DIMENSION(nang), INTENT(INOUT) :: psii

    REAL(r_knd), DIMENSION(nx), INTENT(IN) :: t_xs

    REAL(r_knd), DIMENSION(nx), INTENT(INOUT) :: flux, flkx

    REAL(r_knd), DIMENSION(cmom,nx), INTENT(IN) :: qtot

    REAL(r_knd), DIMENSION(nang,cmom), INTENT(IN) :: ec

    REAL(r_knd), DIMENSION(nang,nx), INTENT(IN) :: dinv

    REAL(r_knd), DIMENSION(cmom-1,nx), INTENT(INOUT) :: fluxm

    REAL(r_knd), DIMENSION(d1,d2,d3,d4), INTENT(IN) :: ptr_in

    REAL(r_knd), DIMENSION(d1,d2,d3,d4), INTENT(OUT) :: ptr_out
!_______________________________________________________________________
!
!   Local variables
!_______________________________________________________________________

    INTEGER(i_knd) :: ilo, ihi, ist, i, l

    REAL(r_knd) :: sum_hv

    REAL(r_knd), DIMENSION(nang) :: psi, pc, den

    REAL(r_knd), DIMENSION(nang,2) :: hv, fxhv

    REAL(r_knd), DIMENSION(nang,nx) :: qm
!_______________________________________________________________________
!
!   Set up the mms source if necessary
!_______________________________________________________________________

    qm = zero
    IF ( src_opt == 3 ) qm = qim(:,:,1,1,oct,g)
!_______________________________________________________________________
!
!   Set up the sweep order in the i-direction. ilo here set according
!   to direction--different than ilo in octsweep. Setup the fixup
!   counter
!_______________________________________________________________________

    IF ( id == 1 ) THEN
      ilo = nx; ihi = 1; ist = -1
    ELSE
      ilo = 1; ihi = nx; ist = 1
    END IF

    fxhv = zero
!_______________________________________________________________________
!
!   Sweep the i cells. Set the boundary condition.
!_______________________________________________________________________

    i_loop: DO i = ilo, ihi, ist

      IF ( i == ilo )  psii = zero
!_______________________________________________________________________
!
!     Compute the angular source. MMS source scales linearly with time.
!_______________________________________________________________________

      psi = qtot(1,i) + qm(:,i)

      DO l = 2, cmom
        psi = psi + ec(:,l)*qtot(l,i)
      END DO
!_______________________________________________________________________
!
!     Compute the numerator for the update formula
!_______________________________________________________________________

      pc = psi + psii*mu*hi
      IF ( vdelt /= zero ) pc = pc + vdelt*ptr_in(:,i,1,1)
!_______________________________________________________________________
!
!     Compute the solution of the center. Use DD for edges. Use fixup
!     if requested.
!_______________________________________________________________________

      IF ( fixup == 0 ) THEN

        psi = pc*dinv(:,i)

        psii = two*psi - psii
        IF ( vdelt /= zero ) ptr_out(:,i,1,1) = two*psi -              &
          ptr_in(:,i,1,1)

      ELSE
!_______________________________________________________________________
!
!       Multi-pass set to zero + rebalance fixup. Determine angles that
!       will need fixup first.
!_______________________________________________________________________

        hv = one; sum_hv = SUM( hv )

        pc = pc * dinv(:,i)

        fixup_loop: DO

          fxhv(:,1) = two*pc - psii
          IF ( vdelt /= zero ) fxhv(:,2) = two*pc - ptr_in(:,i,1,1)

          WHERE ( fxhv < zero ) hv = zero
!_______________________________________________________________________
!
!         Exit loop when all angles are fixed up
!_______________________________________________________________________

          IF ( sum_hv == SUM( hv ) ) EXIT fixup_loop
          sum_hv = SUM( hv )
!_______________________________________________________________________
!
!         Recompute balance equation numerator and denominator and get
!         new cell average flux
!_______________________________________________________________________

          pc = psii*mu*hi*(one+hv(:,1))
          IF ( vdelt /= zero )                                         &
            pc = pc + vdelt*ptr_in(:,i,1,1)*(one+hv(:,2))
          pc = psi + half*pc

          den = t_xs(i) + mu*hi*hv(:,1) + vdelt*hv(:,2)
          
          WHERE( den > tolr )
            pc = pc/den
          ELSEWHERE
            pc = zero
          END WHERE

        END DO fixup_loop
!_______________________________________________________________________
!
!       Fixup done, compute edges
!_______________________________________________________________________

        psi = pc

        psii = fxhv(:,1) * hv(:,1)
        IF ( vdelt /= zero ) ptr_out(:,i,1,1) = fxhv(:,2) * hv(:,2)

      END IF
!_______________________________________________________________________
!
!     Clear the flux arrays
!_______________________________________________________________________

      IF ( id == 1 ) THEN
        flux(i) = zero
        fluxm(:,i) = zero
      END IF
!_______________________________________________________________________
!
!     Compute the flux moments
!_______________________________________________________________________

      flux(i) = flux(i) + SUM( w*psi )
      DO l = 1, cmom-1
        fluxm(l,i) = fluxm(l,i) + SUM( ec(:,l+1)*w*psi )
      END DO
!_______________________________________________________________________
!
!     Calculate min and max scalar fluxes (not used elsewhere currently)
!_______________________________________________________________________

      IF ( id == 2 ) THEN
        fmin = MIN( fmin, flux(i) )
        fmax = MAX( fmax, flux(i) )
      END IF
!_______________________________________________________________________
!
!     Compute leakages (not used elsewhere currently)
!_______________________________________________________________________

      IF ( i+id-1==1 .OR. i+id-1==nx+1 ) THEN
        flkx(i) = flkx(i) + ist*SUM( wmu*psii )
      END IF
!_______________________________________________________________________
!
!     Finish the loop
!_______________________________________________________________________

    END DO i_loop
!_______________________________________________________________________
!_______________________________________________________________________

  END SUBROUTINE dim1_sweep


END MODULE dim1_sweep_module
