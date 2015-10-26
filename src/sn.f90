!-----------------------------------------------------------------------
!
! MODULE: sn_module
!> @brief
!> This module contains the variables and setup subroutines related to
!> the mock discrete ordinates treament in SNAP.
!
!-----------------------------------------------------------------------

MODULE sn_module

  USE global_module, ONLY: i_knd, r_knd, zero, one

  IMPLICIT NONE

  PUBLIC

  SAVE
!_______________________________________________________________________
!
! Module Input Variables
!
! nang     - number of discrete ordinates per octant
! nmom     - scattering order
!_______________________________________________________________________

  INTEGER(i_knd) :: nang=1, nmom=1
!_______________________________________________________________________
!
! Run-time variables
!
! cmom      - computational number of moments according to nmom & ndimen
! noct      - number of directional octants
! mu(nang)  - x direction cosines
! eta(nang) - y direction cosines
! xi(nang)  - z direction cosines
! w(nang)   - angle weights
!
! wmu(nang)  - w*mu
! weta(nang) - w*eta
! wxi(nang)  - w*xi
!
! ec(nang,cmom,noct) - Scattering expansion coefficients
! lma(nmom)          - number of 'm' moments per order l
!_______________________________________________________________________

  INTEGER(i_knd) :: cmom, noct

  INTEGER(i_knd), ALLOCATABLE, DIMENSION(:) :: lma

  REAL(r_knd), ALLOCATABLE, DIMENSION(:) :: mu, eta, xi, w, wmu, weta, &
    wxi

  REAL(r_knd), ALLOCATABLE, DIMENSION(:,:,:) :: ec


  CONTAINS


  SUBROUTINE sn_allocate ( ndimen, istat )

!-----------------------------------------------------------------------
!
! Allocate sn_module arrays.
!
!-----------------------------------------------------------------------

    INTEGER(i_knd), INTENT(IN) :: ndimen

    INTEGER(i_knd), INTENT(INOUT) :: istat
!_______________________________________________________________________
!
!   Allocate cosines and weights
!_______________________________________________________________________

    cmom = nmom
    noct = 2

    ALLOCATE( mu(nang), w(nang), wmu(nang), STAT=istat )
    IF ( istat > 0 ) RETURN

    IF ( ndimen > 1 ) THEN
      cmom = nmom*(nmom+1)/2
      noct = 4
      ALLOCATE( eta(nang), weta(nang), STAT=istat )
    ELSE
      ALLOCATE( eta(0), weta(0), STAT=istat )
    END IF
    IF ( istat > 0 ) RETURN

    IF ( ndimen > 2 ) THEN
      cmom = nmom**2
      noct = 8
      ALLOCATE( xi(nang), wxi(nang), STAT=istat )
    ELSE
      ALLOCATE( xi(0), wxi(0), STAT=istat )
    END IF
    IF ( istat > 0 ) RETURN

    ALLOCATE( ec(nang,cmom,noct), lma(nmom), STAT=istat )
    IF ( istat > 0 ) RETURN

    wmu = zero
    IF ( ndimen > 1 ) weta = zero
    IF ( ndimen > 2 ) wxi = zero
    ec = zero
    lma = 0
!_______________________________________________________________________
!_______________________________________________________________________

  END SUBROUTINE sn_allocate


  SUBROUTINE sn_deallocate

!-----------------------------------------------------------------------
!
! Deallocate sn_module arrays.
!
!-----------------------------------------------------------------------
!_______________________________________________________________________ 

    DEALLOCATE( mu, w, eta, xi, wmu, weta, wxi, ec, lma )
!_______________________________________________________________________
!_______________________________________________________________________

  END SUBROUTINE sn_deallocate


  SUBROUTINE expcoeff ( ndimen )

!-----------------------------------------------------------------------
!
! Compute and store the scattering expansion coefficients. Coefficient
! polynomial is (mu*eta*xi)^l, where l is the moment, starting at 0.
!
!-----------------------------------------------------------------------

    INTEGER(i_knd), INTENT(IN) :: ndimen
!_______________________________________________________________________
!
!   Local variables
!_______________________________________________________________________

    INTEGER(i_knd) :: id, jd, kd, is, js, ks, l, oct, m, mom
!_______________________________________________________________________
!
!   Set the coefficient as simply the angle raised to a power equal to
!   the moment-1. Set octant and loop according to dimension.
!_______________________________________________________________________

    SELECT CASE ( ndimen )

      CASE ( 1 )
        wmu = w*mu
        lma = 1
        DO id = 1, 2
          is = -1
          IF ( id == 2 ) is = 1
          ec(:,1,id) = one
          DO l = 2, nmom
            ec(:,l,id) = (is*mu)**(2*l-3)
          END DO
        END DO

      CASE ( 2 )

        wmu = w*mu
        weta = w*eta

        DO l = 1, nmom
          lma(l) = l
        END DO

        DO jd = 1, 2
          js = -1
          IF ( jd == 2 ) js = 1
          DO id = 1, 2
            is = -1
            IF ( id == 2 ) is = 1
            oct = 2*(jd-1) + id
            ec(:,1,oct) = one
            mom = 2
            DO l = 2, nmom
              DO m = 1, l
                ec(:,mom,oct) = (is*mu)**(2*l-3)*(js*eta)**(m-1)
                mom = mom + 1
              END DO
            END DO
          END DO
        END DO

      CASE ( 3 )

        wmu = w*mu
        weta = w*eta
        wxi = w*xi

        DO l = 1, nmom
          lma(l) = 2*l - 1
        END DO

        DO kd = 1, 2
          ks = -1
          IF ( kd == 2 ) ks = 1
          DO jd = 1, 2
            js = -1
            IF ( jd == 2 ) js = 1
            DO id = 1, 2
              is = -1
              IF ( id == 2 ) is = 1
              oct = 4*(kd-1) + 2*(jd-1) + id
              ec(:,1,oct) = one
              mom = 2
              DO l = 2, nmom
                DO m = 1, 2*l-1
                  ec(:,mom,oct) = (is*mu)**(2*l-3)*(ks*xi*js*eta)**(m-1)
                  mom = mom + 1
                END DO
              END DO
            END DO
          END DO
        END DO

    END SELECT
!_______________________________________________________________________
!_______________________________________________________________________

  END SUBROUTINE expcoeff


END MODULE sn_module
