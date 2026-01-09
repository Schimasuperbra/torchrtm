!*************************************************************************
!*                                                                       *
PROGRAM main_PROSAIL

! 09 22 2011
! This program allows modeling reflectance data from canopy
! - modeling leaf optical properties with PROSPECT-5 (feret et al. 2008)
! - modeling leaf inclination distribution function with the subroutine campbell
! (Ellipsoidal distribution function caracterised by the average leaf 
! inclination angle in degree), or dladgen (2 parameters LIDF)
! - modeling canopy reflectance with 4SAIL (Verhoef et al., 2007)

! This version has been implemented by Jean-Baptiste Feret
! Jean-Baptiste Feret takes the entire responsibility for this version 
! All comments, changes or questions should be sent to:
! jbferet@stanford.edu

! References:
! 	Verhoef et al. (2007) Unified Optical-Thermal Four-Stream Radiative
! 	Transfer Theory for Homogeneous Vegetation Canopies, IEEE TRANSACTIONS 
! 	ON GEOSCIENCE AND REMOTE SENSING, VOL. 45, NO. 6, JUNE 2007
! 	F?ret et al. (2008), PROSPECT-4 and 5: Advances in the Leaf Optical
! 	Properties Model Separating Photosynthetic Pigments, REMOTE SENSING OF 
! 	ENVIRONMENT
! The specific absorption coefficient corresponding to brown pigment is
! provided by Frederic Baret (EMMAH, INRA Avignon, baret@avignon.inra.fr)
! and used with his autorization.
! the model PRO4SAIL is based on a version provided by
!	Wout Verhoef
!	NLR	
!	April/May 2003

! The original 2-parameter LIDF model is developed by and described in:
! 	W. Verhoef, 1998, "Theory of radiative transfer models applied in 
!	optical remote sensing of vegetation canopies", Wageningen Agricultural
!	University,	The Netherlands, 310 pp. (Ph. D. thesis)
! the Ellipsoidal LIDF is taken from:
!   Campbell (1990), Derivtion of an angle density function for canopies 
!   with ellipsoidal leaf angle distribution, Agricultural and Forest 
!   Meteorology, 49 173-176
!*                                                                       *
!*************************************************************************

USE MOD_ANGLE               ! defines pi & rad conversion
USE MOD_staticvar           ! static variables kept in memory for optimization
USE MOD_flag_util           ! flags for optimization
USE MOD_output_PROSPECT     ! output variables of PROSPECT
USE MOD_SAIL                ! variables of SAIL
USE MOD_dataSpec_P5B        
IMPLICIT NONE

! LEAF BIOCHEMISTRY
REAL*8 :: N,Cab,Car,Cbrown,Cw,Cm
! CANOPY
REAL*8 :: lai,LIDFa,LIDFb,psoil
REAL*8 :: skyl,hspot,ihot
REAL*8 :: tts,tto,psi
REAL*8,ALLOCATABLE,SAVE :: resh(:),resv(:)
REAL*8,ALLOCATABLE,SAVE :: rsoil0(:),PARdiro(:),PARdifo(:)
INTEGER :: TypeLidf,ii,iterations
character(len=10) :: time1,time2
character(len=8)  :: thedate
character(len=12) :: thetime
real(kind=4) :: t1,t2,computationtime
integer::cputimeflag
! ANGLE CONVERSION
pi=ATAN(1.)*4.
rd=pi/180.

! PROSPECT output
ALLOCATE (LRT(nw,2),rho(nw),tau(nw))
! SAIL
ALLOCATE (sb(nw),sf(nw),vb(nw),vf(nw),w(nw))
ALLOCATE (m(nw),m2(nw),att(nw),sigb(nw),rinf(nw))
ALLOCATE (PARdiro(nw),PARdifo(nw))
ALLOCATE(tso(nw),tsd(nw),tdd(nw),tdo(nw),rsd(nw),rdd(nw),rso(nw),rdo(nw))
ALLOCATE(rddt(nw),rsdt(nw),rdot(nw),rsodt(nw),rsost(nw),rsot(nw),rsos(nw),rsod(nw))
ALLOCATE(lidf(13))
! resh : hemispherical reflectance
! resv : directional reflectance
ALLOCATE (resh(nw),resv(nw))
ALLOCATE (rsoil_old(nw))

!REAL*8 :: e1_local(nw), e2_local(nw), rinf2_local(nw), re_local(nw), denom_local(nw)




open(unit=99,file='para.ini',status='old')
read(99,*) iterations,cputimeflag,TypeLidf,psoil
read(99,*) Cab,Car,Cbrown,Cw,Cm,N 
read(99,*) LAI,hspot,tts,tto,psi 
write(*,*) "TypeLidf:", TypeLIDF
write(*,*) Cab,Car,Cbrown,Cw,Cm,N 
write(*,*) LAI,hspot,tts,tto,psi 
close(99)
!	TypeLidf=1
! if 2-parameters LIDF: TypeLidf=1
IF (TypeLidf.EQ.1) THEN
    ! LIDFa LIDF parameter a, which controls the average leaf slope
    ! LIDFb LIDF parameter b, which controls the distribution's bimodality
    !	LIDF type 		a 		 b
    !	Planophile 		1		 0
    !	Erectophile    -1	 	 0
    !	Plagiophile 	0		-1
    !	Extremophile 	0		 1
    !	Spherical 	   -0.35 	-0.15
    !	Uniform 0 0
    ! 	requirement: |LIDFa| + |LIDFb| < 1	
    LIDFa	=	-0.35
    LIDFb	=	-0.15
! if ellipsoidal distribution: TypeLidf=2
ELSEIF (TypeLidf.EQ.2) THEN
    ! 	LIDFa	= average leaf angle (degrees) 0 = planophile	/	90 = erectophile
    ! 	LIDFb = 0
    LIDFa	=	30
    LIDFb	=	0
ENDIF

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!LEAF CHEM & STR PROPERTIES!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! INITIAL PARAMETERS
!	Cab		=	40.		! chlorophyll content (?g.cm-2) 
!	Car		=	8.		! carotenoid content (?g.cm-2)
!	Cbrown	=	0.0		! brown pigment content (arbitrary units)
!	Cw		=	0.01	! EWT (cm)
!	Cm		=	0.009	! LMA (g.cm-2)
!	N		=	1.5		! structure coefficient

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!	Soil Reflectance Properties	!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! rsoil1 = dry soil
! rsoil2 = wet soil
ALLOCATE (rsoil0(nw))
!	psoil	=	1.		! soil factor (psoil=0: wet soil / psoil=1: dry soil)
rsoil0=psoil*Rsoil1+(1-psoil)*Rsoil2

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!	4SAIL canopy structure parm	!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!	LAI		=	3.		! leaf area index (m^2/m^2)
!	hspot	=	0.01	! hot spot
!	tts		=	30.		! solar zenith angle (?)
!	tto		=	10.		! observer zenith angle (?)
!	psi		=	0.		! azimuth (?)

init_completed=.false.	! only at first call of PRO4SAIL

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!        CALL PRO4SAIL         !!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
if(cputimeflag.eq.1) then 
    call cpu_time(t1)
else 
    call date_and_time(thedate,time1)
endif 

do ii=1,iterations 
    CALL PRO4SAIL(N,Cab,Car,Cbrown,Cw,Cm,LIDFa,LIDFb,TypeLIDF,LAI,hspot,tts,tto,psi,rsoil0)

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!	direct / diffuse light	!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! the direct and diffuse light are taken into account as proposed by:
    ! Francois et al. (2002) Conversion of 400?1100 nm vegetation albedo 
    ! measurements into total shortwave broadband albedo using a canopy 
    ! radiative transfer model, Agronomie
    skyl	=	0.847- 1.61*sin((90-tts)*rd)+ 1.04*sin((90-tts)*rd)*sin((90-tts)*rd) ! % diffuse radiation
    ! Es = direct
    ! Ed = diffuse
    ! PAR direct
    PARdiro	=	(1-skyl)*Es
    ! PAR diffus
    PARdifo	=	(skyl)*Ed
    ! resv : directional reflectance
    resv	= (rdot*PARdifo+rsot*PARdiro)/(PARdiro+PARdifo)
enddo
if(cputimeflag.eq.1) then 
    call cpu_time(t2)
    write(*,*) 'call cpu_time:',t2-t1
else 
    call date_and_time(thedate,time2)
    write(*,*) 'date: ',thedate,' time1: ',time1,' time2: ',time2  
    read(time1,*) t1
    read(time2,*) t2
    write(*,*) 'call date_and_time:',t2-t1
endif 
computationtime=t2-t1
    OPEN (unit=11,file='Refl_CAN_P5B.txt')
    WRITE(11,'(i4,f10.6)') (lambda(ii),resv(ii), ii=1,nw)
    CLOSE(11)

open(unit=99,file='computationTime',status='unknown')
write(99,*) "the computation time of  ",iterations,' iterations: ',computationtime,' s'
close(99)
open(unit=99,file='outmatrix',status='unknown')
write(*,*) nw
do ii=1,nw
    write(99,*) ii, m(ii), m2(ii), att(ii), sigb(ii), &
         &       rinf(ii), tso(ii), tsd(ii), tdd(ii), tdo(ii), &
         &       rsd(ii), rdd(ii), rso(ii), rdo(ii), rsos(ii), &
         &       rsod(ii), rddt(ii), rsdt(ii), rdot(ii), rsodt(ii), &
         &       rsost(ii), rsot(ii), rho(ii), tau(ii), tsstoo, &
         &       sumint, tss, too,w,lai,rdd(ii),&
         &       ddb, ddf, sdb, sdf, dob, dof, ks, ko, sob, sof, &
         &       tts, tto, cts, cto, psi, ctscto,lidf(ii)



enddo 
close(99)
STOP
END
