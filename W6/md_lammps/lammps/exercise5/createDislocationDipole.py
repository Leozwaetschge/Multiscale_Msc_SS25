#!/usr/bin/python
import numpy as np
import math

latticeConstant=3.639087    
nX=2
nY=30
nZ=48

atomPosition=np.zeros([nX*nY*nZ*2,3],np.float64)

lX=latticeConstant/np.sqrt(2.0)
lY=latticeConstant*np.sqrt(2.0)*np.sin(60.0*np.pi/180.0)
lZ=latticeConstant/np.sqrt(3.0)

stackingVectorX=3.0*lX/2.0/3.0
stackingVectorY=lY/2.0/3.0
#
# Create perfect lattice with axes [110], [112], [111]
#
iSite=0
iStacking=0
for iZ in range(nZ):
  posZ=iZ*lZ
  for iX in range(nX):
    for iY in range(nY):

      posX=lX*iX+iStacking*stackingVectorX
      posY=lY*iY+iStacking*stackingVectorY
      if (posX>=nX*lX) : posX-=nX*lX
      if (posY>=nY*lY) : posY-=nY*lY
      iSite+=1
      atomPosition[iSite-1,0]=posX
      atomPosition[iSite-1,1]=posY
      atomPosition[iSite-1,2]=posZ

      posX=lX*iX+iStacking*stackingVectorX+lX/2.0
      posY=lY*iY+iStacking*stackingVectorY+lY/2.0
      if (posX>=nX*lX) : posX-=nX*lX
      if (posY>=nY*lY) : posY-=nY*lY
      iSite+=1
      atomPosition[iSite-1,0]=posX
      atomPosition[iSite-1,1]=posY
      atomPosition[iSite-1,2]=posZ

  iStacking+=1
  if (iStacking==3) : iStacking=0

numberOfAtoms=iSite
#
# Calculate center of mass and make it the origin
#
comX=0.0
comY=0.0
comZ=0.0
for iSite in range(numberOfAtoms):
  comX+=atomPosition[iSite,0]
  comY+=atomPosition[iSite,1]
  comZ+=atomPosition[iSite,2]
comX/=numberOfAtoms
comY/=numberOfAtoms
comZ/=numberOfAtoms
for iSite in range(numberOfAtoms):
  atomPosition[iSite,0]-=comX
  atomPosition[iSite,1]-=comY
  atomPosition[iSite,2]-=comZ
#
# Add first screw dislocation
#
disPositionZ=nZ*lZ/4.0
disPositionY=0.0
deltaYFlag=False
deltaZFlag=False
for iSite in range(numberOfAtoms):
  if (np.abs(atomPosition[iSite,2]-disPositionZ)<1e-5):
    deltaZ=0.0
    deltaZFlag=True
  else :
    deltaZ=atomPosition[iSite,2]-disPositionZ
  if (np.abs(atomPosition[iSite,1]-disPositionY)<1e-5):
    deltaY=0.0
    deltaYFlag=True
  else :
    deltaY=atomPosition[iSite,1]-disPositionY
  if (deltaZFlag and deltaYFlag):
    displacement=0.0
  else :
    displacement=lX*(np.pi+math.atan2(deltaY,deltaZ))/(2.0*np.pi)
  atomPosition[iSite,0]+=displacement
#
# Add second screw dislocation
#
disPositionZ=-nZ*lZ/4.0
disPositionY=0.0
deltaYFlag=False
deltaZFlag=False
for iSite in range(numberOfAtoms):
  if (np.abs(atomPosition[iSite,2]-disPositionZ)<1e-5):
    deltaZ=0.0
    deltaZFlag=True
  else :
    deltaZ=atomPosition[iSite,2]-disPositionZ
  if (np.abs(atomPosition[iSite,1]-disPositionY)<1e-5):
    deltaY=0.0
    deltaYFlag=True
  else :
    deltaY=atomPosition[iSite,1]-disPositionY
  if (deltaZFlag and deltaYFlag):
    displacement=0.0
  else :
    displacement=lX*(np.pi+math.atan2(deltaY,deltaZ))/(2.0*np.pi)
  atomPosition[iSite,0]-=displacement
#
# Write data to lammps file
#
f=open("dislocationDipole.lammps","w+")
f.write("FCC crystal\r\n\r\n")
f.write("%d atoms\r\n\r\n" % (numberOfAtoms))
f.write("2 atom types\r\n\r\n")
f.write("%f %f xlo xhi\r\n" % (-nX*lX/2.0,nX*lX/2.0))
f.write("%f %f ylo yhi\r\n" % (-nY*lY/2.0,nY*lY/2.0))
f.write("%f %f zlo zhi\r\n\r\n" % (-nZ*lZ/2.0,nZ*lZ/2.0))
f.write("Atoms\r\n\r\n")
for iSite in range(numberOfAtoms):
  f.write("%d 1 %f %f %f\r\n" % (iSite+1,atomPosition[iSite,0],atomPosition[iSite,1],atomPosition[iSite,2]))
f.close()

