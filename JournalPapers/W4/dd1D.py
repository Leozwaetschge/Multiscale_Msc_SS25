import random
import numpy as np
class dd1DClass:
    """
    A class to simulate one dimensional dislocation network
    
    Model outlined in PM Derlet, R Maass, MSMSE 21, 035007 (2013)
    
    Attributes
    
    L : float
    	periodicity length
    lambda0 : float
        internal shear stress length scale
    sigma0 : float
    	internal shear stress amplitude
    dislocationNumber : int
    	number of dislocations
    shearModulus : float
    	elastic shear modulus 
    poissonRatio : float
    	Poisson ratio
    burgersMagnitude : float
    	magnitude of Burgers vector
    dampingCoefficient : float
        dislocation damping coefficient, inverse equal to frictional coefficient
    stepNumberThreshold : int
    	maximum allowed number of iterations for dislocation network relaxation 
    forceTolerance : float
    	fraction tolerance defining zero-force configuration
    timeStep : int
    	discrete time step for trajectory evolution 
  	randomSeed : int, option
    	random seed for pseudorandom sequence (default is 220368)
    dislocationPosition : float array of size dislocationNumber
    	numpy array containing dislocation positions within interval [-L/2,L/2)
    dislocationVelocity : float array of size dislocationNumber
    	numpy array containing dislocation velocities
    dislocationForce : float array of size dislocationNumber
        numpy array containing dislocation forces
	
    Methods
    
    calcForce(externalStress)
    	calculates the force on each dislocation for an applied shear stress externalStress
    relaxNetwork(externalStress)
    	relaxes the dislocation network to a zero-force configuration for an applied shear stress externalStress
    iterateNetwork(externalStress)
    	calculates new dislocation positions for an applied shear stress externalStress for the time interval timeStep
    	
    """
    def __init__(self,L=80e-6,lambda0=2e-6,sigma0=0.01e9,dislocationNumber=60, \
            shearModulus=48e9,poissonRatio=0.34,burgersMagnitude=2.55e-10,dampingCoefficient=5e-5,\
            stepNumberThreshold=100000,forceTolerance=1e-12,timeStep=1e-9,randomSeed=220368):
        """
        
        Initializes an instance of the ddPSI class 
        
        Creates the numpy arrays containing the dislocation positions (random), velocities (zero) and forces (calcuated using 				calcForces(0.0)
        
    	Parameters
    	
	L : float, optional
	    periodicity length (default is 80e-6 meters)
    	lambda0 : float, optional
            internal shear stress length scale (default is 2e-6 mfeters)
    	sigma0 : float, optional
    	    internal shear stress amplitude (default is 0.01e9 Pascals)
    	dislocationNumber : int, optional
    	    number of dislocations (default is 60)
    	shearModulus : float, optional
    	    elastic shear modulus (default is 48e9 Pascals)
    	poissonRatio : float, optional
    	    Poisson ratio (default is 0.34)
    	burgersMagnitude : float, optional
    	    magnitude of Burgers vector (default is 2.55e-10 meters)
    	dampingCoefficient : float, optional
            dislocation damping coefficient, inverse equal to frictional coefficient (default is 5e-5 Newton meters/sec)
    	stepNumberThreshold : int, optional
    	    maximum allowed number of iterations for dislocation network relaxation (default is 100000) 
    	forceTolerance : float, optional
    	    fraction tolerance defining zero-force configuration (default is 1e-12)
    	timeStep : int, optional
    	    discrete time step for trajectory evolution (default is 1e-9 sec)
    	randomSeed : int, option
    	    random seed for pseudorandom sequence (default is 220368)

	"""
        #
        # If randomSeed>0 set random seed (allows for reproducable psuedo-random sequence)
        #
        if (randomSeed>0) : random.seed(randomSeed)
        #
        # Define internal variables
        #
        self.L=L
        self.lambda0=lambda0
        self.sigma0=sigma0
        self.dislocationNumber=dislocationNumber
        self.shearModulus=shearModulus
        self.poissonRatio=poissonRatio
        self.burgersMagnitude=burgersMagnitude
        self.dampingCoefficient=dampingCoefficient
        self.stepNumberThreshold=stepNumberThreshold
        self.forceTolerance=forceTolerance
        self.timeStep=timeStep
        #
        # Pre-calculate constants
        #
        self.h=10e-6 #self.L/float(self.dislocationNumber)
        self.deltaStrainFactor=1.0/(self.L*self.h)
        self.forceFactor1=self.shearModulus/(2.0*np.pi*(1.0-self.poissonRatio))
        self.forceFactor2=self.forceFactor1*self.burgersMagnitude**2*np.pi/self.L
        #
        # Define mask to exclude diagonal in fast matrix evaluation of forces
        #
        self.mask=np.ones((dislocationNumber,dislocationNumber),dtype=bool)
        np.fill_diagonal(self.mask,False)
        #
        # Construct dislocation position array (randomly position dislocations)
        #
        self.dislocationPosition=np.random.rand(self.dislocationNumber)*self.L
        self.dislocationPosition-=self.L/2.0
        #
        # Construct dislocation velocity array (set to zero)
        #
        self.dislocationVelocity=np.zeros(self.dislocationNumber)
        #
        # Construct dislocation force array (set to zero)
        #
        self.dislocationForce=np.zeros(self.dislocationNumber)
        self.calcForce(0.0)

    def calcForce(self,externalStress):
        """
    	
    	Calculate force on each dislocation for an applied shear stress externalStress
    	
    	Parameter
    	
    	externalStress : float
            calculates the dislocation structural order parameter
    		
        """
        #
        # Calculate forces from other dislocations using fast numpy array operatures (see comment below)
        #
        if self.dislocationNumber>0:
            self.dislocationForce=-self.forceFactor2*np.sum(np.reciprocal(np.tan(np.pi*(self.dislocationPosition-self.dislocationPosition[:,None])\
                /self.L),where=self.mask),axis=1,where=self.mask)
        else:
            self.dislocationForce[:]=0.0
        #
        # Calculate forces from external stress and internal stress amplitude
        #
        self.dislocationForce+=(externalStress+self.sigma0*np.cos(2.0*np.pi*self.dislocationPosition/\
                self.lambda0))*self.burgersMagnitude
        #
        # The more intuative approach would be
        #
        #   for dislocationI in range(self.dislocationNumber):
        #        for dislocationJ in range(dislocationI+1,self.dislocationNumber):
        #           dr=self.dislocationPosition[dislocationI]-self.dislocationPosition[dislocationJ]
        #           force=self.forceFactor2/np.tan(np.pi*dr/self.L)
        #           self.dislocationForce[dislocationI]+=force
        #           self.dislocationForce[dislocationJ]-=force
        #
        # Calculate force contribution from external stress and internal stress amplitude
        #
        #   for dislocationI in range(self.dislocationNumber):
        #        self.dislocationForce[dislocationI]+=(externalStress+ \
        #                self.sigma0*np.cos(2.0*np.pi*self.dislocationPosition[dislocationI]/self.lambda0))*self.burgersMagnitude
        #
        # This is 100-1000 times slower!
        #
        return 

    def relaxNetwork(self,externalStress):
        """

        Relax dislocation network to a zero-force configuration for an applied shear stress externalStress
    	
        Parameter
    	
        externalStress : float
            calculates the dislocation structural order parameter	

        Returns

        totalPlasticStrainIncrement : float
            the resulting plastic strain resulting from the dislocation reorganization
        stepNumber : int
        the needed number of network iterations
		
        """
        totalPlasticStrainIncrement=0.0
        #
        # Iterate network until force tolerance threshold is mett
        #
        notConverged=True
        stepNumber=0
        lastForceMagnitude=100.0
        while notConverged and stepNumber<self.stepNumberThreshold :
            stepNumber+=1
            plasticStrainIncrement=self.iterateNetwork(externalStress)
            totalPlasticStrainIncrement+=plasticStrainIncrement
            forceMagnitude=np.sum(np.abs(self.dislocationForce))
            if abs(forceMagnitude-lastForceMagnitude) < self.forceTolerance*abs(forceMagnitude) :
                notConverged=False
            else :
                lastForceMagnitude=forceMagnitude

        return totalPlasticStrainIncrement,stepNumber

    def iterateNetwork(self,externalStress):
        """
	
        Calculates new dislocation positions and velocities for an 
        applied shear stress externalStress for the time interval timeStep
    	
        Parameter
    	
        externalStress : float
        calculates the dislocation structural order parameter	
    
        Returns
    	
        plasticStrainIncrement : float
        the resulting plastic strain resulting from the dislocation reorganization
    		
        """
        #
        # Calculate the current force on each dislocation
        #
        self.calcForce(externalStress)
        #
        # Update dislocation positions and velocities
        #
        self.dislocationVelocity=self.dislocationForce/self.dampingCoefficient
        deltaDislocationPosition=self.dislocationVelocity*self.timeStep
        self.dislocationPosition+=deltaDislocationPosition
        #
        # Map any dislocations back into the interval (0,0]
        #
        self.dislocationPosition=np.where(self.dislocationPosition<-self.L/2.0, \
        	self.dislocationPosition+self.L,self.dislocationPosition)
        self.dislocationPosition=np.where(self.dislocationPosition>=self.L/2.0, \
        	self.dislocationPosition-self.L,self.dislocationPosition)
        #
        # Calculate the resulting plastic strain increment
        #
        plasticStrainIncrement=np.sum(deltaDislocationPosition)*self.burgersMagnitude*self.deltaStrainFactor

        return plasticStrainIncrement
