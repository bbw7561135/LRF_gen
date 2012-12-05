import numpy
import numpy.fft
import math
import scipy.stats


def cube_make_SRA(res, sigma, H):

	N=2**res
	self.X=numpy.zeros([N+1, N+1, N+1])

	self.size=N
	self.halfsize=N/2

	delta=sigma

	self.X[0:N+1:N,0:N+1:N,0:N+1:N]=scipy.stats.norm.rvs(scale=delta, size=[2,2,2]) 

#	print self.X
	#NO=1
	N1=N
	N2=N1/2

  	delta1=delta*pow(3./4.,H)*math.sqrt(1-0.25*pow(4./3.,H)) / pow(2,-H)
	delta2=delta*pow(2.,-2*H)*math.sqrt(1-0.25*(3./2.)**H) / pow(2,-H)
    	delta3=delta*pow(2.,-H) * math.sqrt(1-0.25*pow(2.,H)) / pow(2,-H)

	for stage in range(1, res+1):
		#for NNN in range(1, NO+1):

		delta1*=pow(2.,-H)
		delta2*=pow(2.,-H)
		delta3*=pow(2.,-H)

	# Type 1 analogue (Saupe) cube - Jilesen a
	# cube centre points

		self.X[N2:-N2:N1, N2:-N2:N1, N2:-N2:N1] += ( self.X[2*N2::N1, 2*N2::N1, 2*N2::N1] + self.X[2*N2::N1, 2*N2::N1, :-2*N2:N1] +
							self.X[2*N2::N1, :-2*N2:N1, 2*N2::N1] + self.X[2*N2::N1, :-2*N2:N1, :-2*N2:N1] +
							self.X[:-2*N2:N1, 2*N2::N1, 2*N2::N1] + self.X[:-2*N2:N1, 2*N2::N1, :-2*N2:N1] + 
							self.X[:-2*N2:N1, :-2*N2:N1, 2*N2::N1] + self.X[:-2*N2:N1, :-2*N2:N1, :-2*N2:N1] 
							)/8. +scipy.stats.norm.rvs(scale=delta1, size=self.X[N2:-N2:N1, N2:-N2:N1, N2:-N2:N1].shape)
		
		# Random addition
		self.X[0::N1, 0::N1, 0::N1] += scipy.stats.norm.rvs(scale=delta1, size=self.X[0::N1, 0::N1, 0::N1].shape)

	# Type 2a analogue - square bipyramid - Jilesen b
	# face mid points

		# outer-side points
		self.X[N2:-N2:N1, N2:-N2:N1, 0] = ( self.X[2*N2::N1, 2*N2::N1, 0] + self.X[2*N2::N1, :-2*N2:N1, 0] + self.X[:-2*N2:N1, 2*N2::N1, 0] 
						+ self.X[:-2*N2:N1, :-2*N2:N1, 0] + self.X[N2:-N2:N1, N2:-N2:N1, N2] 
						)/5. + scipy.stats.norm.rvs(scale=delta2, size=self.X[N2:-N2:N1, N2:-N2:N1, 0].shape)
		self.X[N2:-N2:N1, N2:-N2:N1,-1] = ( self.X[2*N2::N1, 2*N2::N1,-1] + self.X[2*N2::N1, :-2*N2:N1,-1] + self.X[:-2*N2:N1, 2*N2::N1,-1] 
						+ self.X[:-2*N2:N1, :-2*N2:N1,-1] + self.X[N2:-N2:N1, N2:-N2:N1,-N2-1] 
						)/5. + scipy.stats.norm.rvs(scale=delta2, size=self.X[N2:-N2:N1, N2:-N2:N1, 0].shape)
		self.X[N2:-N2:N1, 0, N2:-N2:N1] = ( self.X[2*N2::N1, 0, 2*N2::N1] + self.X[2*N2::N1, 0, :-2*N2:N1] + self.X[:-2*N2:N1, 0, 2*N2::N1] 
						+ self.X[:-2*N2:N1, 0, :-2*N2:N1] + self.X[N2:-N2:N1, N2, N2:-N2:N1] 
						)/5. + scipy.stats.norm.rvs(scale=delta2, size=self.X[N2:-N2:N1, N2:-N2:N1, 0].shape)
		self.X[N2:-N2:N1,-1, N2:-N2:N1] = ( self.X[2*N2::N1,-1, 2*N2::N1] + self.X[2*N2::N1,-1, :-2*N2:N1] + self.X[:-2*N2:N1,-1, 2*N2::N1] 
						+ self.X[:-2*N2:N1,-1, :-2*N2:N1] + self.X[N2:-N2:N1,-N2-1, N2:-N2:N1] 
						)/5. + scipy.stats.norm.rvs(scale=delta2, size=self.X[N2:-N2:N1, N2:-N2:N1, 0].shape)
		self.X[0, N2:-N2:N1, N2:-N2:N1] = ( self.X[0, 2*N2::N1, 2*N2::N1] + self.X[0, 2*N2::N1, :-2*N2:N1] + self.X[0, :-2*N2:N1, 2*N2::N1] 
						+ self.X[0, :-2*N2:N1, :-2*N2:N1] + self.X[N2, N2:-N2:N1, N2:-N2:N1] 
						)/5. + scipy.stats.norm.rvs(scale=delta2, size=self.X[N2:-N2:N1, N2:-N2:N1, 0].shape)
		self.X[-1,N2:-N2:N1, N2:-N2:N1] = ( self.X[-1,2*N2::N1, 2*N2::N1] + self.X[-1,2*N2::N1, :-2*N2:N1] + self.X[-1,:-2*N2:N1, 2*N2::N1] 
						+ self.X[-1,:-2*N2:N1, :-2*N2:N1] + self.X[N2:-N2:N1, N2:-N2:N1,-N2-1] 
						)/5. + scipy.stats.norm.rvs(scale=delta2, size=self.X[N2:-N2:N1, N2:-N2:N1, 0].shape)


		# other points 
		if stage!=1:
			self.X[N2:-N2:N1, N2:-N2:N1, N1:-N2:N1] = ( 	self.X[2*N2::N1, 2*N2::N1, N1:-N2:N1] + self.X[2*N2::N1, :-2*N2:N1, N1:-N2:N1] +
								self.X[:-2*N2:N1, 2*N2::N1, N1:-N2:N1] + self.X[:-2*N2:N1, :-2*N2:N1, N1:-N2:N1] + 
								self.X[N2:-N2:N1, N2:-N2:N1, N1+N2::N1] + self.X[N2:-N2:N1, N2:-N2:N1, N1-N2:-2*N2:N1]
								)/6. + scipy.stats.norm.rvs(scale=delta2, size=self.X[N2:-N2:N1, N2:-N2:N1, N1:-N2:N1].shape)

			self.X[N2:-N2:N1, N1:-N2:N1, N2:-N2:N1] = ( 	self.X[2*N2::N1, N1:-N2:N1, 2*N2::N1] + self.X[2*N2::N1, N1:-N2:N1, :-2*N2:N1] +
								self.X[:-2*N2:N1, N1:-N2:N1, 2*N2::N1] + self.X[:-2*N2:N1, N1:-N2:N1, :-2*N2:N1] + 
								self.X[N2:-N2:N1, N1+N2::N1, N2:-N2:N1] + self.X[N2:-N2:N1, N1-N2:-2*N2:N1, N2:-N2:N1]
								)/6. + scipy.stats.norm.rvs(scale=delta2, size=self.X[N2:-N2:N1, N1:-N2:N1, N2:-N2:N1].shape)

			self.X[N1:-N2:N1, N2:-N2:N1, N2:-N2:N1] = ( 	self.X[N1:-N2:N1, 2*N2::N1, 2*N2::N1] + self.X[N1:-N2:N1, 2*N2::N1, :-2*N2:N1] +
								self.X[N1:-N2:N1, :-2*N2:N1, 2*N2::N1] + self.X[N1:-N2:N1, :-2*N2:N1, :-2*N2:N1] + 
								self.X[N1+N2::N1, N2:-N2:N1, N2:-N2:N1] + self.X[N1-N2:-2*N2:N1, N2:-N2:N1, N2:-N2:N1]
								)/6. + scipy.stats.norm.rvs(scale=delta2, size=self.X[N1:-N2:N1, N2:-N2:N1, N2:-N2:N1].shape)

		# Random addition
		self.X[0::N1, 0::N1, 0::N1] += scipy.stats.norm.rvs(scale=delta2, size=self.X[0::N1, 0::N1, 0::N1].shape)
		self.X[N2:-N2:N1, N2:-N2:N1, N2:-N2:N1] += scipy.stats.norm.rvs(scale=delta2, size=self.X[N2:-N2:N1, N2:-N2:N1, N2:-N2:N1].shape)		



	# Type 2b analogue - octohedron - Jilesen c
	# edge middle points
	#
	# Maybe an error here in Lu et al.'s technique that I have attempted to correct

		# outer edges x12!

		self.X[N2:-N2:N1, 0, 0] = ( self.X[2*N2::N1, 0, 0] + self.X[:-2*N2:N1, 0, 0] + self.X[N2:-N2:N1, 0, N2] + self.X[N2:-N2:N1, N2, 0] 
					)/4. + scipy.stats.norm.rvs(scale=delta3, size=self.X[N2:-N2:N1, 0, 0].shape)
		self.X[N2:-N2:N1, 0,-1] = ( self.X[2*N2::N1, 0,-1] + self.X[:-2*N2:N1, 0,-1] + self.X[N2:-N2:N1, 0, -N2-1] + self.X[N2:-N2:N1, N2,-1] 
					)/4. + scipy.stats.norm.rvs(scale=delta3, size=self.X[N2:-N2:N1, 0, 0].shape)
		self.X[N2:-N2:N1,-1, 0] = ( self.X[2*N2::N1,-1, 0] + self.X[:-2*N2:N1,-1, 0] + self.X[N2:-N2:N1,-1, N2] + self.X[N2:-N2:N1, -N2-1, 0] 
					)/4. + scipy.stats.norm.rvs(scale=delta3, size=self.X[N2:-N2:N1, 0, 0].shape)
		self.X[N2:-N2:N1,-1,-1] = ( self.X[2*N2::N1,-1,-1] + self.X[:-2*N2:N1,-1,-1] + self.X[N2:-N2:N1,-1, -N2-1] + self.X[N2:-N2:N1, -N2-1,-1] 
					)/4. + scipy.stats.norm.rvs(scale=delta3, size=self.X[N2:-N2:N1, 0, 0].shape)

		self.X[0, N2:-N2:N1, 0] = ( self.X[0, 2*N2::N1, 0] + self.X[0, :-2*N2:N1, 0] + self.X[0, N2:-N2:N1, N2] + self.X[N2, N2:-N2:N1, 0] 
					)/4. + scipy.stats.norm.rvs(scale=delta3, size=self.X[0, N2:-N2:N1, 0].shape)
		self.X[0, N2:-N2:N1,-1] = ( self.X[0, 2*N2::N1,-1] + self.X[0, :-2*N2:N1,-1] + self.X[0, N2:-N2:N1, -N2-1] + self.X[N2, N2:-N2:N1,-1] 
					)/4. + scipy.stats.norm.rvs(scale=delta3, size=self.X[0, N2:-N2:N1, 0].shape)
		self.X[-1, N2:-N2:N1, 0] = ( self.X[-1, 2*N2::N1, 0] + self.X[-1, :-2*N2:N1, 0] + self.X[-1, N2:-N2:N1, N2] + self.X[-N2-1, N2:-N2:N1, 0] 
					)/4. + scipy.stats.norm.rvs(scale=delta3, size=self.X[0, N2:-N2:N1, 0].shape)
		self.X[-1, N2:-N2:N1,-1] = ( self.X[-1, 2*N2::N1,-1] + self.X[-1, :-2*N2:N1,-1] + self.X[-1, N2:-N2:N1, -N2-1] + self.X[-N2-1, N2:-N2:N1,-1] 
					)/4. + scipy.stats.norm.rvs(scale=delta3, size=self.X[0, N2:-N2:N1, 0].shape)

		self.X[0, 0, N2:-N2:N1] = ( self.X[0, 0, 2*N2::N1] + self.X[0, 0, :-2*N2:N1] + self.X[0, N2, N2:-N2:N1] + self.X[N2, 0, N2:-N2:N1] 
					)/4. + scipy.stats.norm.rvs(scale=delta3, size=self.X[0, 0, N2:-N2:N1].shape)
		self.X[0,-1, N2:-N2:N1] = ( self.X[0,-1, 2*N2::N1] + self.X[0,-1, :-2*N2:N1] + self.X[0, -N2-1, N2:-N2:N1] + self.X[N2,-1, N2:-N2:N1] 
					)/4. + scipy.stats.norm.rvs(scale=delta3, size=self.X[0, 0, N2:-N2:N1].shape)
		self.X[-1, 0, N2:-N2:N1] = ( self.X[-1, 0, 2*N2::N1] + self.X[-1, 0, :-2*N2:N1] + self.X[-1, N2, N2:-N2:N1] + self.X[-N2-1, 0, N2:-N2:N1] 
					)/4. + scipy.stats.norm.rvs(scale=delta3, size=self.X[0, 0, N2:-N2:N1].shape)
		self.X[-1,-1, N2:-N2:N1] = ( self.X[-1,-1, 2*N2::N1] + self.X[-1,-1, :-2*N2:N1] + self.X[-1, -N2-1, N2:-N2:N1] + self.X[-N2-1,-1, N2:-N2:N1] 
					)/4. + scipy.stats.norm.rvs(scale=delta3, size=self.X[0, 0, N2:-N2:N1].shape)

		# other points  

		if stage>1:
			self.X[N2:-N2:N1, N1:-N2:N1, 0] = (self.X[N2:-N2:N1, N1:-N2:N1, N2] + self.X[N2:-N2:N1, N1+N2::N1, 0] + self.X[N2:-N2:N1, N1-N2:-2*N2:N1, 0] +
							 self.X[2*N2::N1, N1:-N2:N1, 0] + self.X[:-2*N2:N1, N1:-N2:N1, 0] 
							)/5. + scipy.stats.norm.rvs(scale=delta3, size=self.X[N2:-N2:N1, N1:-N2:N1, 0].shape)
			self.X[N2:-N2:N1, N1:-N2:N1,-1] = (self.X[N2:-N2:N1, N1:-N2:N1,-N2-1] + self.X[N2:-N2:N1, N1+N2::N1,-1] + self.X[N2:-N2:N1, N1-N2:-2*N2:N1,-1] +
							 self.X[2*N2::N1, N1:-N2:N1, -1] + self.X[:-2*N2:N1, N1:-N2:N1, -1]  
							)/5. + scipy.stats.norm.rvs(scale=delta3, size=self.X[N2:-N2:N1, N1:-N2:N1, 0].shape) 
			self.X[N1:-N2:N1, N2:-N2:N1, 0] = (self.X[N1:-N2:N1, N2:-N2:N1, N2] + self.X[N1:-N2:N1, 2*N2::N1, 0] + self.X[N1:-N2:N1, :-2*N2:N1, 0] +
							 self.X[N1+N2::N1, N2:-N2:N1, 0] + self.X[N1-N2:-2*N2:N1, N2:-N2:N1, 0]  
							)/5. + scipy.stats.norm.rvs(scale=delta3, size=self.X[N1:-N2:N1, N2:-N2:N1, 0].shape)
			self.X[N1:-N2:N1, N2:-N2:N1,-1] = (self.X[N1:-N2:N1, N2:-N2:N1, -N2-1] + self.X[N1:-N2:N1, 2*N2::N1,-1] + self.X[N1:-N2:N1, :-2*N2:N1,-1] +
							 self.X[N1+N2::N1, N2:-N2:N1, -1] + self.X[N1-N2:-2*N2:N1, N2:-N2:N1, -1]  
							)/5. + scipy.stats.norm.rvs(scale=delta3, size=self.X[N1:-N2:N1, N2:-N2:N1, 0].shape)



			self.X[N2:-N2:N1, 0, N1:-N2:N1] = (self.X[N2:-N2:N1, N2, N1:-N2:N1] + self.X[N2:-N2:N1, 0, N1+N2::N1] + self.X[N2:-N2:N1, 0, N1-N2:-2*N2:N1] +
							 self.X[2*N2::N1, 0, N1:-N2:N1] + self.X[:-2*N2:N1, 0, N1:-N2:N1]  
							)/5. + scipy.stats.norm.rvs(scale=delta3, size=self.X[N2:-N2:N1, 0, N1:-N2:N1].shape) 
			self.X[N2:-N2:N1,-1, N1:-N2:N1] = (self.X[N2:-N2:N1,-N2-1, N1:-N2:N1] + self.X[N2:-N2:N1,-1, N1+N2::N1] + self.X[N2:-N2:N1,-1, N1-N2:-2*N2:N1] +
							 self.X[2*N2::N1, -1, N1:-N2:N1] + self.X[:-2*N2:N1, -1, N1:-N2:N1]  
							)/5. + scipy.stats.norm.rvs(scale=delta3, size=self.X[N2:-N2:N1,-1, N1:-N2:N1].shape)
			self.X[N1:-N2:N1, 0, N2:-N2:N1] = (self.X[N1:-N2:N1, N2, N2:-N2:N1] + self.X[N1:-N2:N1, 0, 2*N2::N1] + self.X[N1:-N2:N1, 0, :-2*N2:N1] +
							 self.X[N1+N2::N1, 0, N2:-N2:N1] + self.X[N1-N2:-2*N2:N1, 0, N2:-N2:N1]  
							)/5. + scipy.stats.norm.rvs(scale=delta3, size=self.X[N1:-N2:N1, 0, N2:-N2:N1].shape) 
			self.X[N1:-N2:N1,-1, N2:-N2:N1] = (self.X[N1:-N2:N1, -N2-1, N2:-N2:N1] + self.X[N1:-N2:N1, -1, 2*N2::N1] + self.X[N1:-N2:N1, -1, :-2*N2:N1] +
							 self.X[N1+N2::N1, -1, N2:-N2:N1] + self.X[N1-N2:-2*N2:N1, -1, N2:-N2:N1]  
							)/5. + scipy.stats.norm.rvs(scale=delta3, size=self.X[N1:-N2:N1, 0, N2:-N2:N1].shape) 

			self.X[0, N2:-N2:N1, N1:-N2:N1] = (self.X[N2, N2:-N2:N1, N1:-N2:N1] + self.X[0, N2:-N2:N1, N1+N2::N1] + self.X[0, N2:-N2:N1, N1-N2:-2*N2:N1] +
							 self.X[0, 2*N2::N1, N1:-N2:N1] + self.X[0, :-2*N2:N1, N1:-N2:N1]  
							)/5. + scipy.stats.norm.rvs(scale=delta3, size=self.X[0, N2:-N2:N1, N1:-N2:N1].shape) 
			self.X[-1, N2:-N2:N1, N1:-N2:N1] = (self.X[-N2-1, N2:-N2:N1, N1:-N2:N1] + self.X[-1, N2:-N2:N1, N1+N2::N1] +
							 self.X[-1, N2:-N2:N1, N1-N2:-2*N2:N1] + self.X[-1, 2*N2::N1, N1:-N2:N1] + self.X[-1, :-2*N2:N1, N1:-N2:N1] 
							)/5. + scipy.stats.norm.rvs(scale=delta3, size=self.X[0, N2:-N2:N1, N1:-N2:N1].shape)
			self.X[0, N1:-N2:N1, N2:-N2:N1] = (self.X[N2, N1:-N2:N1, N2:-N2:N1] + self.X[0, N1:-N2:N1, 2*N2::N1] + self.X[0, N1:-N2:N1, :-2*N2:N1] +
							 self.X[0, N1+N2::N1, N2:-N2:N1] + self.X[0, N1-N2:-2*N2:N1, N2:-N2:N1]  
							)/5. + scipy.stats.norm.rvs(scale=delta3, size=self.X[0, N1:-N2:N1, N2:-N2:N1].shape)
			self.X[-1, N1:-N2:N1, N2:-N2:N1] = (self.X[-N2-1, N1:-N2:N1, N2:-N2:N1] + self.X[-1, N1:-N2:N1, 2*N2::N1] + self.X[-1, N1:-N2:N1, :-2*N2:N1] +
							 self.X[-1, N1+N2::N1, N2:-N2:N1] + self.X[-1, N1-N2:-2*N2:N1, N2:-N2:N1]  
							)/5. + scipy.stats.norm.rvs(scale=delta3, size=self.X[0, N1:-N2:N1, N2:-N2:N1].shape) 


			self.X[N2:-N2:N1, N1:-N1:N1, N1:-N1:N1] = ( self.X[N2:-N2:N1, N1:-N1:N1, N1+N2:-N1+N2:N1] + self.X[N2:-N2:N1, N1:-N1:N1, N1-N2:-N1-N2:N1] +
								self.X[N2:-N2:N1, N1+N2:-N1+N2:N1, N1:-N1:N1] + self.X[N2:-N2:N1, N1-N2:-N1-N2:N1, N1:-N1:N1] + 
								self.X[2*N2::N1, N1:-N1:N1, N1:-N1:N1] + self.X[:-2*N2:N1, N1:-N1:N1, N1:-N1:N1] 
								)/6. + scipy.stats.norm.rvs(scale=delta3, size=self.X[N2:-N2:N1, N1:-N1:N1, N1:-N1:N1].shape)
			self.X[N1:-N1:N1, N2:-N2:N1, N1:-N1:N1] = ( self.X[N1:-N1:N1, N2:-N2:N1, N1+N2:-N1+N2:N1] + self.X[N1:-N1:N1, N2:-N2:N1, N1-N2:-N1-N2:N1] +
								self.X[ N1+N2:-N1+N2:N1, N2:-N2:N1,N1:-N1:N1] + self.X[ N1-N2:-N1-N2:N1, N2:-N2:N1,N1:-N1:N1] + 
								self.X[N1:-N1:N1, 2*N2::N1, N1:-N1:N1] + self.X[N1:-N1:N1, :-2*N2:N1, N1:-N1:N1]  
								)/6. + scipy.stats.norm.rvs(scale=delta3, size=self.X[N1:-N1:N1, N2:-N2:N1, N1:-N1:N1].shape)
			self.X[N1:-N1:N1, N1:-N1:N1, N2:-N2:N1] = ( self.X[N1:-N1:N1, N1+N2:-N1+N2:N1, N2:-N2:N1] + self.X[N1:-N1:N1, N1-N2:-N1-N2:N1, N2:-N2:N1] +
								self.X[N1+N2:-N1+N2:N1, N1:-N1:N1, N2:-N2:N1] + self.X[N1-N2:-N1-N2:N1, N1:-N1:N1, N2:-N2:N1] + 
								self.X[N1:-N1:N1, N1:-N1:N1, 2*N2::N1] + self.X[N1:-N1:N1, N1:-N1:N1, :-2*N2:N1]  
								)/6. + scipy.stats.norm.rvs(scale=delta3, size=self.X[N1:-N1:N1, N1:-N1:N1, N2:-N2:N1].shape)

		# random addition

		self.X[0::N1, 0::N1, 0::N1] += scipy.stats.norm.rvs(scale=delta3, size=self.X[0::N1, 0::N1, 0::N1].shape)
		self.X[N2:-N2:N1, N2:-N2:N1, N2:-N2:N1] += scipy.stats.norm.rvs(scale=delta3, size=self.X[N2:-N2:N1, N2:-N2:N1, N2:-N2:N1].shape)

		self.X[N2:-N2:N1, N2:-N2:N1, 0] += scipy.stats.norm.rvs(scale=delta3, size=self.X[N2:-N2:N1, N2:-N2:N1, 0].shape)
		self.X[N2:-N2:N1, N2:-N2:N1,-1] += scipy.stats.norm.rvs(scale=delta3, size=self.X[N2:-N2:N1, N2:-N2:N1,-1].shape)
		self.X[N2:-N2:N1, 0, N2:-N2:N1] += scipy.stats.norm.rvs(scale=delta3, size=self.X[N2:-N2:N1, 0, N2:-N2:N1].shape)
		self.X[N2:-N2:N1,-1, N2:-N2:N1] += scipy.stats.norm.rvs(scale=delta3, size=self.X[N2:-N2:N1,-1, N2:-N2:N1].shape)
		self.X[0, N2:-N2:N1, N2:-N2:N1] += scipy.stats.norm.rvs(scale=delta3, size=self.X[0, N2:-N2:N1, N2:-N2:N1].shape)
		self.X[-1, N2:-N2:N1, N2:-N2:N1] += scipy.stats.norm.rvs(scale=delta3, size=self.X[-1, N2:-N2:N1, N2:-N2:N1].shape)

		if stage!=1:
			self.X[N2:-N2:N1, N2:-N2:N1, N1:-N2:N1] += scipy.stats.norm.rvs(scale=delta3, size=self.X[N2:-N2:N1, N2:-N2:N1, N1:-N2:N1].shape)
			self.X[N2:-N2:N1, N1:-N2:N1, N2:-N2:N1] += scipy.stats.norm.rvs(scale=delta3, size=self.X[N2:-N2:N1, N1:-N2:N1, N2:-N2:N1].shape)
			self.X[N1:-N2:N1, N2:-N2:N1, N2:-N2:N1] += scipy.stats.norm.rvs(scale=delta3, size=self.X[N1:-N2:N1, N2:-N2:N1, N2:-N2:N1].shape)
			
		N1/=2
		N2/=2
		#print stage
		#print self.X

	#X=scipy.stats.norm.rvs(size=[N+1,N+1,N+1])

