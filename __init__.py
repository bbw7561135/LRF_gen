import numpy
import numpy.fft
#import numpy.ma
import math
import scipy.stats

import FFT_gen
import SRA_gen
import plots


def get_corr(cube):
	bins=int(pow(cube.shape[0]/2.,2) + pow(cube.shape[1]/2.,2) + pow(cube.shape[2]/2.,2) + 1)
	covs=[[] for x in xrange(bins)]

	ff_c=numpy.fft.rfftn(cube)
	ff_c=ff_c*numpy.conj(ff_c)
	c=numpy.fft.irfftn(ff_c)
	c=numpy.fft.fftshift(c)

	for i in range(c.shape[0]):
		for j in range(c.shape[1]):
			for k in range(c.shape[2]):
				dist=pow(i-c.shape[0]/2,2)+pow(j-c.shape[0]/2,2)+pow(k-c.shape[0]/2,2)
				covs[dist].append(c[i,j,k])

	return covs

def get_1DPS(cube):
	bins=int(pow(cube.shape[0]/2.,2) + pow(cube.shape[1]/2.,2) + pow(cube.shape[2]/2.,2) + 1)
	covs=[[] for x in xrange(bins)]

	ff_c=numpy.fft.rfftn(cube)
	ff_c=ff_c*numpy.conj(ff_c)

	c=numpy.abs(numpy.fft.fftshift(ff_c, axes=[0,1]))

	for i in range(c.shape[0]):
		for j in range(c.shape[1]):
			for k in range(c.shape[2]):
				dist=pow(i-c.shape[0]/2,2)+pow(j-c.shape[0]/2,2)+pow(k,2)
				covs[dist].append(c[i,j,k])
	return covs

def get_colPS(cube):
	bins=int(pow(cube.shape[0]/2.,2) + pow(cube.shape[1]/2.,2) + 1)
	covs=[[] for x in xrange(bins)]

	cube2=numpy.sum(cube, axis=2)

	ff_c=numpy.fft.rfftn(cube2)
	ff_c=ff_c*numpy.conj(ff_c)

	c=numpy.abs(numpy.fft.fftshift(ff_c, axes=[0]))
	for i in range(c.shape[0]):
		for j in range(c.shape[1]):
			dist=pow(i-c.shape[0]/2,2)+pow(j,2)
			covs[dist].append(c[i,j])
	return covs

def get_colcorr(cube):
	bins=int(pow(cube.shape[0]/2.,2) + pow(cube.shape[1]/2.,2) + 1)
	covs=[[] for x in xrange(bins)]

	cube2=numpy.sum(cube, axis=2)

	ff_c=numpy.fft.rfftn(cube2)
	ff_c=ff_c*numpy.conj(ff_c)
	c=numpy.fft.irfftn(ff_c)/(cube2.shape[0]*(cube2.shape[1]+2))
	c=numpy.fft.fftshift(c)

	for i in range(c.shape[0]):
		for j in range(c.shape[1]):
			dist=pow(i-c.shape[0]/2,2)+pow(j-c.shape[0]/2,2)
			covs[dist].append(c[i,j])
	return covs


class LRF_cube:

	def __init__(self, res, sigma, gamma, method, outer=0., m_func=None, s_func=None):
		self.res=res
		self.gamma=gamma
		self.method=method
		self.outer_scale=outer

		self.beta=gamma
		self.cube_size=2**res
		self.half_cube_size=2**(res-1)
		self.cube_shape=(self.cube_size,self.cube_size,self.cube_size)

		if method=="FFT":
			self.cube=FFT_gen.cube_make_FFT(self.half_cube_size, self.beta, self.outer_scale, sigma, m_func, s_func)
		if method=="FFT2":
			self.cube=FFT_gen.cube_make_FFT2(self.half_cube_size, self.beta, self.outer_scale, sigma)
		elif method=="SRA":
			self.cube=SRA_gen.cube_make_SRA(self.res, sigma, (self.beta-3)/2. )

		self.log_cube=None

	def normalise(self):

		m=numpy.mean(self.cube[:self.halfsize,:,:])
		s=numpy.std(self.cube[:self.halfsize,:,:])
		self.cube-=m
		self.cube/=s

	def dump(self,filename):

		fileobj=open(filename, "w")

		for i in range(self.cube.shape[0]):
			for j in range(self.cube.shape[1]):
				for k in range(self.cube.shape[2]):

					fileobj.write("%i\t%i\t%i\t%.3f\n" %(i,j,k,self.cube[i,j,k]))

		fileobj.close()

	def log_cube_make(self):
		if self.log_cube==None:
			self.log_cube=numpy.log(self.cube)

	
	def moving_av(self, size, log):

		if size==1:
			if log==False:
				self.means_array=self.cube
			else:
				if self.log_cube==None:
					self.log_cube_make() 
				self.means_array=self.log_cube

		else:
				

			if self.method=="FFT":			# As periodic can wrap around

				self.means_array=numpy.zeros(self.cube_shape)
		
				if log==False:
					cubeA=numpy.append(self.cube, self.cube[:size,:,:], axis=0)	# wrapping round edges of array
				else:
					if self.log_cube==None:
						self.log_cube_make() 
					cubeA=numpy.append(self.log_cube, self.log_cube[:size,:,:], axis=0)	# wrapping round edges of array	

				cubeB=numpy.append(cubeA, cubeA[:,:size,:], axis=1)
				cube2=numpy.append(cubeB, cubeB[:,:,:size], axis=2)

				for i in range(self.cube.shape[0]):
					for j in range(self.cube.shape[1]):
						for k in range(self.cube.shape[2]):
							self.means_array[i,j,k]=numpy.mean(cube2[i:i+size,j:j+size,k:k+size])

