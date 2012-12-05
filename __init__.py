import numpy
import numpy.fft
import numpy.ma
import math
import scipy.stats
#import SRA_sub

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


class LRF_cube:

	def __init__(self, res, sigma, gamma, method):
		self.res=res
		self.gamma=gamma

		self.beta=0.75*gamma +1
		self.cube_size=2**res
		self.half_cube_size=2**(res-1)
		self.cube_shape=(self.cube_size,self.cube_size,self.cube_size)

		if method=="FFT":
			self.cube=cube_make_FFT(self.half_cube_size, self.beta)

		elif method=="SRA":
			self.cube=cube_make_SRA(self.res, sigma, (self.beta-3)/2. )

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
		self.log_cube=numpy.log(self.cube)

	
	def moving_av(self, size, log):

		self.means_array=numpy.zeros(self.cube_shape)
		
		if log==False:
			cubeA=numpy.append(self.cube, self.cube[:size,:,:], axis=0)	# wrapping round edges of array
		else: 
			cube1=numpy.log(self.cube)
			cubeA=numpy.append(cube1, cube1[:size,:,:], axis=0)	# wrapping round edges of array	

		cubeB=numpy.append(cubeA, cubeA[:,:size,:], axis=1)
		cube2=numpy.append(cubeB, cubeB[:,:,:size], axis=2)

		for i in range(cube.shape[0]):
			for j in range(cube.shape[1]):
				for k in range(cube.shape[2]):
					self.means_array[i,j,k]=numpy.mean(cube2[i:i+size,j:j+size,k:k+size])


def cube_make_FFT(cube_half_length, beta):
	half_cube_shape=cube_half_length*2,cube_half_length*2,cube_half_length+1
	cube_shape=cube_half_length*2,cube_half_length*2,cube_half_length*2
	double_cube_shape=cube_half_length*4,cube_half_length*4,cube_half_length*4

	cube=scipy.stats.uniform.rvs(loc=0, scale=2*math.pi, size=half_cube_shape)

	cube=numpy.cos(cube)+1j*numpy.sin(cube)	


	index=-beta	# gives-10/3 for logN power spec


	k_cube=numpy.fromfunction(lambda i,j,k: pow(pow(i-cube_half_length,2)+pow(j-cube_half_length,2)+pow(k,2),index/2.),half_cube_shape)
	dist=numpy.fromfunction(lambda i,j,k: numpy.sqrt(pow(i-cube_half_length,2)+pow(j-cube_half_length,2)+pow(k,2)),half_cube_shape)
#	trans=numpy.ma.masked_greater(dist, 16.)
#	trans=numpy.ma.filled(trans-trans+1,fill_value=0.)
#	k_cube=trans*k_cube
	k_cube=numpy.ma.filled(numpy.ma.masked_invalid(k_cube), fill_value=0)
	trans2=numpy.ma.masked_less(dist,1.)
	trans2=numpy.ma.filled(trans2-trans2+1,fill_value=0)
	k_cube=trans2*k_cube

	k_cube=numpy.fft.ifftshift(k_cube, axes=(0,1))


	k_cube=numpy.sqrt(numpy.abs(k_cube))

	m1=numpy.real(cube[0,0,0])
	cube=cube*k_cube[:,:,:cube_half_length+1]

	# take fft

	fft_cube=numpy.fft.irfftn(cube)
	fft_cube=numpy.fft.fftshift(fft_cube)

	# exponentiate

	#max1=numpy.max(fft_cube[:cube_half_length/2])
	mean=0#m1*0.38147#numpy.mean(fft_cube)
	std=2E-5#numpy.std(fft_cube)

	#print mean, numpy.mean(fft_cube), std, numpy.std(fft_cube)

	cube=numpy.exp((fft_cube-mean)/(std))

	#f_cube=(numpy.fft.rfftn(cube))
	#return numpy.fft.ifftshift(numpy.abs(f_cube*numpy.conj(f_cube)), axes=(0,1))

	return cube
