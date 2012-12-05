import numpy
import numpy.fft
import numpy.ma
import math
import scipy.stats


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
