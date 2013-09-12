import numpy
import numpy.fft
import numpy.ma
import math
import scipy.stats
import scipy.interpolate
import warnings
import mpmath

warnings.filterwarnings("ignore", category=RuntimeWarning)



def cube_make_FFT(cube_half_length, beta, outer_scale, sigma, m_func=None, s_func=None):
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
	trans2=numpy.ma.masked_less(dist,outer_scale)
	trans2=numpy.ma.filled(trans2-trans2+1,fill_value=0)
	k_cube=trans2*k_cube-(trans2-1)*pow(scipy.stats.norm.rvs(scale=1E-5, size=half_cube_shape),2)

	k_cube=numpy.fft.ifftshift(k_cube, axes=(0,1))
	var1= math.sqrt(numpy.sum(numpy.abs(k_cube)))


	k_cube=numpy.sqrt(numpy.abs(k_cube))

	m1=numpy.real(cube[0,0,0])
	cube=cube*k_cube[:,:,:cube_half_length+1]

	# take fft

	fft_cube=numpy.fft.irfftn(cube)
	fft_cube=numpy.fft.fftshift(fft_cube)

	# exponentiate

	#max1=numpy.max(fft_cube[:cube_half_length/2])
	mean=0#m1*0.38147#numpy.mean(fft_cube)
	std=var1/pow(2*cube_half_length,3)#*1.377

	if m_func:
		m=numpy.fromfunction(m_func , cube_shape) #0
	else:
		m=0

	if s_func:
		s=numpy.fromfunction(s_func, cube_shape) #1
		s*=sigma/numpy.mean(s)
	else:
		s=sigma

	#print numpy.std(((fft_cube-mean)/(std))*s )

	#print mean, numpy.mean(fft_cube), std, numpy.std(fft_cube)

	cube=numpy.exp( m + ((fft_cube-mean)/(std))*s  )

	#f_cube=(numpy.fft.rfftn(cube))
	#return numpy.fft.ifftshift(numpy.abs(f_cube*numpy.conj(f_cube)), axes=(0,1))

	return cube

def corr_func(i,j,k,cube_half_length, beta, outer_scale, sigma):

	mpmath.mp.dps = 10

	x=math.sqrt(pow(i-cube_half_length,2)+pow(j-cube_half_length,2)+pow(k-cube_half_length,2))
	y=((math.exp(pow(sigma,2))-1)/2)*mpmath.re( pow(x, beta-3) *( pow(1j, 3-beta) * mpmath.gammainc(2-beta, a=-x*1.0j/2.54) + pow(-1j, 3-beta) * mpmath.gammainc(2-beta, a=x*1.0j/2.54) ) ) + 1
	y=mpmath.log(y)
	return float(mpmath.nstr(y))


def cube_make_FFT2(cube_half_length, beta, outer_scale, sigma):
	half_cube_shape=cube_half_length*2,cube_half_length*2,cube_half_length+1
	cube_shape=cube_half_length*2,cube_half_length*2,cube_half_length*2
	double_cube_shape=cube_half_length*4,cube_half_length*4,cube_half_length*4

	cube=scipy.stats.uniform.rvs(loc=0, scale=2*math.pi, size=half_cube_shape)

	cube=numpy.cos(cube)+1j*numpy.sin(cube)	

	k0=cube_half_length/(outer_scale*math.pi)

	xx=numpy.linspace(1E-9, math.sqrt(3)*cube_half_length, num=1001)
	yy=[]
	for x in xx:
		y=((math.exp(pow(sigma,2))-1)/(-2))*mpmath.re( pow(x, beta-3) *( pow(1j, 3-beta) * mpmath.gammainc(2-beta, a=-x*1.0j/k0) + pow(-1j, 3-beta) * mpmath.gammainc(2-beta, a=x*1.0j/k0) ) ) + 1
		y=mpmath.log(y/pow(sigma,2))
		yy.append( float(mpmath.nstr(y)) )
		#print x, y
	xx[0]=0.

	S=scipy.interpolate.InterpolatedUnivariateSpline(xx, yy)


	k_cube=numpy.zeros(cube_shape)
	for i in range(k_cube.shape[0]):
		for j in range(k_cube.shape[1]):
			for k in range(k_cube.shape[2]):
				#k_cube[i,j,k]=corr_func(i,j,k,cube_half_length, beta, outer_scale, sigma)
				dist=math.sqrt(pow(i-cube_half_length,2)+pow(j-cube_half_length,2)+pow(k-cube_half_length,2))#/(math.sqrt(3)*cube_half_length)
				k_cube[i,j,k]=S(dist)#yy[int(dist)]		

	k_cube=numpy.fft.fftshift(k_cube)
	k_cube=numpy.fft.rfftn(k_cube)

	var1=math.sqrt((numpy.sum(k_cube)))

	k_cube=numpy.sqrt(numpy.abs(k_cube))

	m1=numpy.real(cube[0,0,0])
	cube=cube*k_cube[:,:,:cube_half_length+1]

	# take fft

	fft_cube=numpy.fft.irfftn(cube)
	fft_cube=numpy.fft.fftshift(fft_cube)

	# exponentiate

	#max1=numpy.max(fft_cube[:cube_half_length/2])
	mean=0#m1*0.38147#numpy.mean(fft_cube)
	std=5.25E-6*var1#numpy.std(fft_cube)


	#print mean, numpy.mean(fft_cube), std, numpy.std(fft_cube)

	cube=numpy.exp((fft_cube-mean)*sigma/(std))

	#f_cube=(numpy.fft.rfftn(cube))
	#return numpy.fft.ifftshift(numpy.abs(f_cube*numpy.conj(f_cube)), axes=(0,1))

	return cube
