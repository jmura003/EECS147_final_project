import array
import numpy
import sys
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from PIL import Image

mod = SourceModule("""
  __global__ void rasterizePixel(float *v0, float *v1, float *v2, int *c0, int *c1, int *c2, int *image)
  {
    int index = threadIdx.x + threadIdx.y*3;
    float p[2] = {threadIdx.x + 0.5f, threadIdx.y + 0.5f};
    float area = (v2[0] - v0[0]) * (v1[1] - v0[1]) - (v2[1] - v0[1]) * (v1[0] - v0[0]);
    float w0 = (p[0] - v1[0]) * (v2[1] - v1[1]) - (p[1] - v1[1]) * (v2[0] - v1[0]);
    float w1 = (p[0] - v2[0]) * (v0[1] - v2[1]) - (p[1] - v2[1]) * (v0[0] - v2[0]);
    float w2 = (p[0] - v0[0]) * (v1[1] - v0[1]) - (p[1] - v0[1]) * (v1[0] - v0[0]);
    if (w0 >= 0 && w1 >= 0 && w2 >= 0) {
        w0 /= area;
        w1 /= area;
        w2 /= area;
        float r = w0 * c0[0] + w1 * c1[0] + w2 * c2[0];
        float g = w0 * c0[1] + w1 * c1[1] + w2 * c2[1];
        float b = w0 * c0[2] + w1 * c1[2] + w2 * c2[2];
        if(index % 3 == 0) {
            image[index] = int(r*255); //looking at red value
        }
        else if(index % 3 == 1) {
            image[index] = int(g*255); //looking at green value
        }
        else if(index % 3 == 2) {
            image[index] = int(b*255); //looking at blue value
        }
    }
  }
  """)

w = 512
h = 512
maxval = 255

# PPM file header
ppm_header = 'P6 ' + str(w) + ' ' + str(h) + ' ' + str(maxval) + '\n'

v0 = numpy.array([491.407, 411.407])
v1 = numpy.array([148.593, 68.5928])
v2 = numpy.array([148.593, 411.407])
c0 = numpy.array([1, 0, 0])
c1 = numpy.array([0, 1, 0])
c2 = numpy.array([0, 0, 1])

v0_gpu = cuda.mem_alloc(sys.getsizeof(v0))
cuda.memcpy_htod(v0_gpu, v0)

v1_gpu = cuda.mem_alloc(sys.getsizeof(v1))
cuda.memcpy_htod(v1_gpu, v1)

v2_gpu = cuda.mem_alloc(sys.getsizeof(v2))
cuda.memcpy_htod(v2_gpu, v2)

c0_gpu = cuda.mem_alloc(sys.getsizeof(c0))
cuda.memcpy_htod(c0_gpu, c0)

c1_gpu = cuda.mem_alloc(sys.getsizeof(c1))
cuda.memcpy_htod(c1_gpu, c1)

c2_gpu = cuda.mem_alloc(sys.getsizeof(c2))
cuda.memcpy_htod(c2_gpu, c2)

# PPM image data (filled with black)
image = numpy.array([0, 0, 0] * w * h)
image_gpu = cuda.mem_alloc(sys.getsizeof(image))
cuda.memcpy_htod(image_gpu, image)

func = mod.get_function("rasterizePixel")
func(v0_gpu, v1_gpu, v2_gpu, c0_gpu, c1_gpu, c2_gpu, image_gpu, grid=(512,3), block=(13,13,3))

cuda.memcpy_dtoh(image, image_gpu)

# Save the PPM image as a binary file
with open('temp.ppm', 'wb') as f:
	f.write(bytearray(ppm_header, 'ascii'))
	image.tofile(f)
