import array
import numpy
import sys
import math
import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
from PIL import Image

mod = SourceModule("""
#include <stdio.h>

__global__ void rasterizePixel(float *v0, float *v1, float *v2, int *c0, int *c1, int *c2, float *image)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int j = ty + blockIdx.y * blockDim.y; // rows
    int i = tx + blockIdx.x * blockDim.x; // cols

    float area = (v2[0] - v0[0]) * (v1[1] - v0[1]) - (v2[1] - v0[1]) * (v1[0] - v0[0]);
    float p[2] = {i + 0.5f, j + 0.5f};
    int index = (i + j * 512) * 3;
    float alpha = (p[0] - v1[0]) * (v2[1] - v1[1]) - (p[1] - v1[1]) * (v2[0] - v1[0]);
    float beta = (p[0] - v2[0]) * (v0[1] - v2[1]) - (p[1] - v2[1]) * (v0[0] - v2[0]);
    float gamma = (p[0] - v0[0]) * (v1[1] - v0[1]) - (p[1] - v0[1]) * (v1[0] - v0[0]);

    //if(threadIdx.x == 0) {
        //printf("hi");
    //}

    if(v0[0] > 400) {
        image[index] = 255;
    }

    if(alpha >= 0 && beta >= 0 && gamma >= 0){
      alpha = alpha / area;
      beta = beta / area;
      gamma = gamma / area;
      float r = alpha;
      float g = beta;
      float b = gamma;
      if(i < 512 && j < 512)
      {
        image[index] = (r * 255);
        image[index + 1] = (g * 255);
        image[index + 2] = (b * 255);
      }
    }
}
  """)

w = 512
h = 512
maxval = 255

# PPM file header
ppm_header = 'P6 ' + str(w) + ' ' + str(h) + ' ' + str(maxval) + '\n'

v0 = numpy.array([491.407, 411.407]).astype(numpy.float32)
v1 = numpy.array([148.593, 68.5928]).astype(numpy.float32)
v2 = numpy.array([148.593, 411.407]).astype(numpy.float32)
c0 = numpy.array([1, 0, 0]).astype(numpy.float32)
c1 = numpy.array([0, 1, 0]).astype(numpy.float32)
c2 = numpy.array([0, 0, 1]).astype(numpy.float32)

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
#image = numpy.array([0, 0, 0] * w * h)#.astype(numpy.float32)
image = numpy.zeros(3*w*h).astype(numpy.float32)
#image = numpy.full(3*w*h, 0).astype(numpy.float32)

# image_orig = image.copy()
# image_gpu = gpuarray.to_gpu(image)

# image_gpu = cuda.mem_alloc(sys.getsizeof(image))
# cuda.memcpy_htod(image_gpu, image)

func = mod.get_function("rasterizePixel")
gridParam = int(math.ceil(512/32))
print(gridParam)
# func(v0_gpu, v1_gpu, v2_gpu, c0_gpu, c1_gpu, c2_gpu, cuda.InOut(image), grid=(gridParam, gridParam, 1), block=(32,32,1))
func(cuda.InOut(v0), cuda.InOut(v1), cuda.InOut(v2), cuda.InOut(c0), cuda.InOut(c1), cuda.InOut(c2), cuda.InOut(image), grid=(gridParam, gridParam, 1), block=(32,32,1))

# cuda.memcpy_dtoh(imagenew, image_gpu)

# for i in image:
#     if i != 0:
#         print(i)

# image = image_gpu.get()

# Save the PPM image as a binary file
with open('temp.ppm', 'wb') as f:
    f.write(bytearray(ppm_header, 'ascii'))
    image.tofile(f)
    im = Image.open("temp.ppm")
    im.save("testpic.png")

# __device__ inline
# float edgeFunction(const float*a, const float*b, const float*c)
# { return (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0]); }
