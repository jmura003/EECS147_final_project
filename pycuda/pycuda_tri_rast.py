import array
import numpy
import sys
import math
import random
import time
import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
from PIL import Image

mod = SourceModule("""
__device__ inline
float edgeFunction(const float a[], const float b[], const float c[])
{ return (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0]); }

__global__ void rasterizePixel(float* triangles, float *image, int size)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int j = ty + blockIdx.y * blockDim.y; // rows
    int i = tx + blockIdx.x * blockDim.x; // cols

    float p[2] = {i + 0.5f, j + 0.5f};
    for(int k = 0; k < size*6; k+=6)
    {
      float v0[2] = {triangles[k], triangles[k + 1]};
      float v1[2] = {triangles[k + 2], triangles[k + 3]};
      float v2[2] = {triangles[k + 4], triangles[k + 5]};
      float area = edgeFunction(v0, v1, v2);
      int index = (i + j * 512) * 3;
      float alpha = edgeFunction(v1, v2, p);
      float beta = edgeFunction(v2, v0, p);
      float gamma = edgeFunction(v0, v1, p);

      __syncthreads();

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

      __syncthreads();
    }
}
""")

w = 512
h = 512
maxval = 255

triangles = []

# PPM file header
ppm_header = 'P6 ' + str(w) + ' ' + str(h) + ' ' + str(maxval) + '\n'

def edgeFunction(a,b,c):
    return (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0])

def makeTriangles(n, w, h): #Does not prevent creating duplicate triangles
	#n determines number of triangles within a w x h image to generate in an array

    global triangles
    for i in range(0, n):
        check = [[0,0],[0,0],[0,0]]
        checkVal = 0
        while checkVal == 0:
            for j in range(0, 3):
                x = random.uniform(0, w)
                y = random.uniform(0, h)
                # triangles.append(x)
                # triangles.append(y)
                check[j] = [x,y]

            if edgeFunction(check[0], check[1], check[2]) >= 0:
                checkVal = 1

        for k in range(0, 3):
            for l in range(0, 2):
                triangles.append(check[k][l])

    npTriangles = numpy.array(triangles, dtype=numpy.float32)
    return npTriangles

# PPM image data (filled with black)
image = numpy.zeros(3*w*h).astype(numpy.float32)

sync = pycuda.driver.Stream()
func = mod.get_function("rasterizePixel")
gridParam = int(math.ceil(512/32))

sizeInput = input("Enter the number of triangles: ")
trianglesParam = makeTriangles(sizeInput, 512, 512)

size = numpy.int32(sizeInput)
start = time.time()
func(cuda.InOut(trianglesParam), cuda.InOut(image), size, grid=(gridParam, gridParam, 1), block=(32,32,1))
sync.synchronize();

end = time.time()

# Save the PPM image as a binary file
with open('temp.ppm', 'wb') as f:
    f.write(bytearray(ppm_header, 'ascii'))
    image.tofile(f)
    im = Image.open("temp.ppm")
    im.save("testpic.png")

print("Time elapsed for " + str(sizeInput) + " triangles is " + str(end - start))
