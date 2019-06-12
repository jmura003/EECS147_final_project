typedef unsigned char Rgb[3]; 
typedef float Vec2[2]; 
typedef float Vec3[3]; 

#include <stdio.h>

#define ww 16
#define hh 16

__device__ __host__ inline 
float edgeFunction(const Vec2 &a, const Vec2 &b, const Vec2 &c) 
{ return (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0]); } 

__global__ void rasterize_triangle(unsigned char * framebuffer_d, const float * x0_d, const float * x1_d, const float * x2_d, 
    const float * y0_d, const float * y1_d, const float * y2_d, const int w, const int h, const int num_triangles){

        for(int k = 0; k < num_triangles; k++){
            
            
            int tx = threadIdx.x;
            int ty = threadIdx.y;
            int j = ty + blockIdx.y * blockDim.y; // rows
            int i = tx + blockIdx.x * blockDim.x; // cols

            Vec2 a = {x0_d[k], y0_d[k]};
            Vec2 b = {x1_d[k], y1_d[k]};
            Vec2 c = {x2_d[k], y2_d[k]};


            float area = edgeFunction(a,b,c);
        
            Vec2 p = {i + 0.5f, j + 0.5f};
            int index = (i + j * w)*3;   
            float alpha = edgeFunction(b,c,p);
            float beta = edgeFunction(c,a,p);
            float gamma = edgeFunction(a,b,p);
            if(alpha >= 0 && beta >= 0 && gamma >= 0){
                alpha = alpha / area;
                beta = beta / area;
                gamma = gamma / area;
                float r = alpha;
                float g = beta;
                float bb = gamma;
                if(i < 512 && j < 512){
                    framebuffer_d[index] = (unsigned char)(r * 255);
                    framebuffer_d[index + 1] = (unsigned char)(g * 255);
                    framebuffer_d[index + 2] = (unsigned char)(bb * 255);
                }
            }
        }

}

void basicTriRast(unsigned char * framebuffer_d, const float * x0_d, const float * x1_d, const float * x2_d,
    const float * y0_d, const float * y1_d, const float * y2_d,  const int w, const int h, const int num_triangles){
    const unsigned int BLOCK_SIZE = 32;
    
    dim3 BlocksPerGrid(ceil(double(512)/BLOCK_SIZE),ceil(double(512)/BLOCK_SIZE),1);
    dim3 ThreadsPerBlock(BLOCK_SIZE,BLOCK_SIZE,1);
    
    rasterize_triangle<<<BlocksPerGrid, ThreadsPerBlock>>>(framebuffer_d, x0_d, x1_d, x2_d, y0_d, y1_d, y2_d, w, h, num_triangles);
}