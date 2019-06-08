typedef unsigned char Rgb[3]; 
typedef float Vec2[2]; 
typedef float Vec3[3]; 

#include <stdio.h>

#define ww 16
#define hh 16

__device__ inline 
float edgeFunction(const Vec2 &a, const Vec2 &b, const Vec2 &c) 
{ return (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0]); } 

__global__ void rasterize_triangle(unsigned char * framebuffer_d, const float * v0_d, const float * v1_d, const float * v2_d, 
    const int w, const int h){

        
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int j = ty + blockIdx.y * blockDim.y; // rows
        int i = tx + blockIdx.x * blockDim.x; // cols

 

        Vec2 a = {v0_d[0], v0_d[1]};
        Vec2 b = {v1_d[0], v1_d[1]};
        Vec2 c = {v2_d[0], v2_d[1]};

        float area = edgeFunction(a,b,c);
        //for(int j = threadIdx.y; j < w; j += blockDim.y){
        //    for(int i = threadIdx.x; i < h; i += blockDim.x){
        //for(int idx = 0; idx < w; idx++){
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
            //}
            //}
        //}

}

void basicTriRast(unsigned char * framebuffer_d, const float * v0_d, const float * v1_d, const float * v2_d, const int w, const int h, const int SIZE){
    //dim3 DimGrid(ceil((w*h)/512.0),1,1);
    const unsigned int BLOCK_SIZE = 32;
    //ceil(double(512)/BLOCK_SIZE)
    dim3 BlocksPerGrid(ceil(double(512)/BLOCK_SIZE),ceil(double(512)/BLOCK_SIZE),1);
    dim3 ThreadsPerBlock(BLOCK_SIZE,BLOCK_SIZE,1);
    // dim3 ThreadsPerBlock(256,1,1);
    // dim3 BlocksPerGrid(ceil(double(SIZE)/BLOCK_SIZE),1,1);
    rasterize_triangle<<<BlocksPerGrid, ThreadsPerBlock>>>(framebuffer_d, v0_d, v1_d, v2_d, w, h);
}