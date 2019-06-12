#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <string.h>
#include <iostream>
#include "kernel.cu"
#include "support.cu"


typedef float Vec2[2]; 
typedef float Vec3[3]; 
typedef unsigned char Rgb[3]; 

#define w 512
#define h 512

//using namespace std::chrono;

// inline 
// float edgeFunction(const Vec2 &a, const Vec2 &b, const Vec2 &c) 
// { return (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0]); } 
 

int main(int argc, char * argv[]){
    Timer timer;

    srand(217);
    int num_triangles = 0;
    if(argc == 1){
        num_triangles = 1;
    }
    else{
        num_triangles = atoi(argv[1]); 
    }

    cudaError_t cuda_ret;

    // Vec2 v0 = {491.407, 411.407}; 
    // Vec2 v1 = {148.593, 68.5928}; 
    // Vec2 v2 = {148.593, 411.407}; 


    // float * v0_h = new float[2];
    // float * v1_h = new float[2];
    // float * v2_h = new float[2];

    float * x0_d;
    float * x1_d;
    float * x2_d;
    float * y0_d;
    float * y1_d;
    float * y2_d;

    float * x0 = new float[num_triangles];
    float * x1 = new float[num_triangles];
    float * x2 = new float[num_triangles];
    float * y0 = new float[num_triangles];
    float * y1 = new float[num_triangles];
    float * y2 = new float[num_triangles];
    float area = 0;
    for(int i = 0; i < num_triangles; i++){
        do{
            x0[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/512));
            x1[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/512));
            x2[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/512));
            y0[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/512));
            y1[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/512));
            y2[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/512));
            Vec2 v0 = {x0[i], y0[i]}; 
            Vec2 v1 = {x1[i], y1[i]}; 
            Vec2 v2 = {x2[i], y2[i]};  
            area = edgeFunction(v0,v1,v2);
        } while(area < 0);
    }


    unsigned char * framebuffer = new unsigned char[w * h * 3];
    memset(framebuffer, 0x0, w * h * 3); 

    printf("allocating memory for frame buffer...\n"); fflush(stdout);

    //Rgb * framebuffer_device;
    unsigned char * framebuffer_device;// = new float[w*h*3][3];
    cudaMalloc((void**) &framebuffer_device, w * h * 3);
    cudaMalloc((void**) &x0_d, sizeof(float) * num_triangles);
    cudaMalloc((void**) &x1_d, sizeof(float) * num_triangles);
    cudaMalloc((void**) &x2_d, sizeof(float) * num_triangles);
    cudaMalloc((void**) &y0_d, sizeof(float) * num_triangles);
    cudaMalloc((void**) &y1_d, sizeof(float) * num_triangles);
    cudaMalloc((void**) &y2_d, sizeof(float) * num_triangles);

    cudaDeviceSynchronize();

    //copying host to device
    printf("copying host to device..."); fflush(stdout);
    startTime(&timer);
    cudaMemcpy(framebuffer_device, framebuffer, w * h * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(x0_d, x0, sizeof(float) * num_triangles, cudaMemcpyHostToDevice);
    cudaMemcpy(x1_d, x1, sizeof(float) * num_triangles, cudaMemcpyHostToDevice);
    cudaMemcpy(x2_d, x2, sizeof(float) * num_triangles, cudaMemcpyHostToDevice);
    cudaMemcpy(y0_d, y0, sizeof(float) * num_triangles, cudaMemcpyHostToDevice);
    cudaMemcpy(y1_d, y1, sizeof(float) * num_triangles, cudaMemcpyHostToDevice);
    cudaMemcpy(y2_d, y2, sizeof(float) * num_triangles, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n",elapsedTime(timer));

    //printf("v0: %f, %f\n",v0_d[0],v0_d[1]);
    //invoke kernel here

    printf("running the kernel....\n"); fflush(stdout);
    startTime(&timer);


    basicTriRast(framebuffer_device, x0_d, x1_d, x2_d, y0_d, y1_d, y2_d, w, h, num_triangles);

    stopTime(&timer); 
    printf("execution time of kernel for %d triangle/s: ", num_triangles);
    printf("%f s\n",elapsedTime(timer));
    //std::cout << "execution time of kernel for " << num_triangles << " amount of triangles: " << (float)elapsedTime(timer) << std::endl;

    cuda_ret = cudaDeviceSynchronize();
	if(cuda_ret != cudaSuccess) FATAL("Unable to launch kernel"); fflush(stdout);

    cudaMemcpy(framebuffer, framebuffer_device, w * h * 3, cudaMemcpyDeviceToHost);
    // cudaMemcpy(v0_h, v0_d, sizeof(float) * 2, cudaMemcpyDeviceToHost);
    // cudaMemcpy(v1_h, v1_d, sizeof(float) * 2, cudaMemcpyDeviceToHost);
    // cudaMemcpy(v2_h, v2_d, sizeof(float) * 2, cudaMemcpyDeviceToHost);



    cudaDeviceSynchronize();

    //verify results below

    std::ofstream ofs; 
    ofs.open("./raster2d_gpu.ppm"); 
    ofs << "P6\n" << w << " " << h << "\n255\n"; 
    ofs.write((char*)framebuffer, w * h * 3); 
    ofs.close();


    delete [] framebuffer;
    delete [] x0;
    delete [] x1;
    delete [] x2;
    delete [] y0;
    delete [] y1;
    delete [] y2;
    cudaFree(framebuffer_device);
    cudaFree(x0_d);
    cudaFree(x1_d);
    cudaFree(x2_d);
    cudaFree(y0_d);
    cudaFree(y1_d);
    cudaFree(y2_d);
    



    return 0;
}