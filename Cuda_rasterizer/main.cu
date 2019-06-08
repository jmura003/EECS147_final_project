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

int main(int argc, char * argv[]){

    cudaError_t cuda_ret;

    Vec2 v0 = {491.407, 411.407}; 
    Vec2 v1 = {148.593, 68.5928}; 
    Vec2 v2 = {148.593, 411.407}; 


    float * v0_h = new float[2];
    float * v1_h = new float[2];
    float * v2_h = new float[2];
    // float * v0_h;
    // float * v1_h;
    // float * v2_h;
    float * v0_d;
    float * v1_d;
    float * v2_d;

    // v0_h = (float*)malloc(sizeof(float)*2);
    // v1_h = (float*)malloc(sizeof(float)*2);
    // v2_h = (float*)malloc(sizeof(float)*2);

    v0_h[0] = v0[0]; v0_h[1] = v0[1];
    v1_h[0] = v1[0]; v1_h[1] = v1[1];
    v2_h[0] = v2[0]; v2_h[1] = v2[1];
    printf("v0_h: ");
    printf("%f, %f\n",v0_h[0],v0_h[1]);

    const int SIZE = w * h * 3;
    //Rgb * framebuffer = new Rgb[w * h]; //host variable
    //int f_sz = w * h;

    //float ** framebuffer = new float[f_sz + 3];
    //float ** framebuffer = new float*[f_sz];
    //float ** framebuffer = (float**)malloc(f_sz);
    unsigned char * framebuffer = new unsigned char[w * h * 3];
    memset(framebuffer, 0x0, w * h * 3); 

    printf("allocating memory for frame buffer...\n"); fflush(stdout);

    //Rgb * framebuffer_device;
    unsigned char * framebuffer_device;// = new float[w*h*3][3];
    cudaMalloc((void**) &framebuffer_device, w * h * 3);
    cudaMalloc((void**) &v0_d, sizeof(float) * 2);
    cudaMalloc((void**) &v1_d, sizeof(float) * 2);
    cudaMalloc((void**) &v2_d, sizeof(float) * 2);

    cudaDeviceSynchronize();

    //copying host to device
    printf("copying host to device...\n"); fflush(stdout);
    cudaMemcpy(framebuffer_device, framebuffer, w * h * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(v0_d, v0_h, sizeof(float) * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(v1_d, v1_h, sizeof(float) * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(v2_d, v2_h, sizeof(float) * 2, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    //printf("v0: %f, %f\n",v0_d[0],v0_d[1]);
    //invoke kernel here

    printf("running the kernel....\n");

    basicTriRast(framebuffer_device, v0_d, v1_d, v2_d, w, h, SIZE);


    cuda_ret = cudaDeviceSynchronize();
	if(cuda_ret != cudaSuccess) FATAL("Unable to launch kernel"); fflush(stdout);

    cudaMemcpy(framebuffer, framebuffer_device, w * h * 3, cudaMemcpyDeviceToHost);
    cudaMemcpy(v0_h, v0_d, sizeof(float) * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(v1_h, v1_d, sizeof(float) * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(v2_h, v2_d, sizeof(float) * 2, cudaMemcpyDeviceToHost);

    //std::cout << "v0: " << v0_h[0] << std::endl;

    // std::ofstream fb;
    // fb.open("./framebuffer.txt");
    // for(int j = 0; j < w; j++){
    //     for(int i = 0; i < h; i++){
    //         fb << "r: " << (float)framebuffer[i + j*w] << "\n";
    //         fb << "g: " << (float)framebuffer[i + j*w + 1] << "\n";
    //         fb << "b: " << (float)framebuffer[i + j*w + 2] << "\n";
    //     }
    // }
    // fb.close();
    // Rgb * new_framebuffer = new Rgb[w * h]; 
    // for(int i = 0; i < 512; i++){
    //     for(int j = 0; j < 512; j++){
    //         new_framebuffer[j + i*w][0] = framebuffer[j + i*w*3];
    //         new_framebuffer[j + i*w][1] = framebuffer[j + i*w*3 + 1];
    //         new_framebuffer[j + i*w][2] = framebuffer[j + i*w*3 + 2];
    //     }
    // }
    //memcpy(new_framebuffer,framebuffer,w * h * 3);


    cudaDeviceSynchronize();

    //verify results below

    std::ofstream ofs; 
    ofs.open("./raster2d.ppm"); 
    ofs << "P6\n" << w << " " << h << "\n255\n"; 
    ofs.write((char*)framebuffer, w * h * 3); 
    ofs.close();


    delete [] framebuffer;
    free(v0_h);
    free(v1_h);
    free(v2_h);
    cudaFree(framebuffer_device);
    cudaFree(v0_d);
    cudaFree(v1_d);
    cudaFree(v2_d);



    return 0;
}