// c++ -o raster2d raster2d.cpp
// (c) www.scratchapixel.com

#include <iostream> 
#include <cstdio> 
#include <cstdlib> 
#include <fstream> 
#include <string.h>
#include <time.h>
#include <chrono>
 
typedef float Vec2[2]; 
typedef float Vec3[3]; 
typedef unsigned char Rgb[3];

using namespace std::chrono;
 
inline 
float edgeFunction(const Vec2 &a, const Vec2 &b, const Vec2 &c) 
{ return (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0]); } 
 
int main(int argc, char **argv) 
{ 
    srand(217);
    // for(int i = 0; i < 6; i++){
    //     temp[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/512));
    //     std::cout << "here is a random number: " << temp[i] << std::endl;
    // }
    float area = 0;
    // Vec2 v0 = {0,0};
    // Vec2 v1 = {0,0};
    // Vec2 v2 = {0,0};
    int num_triangles = 0;
    if(argc == 1){
        num_triangles = 1;
    }
    else{
        num_triangles = atoi(argv[1]); 
    }
    float x0[num_triangles];
    float x1[num_triangles];
    float x2[num_triangles];
    float y0[num_triangles];
    float y1[num_triangles];
    float y2[num_triangles];
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
    

    Vec3 c0 = {1, 0, 0}; 
    Vec3 c1 = {0, 1, 0}; 
    Vec3 c2 = {0, 0, 1}; 
 
    const uint32_t w = 512; 
    const uint32_t h = 512; 
 
    Rgb *fffff = new Rgb[w * h]; 
    unsigned char * framebuffer = new unsigned char[w * h * 3];
    memset(framebuffer, 0x0, w * h * 3); 
 
    //std::cout << "area: " << area << endl;

    auto start = high_resolution_clock::now();

    for(int k = 0; k < num_triangles; k++){
        Vec2 v0 = {x0[k], y0[k]}; 
        Vec2 v1 = {x1[k], y1[k]}; 
        Vec2 v2 = {x2[k], y2[k]}; 
        area = edgeFunction(v0,v1,v2);
        for (uint32_t j = 0; j < h; ++j) { 
            for (uint32_t i = 0; i < w; ++i) { 
                Vec2 p = {i + 0.5f, j + 0.5f}; 
                float w0 = edgeFunction(v1, v2, p); 
                float w1 = edgeFunction(v2, v0, p); 
                float w2 = edgeFunction(v0, v1, p); 
                if (w0 >= 0 && w1 >= 0 && w2 >= 0) { 
                    w0 /= area; 
                    w1 /= area; 
                    w2 /= area; 
                    // float r = w0 * c0[0] + w1 * c1[0] + w2 * c2[0]; 
                    // float g = w0 * c0[1] + w1 * c1[1] + w2 * c2[1]; 
                    // float b = w0 * c0[2] + w1 * c1[2] + w2 * c2[2]; 
                    float r = w0; 
                    float g = w1;
                    float b = w2;
                    fffff[j * w + i][0] = (unsigned char)(r * 255); 
                    fffff[j * w + i][1] = (unsigned char)(g * 255); 
                    fffff[j * w + i][2] = (unsigned char)(b * 255); 
                    framebuffer[(j * w + i)*3] = (unsigned char)(r * 255); 
                    framebuffer[(j * w + i)*3 + 1] = (unsigned char)(g * 255); 
                    framebuffer[(j * w + i)*3 + 2] = (unsigned char)(b * 255); 

                } 
            } 
        }
    } 
    auto stop = high_resolution_clock::now();

    auto duration = duration_cast<microseconds>(stop-start);

    std::cout << "time on cpu for " << num_triangles << " triangles: ";
    std::cout << duration.count() << " micro seconds" << std::endl;
 
    std::ofstream ofs; 
    ofs.open("./raster2d_cpu.ppm"); 
    ofs << "P6\n" << w << " " << h << "\n255\n"; 
    ofs.write((char*)framebuffer, w * h * 3); 
    ofs.close(); 
 
    delete [] framebuffer; 
    //delete [] fbwf;
    delete [] fffff;
 
    return 0; 
} 