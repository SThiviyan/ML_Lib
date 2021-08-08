//
//  ImageReader.cpp
//  ML_Lib
//
//  Created by Thiviyan Saravanamuthu on 27.07.21.
//

#include "ImageReader.hpp"
#include <fstream>
#include <unistd.h>

#define STB_IMAGE_IMPLEMENTATION
#include "extern/stb_image.h"



int flipint (int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    
    ch1=i&255;
    ch2=(i>>8)&255;
    ch3=(i>>16)&255;
    ch4=(i>>24)&255;
    
    return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}

void ML_Lib::ImageReader::png_to_matrix(std::vector<Matrix> &ImageMatrices, std::string imagepath, int imagewidth, int imageheight, colorchannels channels)
{
    ImageMatrices.clear();
    int numchannels = 4;
    int outputtedchannels = 3;
    unsigned char *image = stbi_load(imagepath.c_str(), &imagewidth, &imageheight, &numchannels, outputtedchannels);
    
    
    for (int channel = 1; channel <= outputtedchannels; channel++) {
        Matrix FinalMatrix(imageheight, imagewidth);
        for (int n = 0; n < imageheight; n++) {
            for (int j = 0; j < imagewidth; j++) {
                FinalMatrix(n, j) = int(image[n * j * channel]);
            }
        }
        ImageMatrices.push_back(FinalMatrix);
    }
    
    
    delete image;
}


void ML_Lib::ImageReader::return_mnist_dataset(std::string imagepath, std::string labelpath, std::vector<std::vector<float>> &ImageVectors, std::vector<float> &LabelVectors, bool normalizeVals)
{
    if(access(imagepath.c_str(), W_OK) != 0)
    {
        throw std::runtime_error("ImagePath not accessible");
    }
    
    
    if(access(labelpath.c_str(), W_OK) != 0)
    {
        throw std::runtime_error("LabelPath not accessible");
    }
    
    ImageVectors.clear();
    LabelVectors.clear();
    
    std::ifstream images(imagepath.c_str(), std::ios::binary);
    std::ifstream labels(labelpath.c_str(), std::ios::binary);
    
    
    if(images.is_open())
    {
        int magicnum = 0;
        int numofimages = 0;
        int picrows = 0;
        int piccols = 0;
        
        images.read((char*)&magicnum, sizeof(magicnum));
        magicnum = flipint(magicnum);
         
        if(magicnum != 2051)
        {
           throw std::runtime_error("Not a MNIST Image file!");
        }
        
        
        
        images.read((char*)&numofimages, sizeof(numofimages));
        numofimages = flipint(numofimages);
        
        images.read((char*)&picrows, sizeof(picrows));
        picrows = flipint(picrows);
        
        images.read((char*)&piccols, sizeof(piccols));
        piccols = flipint(piccols);
        
        ImageVectors.resize(numofimages, std::vector<float>());
        
        for (size_t n = 0; n < numofimages; n++)
        {
            for (size_t j = 0; j < picrows; j++)
            {
                
                //ImageVectors.push_back(std::vector<float>());
                
                for (int m = 0; m < piccols; m++)
                {
                    
                    char temp = 0;
                    images.read((char*)&temp, sizeof(temp));
                    //temp = flipint(temp);
                    
                    float num = (float)temp;
                    
                    if(normalizeVals)
                        num = num / 255;
                    
                    
                    ImageVectors[n].push_back(num);
                }
            }
        }
        
    }
    
    images.close();

    
    if(labels.is_open())
    {
        int magicnum = 0;
        int numoflabels = 0;
        
        labels.read((char*)&magicnum, sizeof(magicnum));
        magicnum = flipint(magicnum);
        
        
        labels.read((char*)&numoflabels, sizeof(numoflabels));
        numoflabels = flipint(numoflabels);
        
        for (unsigned int n = 0; n < numoflabels; n++) {
            char temp = 0;
            labels.read((char*)&temp, sizeof(temp));
            LabelVectors.push_back((float)temp);
        }
        
    }
    
    labels.close();
}
 
