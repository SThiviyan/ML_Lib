//
//  ImageReader.cpp
//  ML_Lib
//
//  Created by Thiviyan Saravanamuthu on 27.07.21.
//

#include "ImageReader.hpp"
#include <fstream>

int flipint (int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    
    ch1=i&255;
    ch2=(i>>8)&255;
    ch3=(i>>16)&255;
    ch4=(i>>24)&255;
    
    return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}

std::vector<ML_Lib::Matrix> ML_Lib::ImageReader::png_to_matrix(std::string imagepath, bool IncludeAllChannels)
{
    std::vector<Matrix> Matrices;
 
    
    
    
    
    return Matrices;
}


void ML_Lib::ImageReader::return_mnist_dataset(std::string imagepath, std::string labelpath, std::vector<std::vector<float>> &ImageVectors, std::vector<float> &LabelVectors)
{
    std::fstream images;
    std::fstream labels;
    
    images.open(imagepath, std::ios::binary);
    labels.open(labelpath, std::ios::binary);
    
    if(images.is_open())
    {
        int magicnum = 0;
        int numofimages = 0;
        int picrows = 0;
        int piccols = 0;
        
        images.read((char*)&magicnum, sizeof(magicnum));
        magicnum = flipint(magicnum);
        
        images.read((char*)&numofimages, sizeof(numofimages));
        numofimages = flipint(numofimages);
        
        images.read((char*)&picrows, sizeof(picrows));
        picrows = flipint(picrows);
        
        images.read((char*)&piccols, sizeof(piccols));
        piccols = flipint(piccols);
        
        for (int n = 0; n < numofimages; n++)
        {
            for (int j = 0; j < picrows; j++)
            {
                
                ImageVectors.push_back(std::vector<float>());
                
                for (int m = 0; m < piccols; m++)
                {
                    char temp = 0;
                    images.read((char*)&temp, sizeof(temp));
                    ImageVectors[j].push_back(temp);
                }
                
            }
        }
        
    }
    
    if(labels.is_open())
    {
        int magicnum = 0;
        int numoflabels = 0;
        
        images.read((char*)&magicnum, sizeof(magicnum));
        magicnum = flipint(magicnum);
        
        images.read((char*)&numoflabels, sizeof(numoflabels));
        numoflabels = flipint(numoflabels);
        
        for (int n = 0; n < numoflabels; n++) {
            char num = 0;
            images.read((char*)&num, sizeof(num));
            LabelVectors.push_back(num);
        }
        
    }
    
    
    
    images.close();
    labels.close();
}
 
