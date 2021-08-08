//
//  ImageReader.hpp
//  ML_Lib
//
//  Created by Thiviyan Saravanamuthu on 27.07.21.
//

#ifndef ImageReader_hpp
#define ImageReader_hpp

#include <iostream>
#include <fstream>
#include <vector>
#include "SharedDep/Matrix/Matrix.hpp"

namespace ML_Lib {
    
    enum colorchannels
    {
        RGB,
        RGBA
    };
    
    class ImageReader
    {
        
    public:
        
        static void png_to_matrix(std::vector<Matrix> &ImageMatrices,std::string imagepath, int imagewidth, int imageheight, colorchannels channels);
        std::vector<Matrix> svg_to_matrix(std::string imagepath, bool IncludeAllChannels);
        std::vector<Matrix> jpeg_to_matrix(std::string imagepath, bool IncludeAllChannels);
        static void return_mnist_dataset(std::string imagepath, std::string labelpath, std::vector<std::vector<float>> &Images, std::vector<float> &Labels, bool normalizeVals);
        
    };

}


#endif /* ImageReader_hpp */
