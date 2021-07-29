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
    
    class ImageReader
    {
    public:
        
        std::vector<Matrix> png_to_matrix(std::string imagepath, bool IncludeAllChannels);
        std::vector<Matrix> svg_to_matrix(std::string imagepath, bool IncludeAllChannels);
        std::vector<Matrix> jpeg_to_matrix(std::string imagepath, bool IncludeAllChannels);
        void return_mnist_dataset(std::string imagepath, std::string labelpath, std::vector<std::vector<float>> &Images, std::vector<float> &Labels);
        
    };

}


#endif /* ImageReader_hpp */
