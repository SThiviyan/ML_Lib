//
//  Convolutional_Layer.hpp
//  ML_Lib
//
//  Created by Thiviyan Saravanamuthu on 26.07.21.
//

#ifndef Convolutional_Layer_hpp
#define Convolutional_Layer_hpp

#include <iostream>
#include <vector>
#include "Matrix.hpp"

namespace ML_Lib
{
    class Conv_Layer
    {
    public:
        Conv_Layer();
        
        
    private:
        std::vector<Matrix*> InputImages; //Input Images (3 matrices for 3 Color Channels RGB)
        std::vector<Matrix*> Filters; // Filters for 3 Channels
        int filterWidth, filterHeight; // Filter Size
        int Stride; // Movement of the Filter
        
    };


}


#endif /* Convolutional_Layer_hpp */
