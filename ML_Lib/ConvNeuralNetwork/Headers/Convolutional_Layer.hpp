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
#include "SharedDep/Matrix/Matrix.hpp"

namespace ML_Lib
{
    class Conv_Layer
    {
    public:
        //MARK: Initialization
        Conv_Layer(Conv_Layer* PreviousLayer, std::vector<Matrix>* OptionalInput); //Constructor
        void InitFilters(int NumImages); // Creates Filters for this Layer
        void AddPadding(); // Increasing Image Size
        
        
        //MARK: Calculations
        void feedforward(ActivationFunction Ac); //forward propagation function
        static Matrix CalculateSingleFeatureMap(Matrix &Filter, Matrix &ChannelImage, int Stride); // Calculates One Feature Map. 1 filter on one Image and pushes to ActivationMaps
        
        
        
        //MARK: Get Functions
        std::vector<Matrix> GetInputImages();
        std::vector<Matrix> GetFilters();
        std::vector<Matrix> GetOutputArray();
     
        
    private:
        std::vector<Matrix> &InputImages; //Input Images (3 matrices for 3 Color Channels RGB)
        std::vector<Matrix> Filters; // Filters for 3 Channels
        std::vector<Matrix> ActivationMaps; //OutputArray (Last Calculation of the Layer)
        int filterWidth, filterHeight; // Filter Size
        int Stride; // Movement of the Filter
        
    };


}


#endif /* Convolutional_Layer_hpp */
