//
//  Convolutional_Layer.cpp
//  ML_Lib
//
//  Created by Thiviyan Saravanamuthu on 26.07.21.
//

#include "Convolutional_Layer.hpp"

ML_Lib::Conv_Layer::Conv_Layer(Conv_Layer* PreviousLayer, std::vector<Matrix>* OptionalInput)
:InputImages(*OptionalInput)
{
    if(PreviousLayer != nullptr)
    {
        InputImages = PreviousLayer->GetOutputArray();
    }
    InitFilters(int(InputImages.size()));
}

void ML_Lib::Conv_Layer::InitFilters(int NumImages)
{
    filterWidth = 3;
    filterHeight = 3;
    
    for (int n = 0; n < NumImages * 2; n++) {
        Filters.push_back(Matrix(filterHeight, filterWidth));
        Filters[n].RandonWeightInitwithRange(-10, 10);
    }
    
}


void ML_Lib::Conv_Layer::feedforward(ML_Lib::ActivationFunction Ac)
{
    int runs = int(Filters.size() / InputImages.size());
    for (int n = 1; n <= runs; n++)
    {
        for (int filterindex = 0; filterindex < 3; filterindex++) {
            
        }
    }
    
}


ML_Lib::Matrix ML_Lib::Conv_Layer::CalculateSingleFeatureMap(Matrix &Filter, Matrix &ChannelImage)
{
    
    
    Matrix M(2, 2);
    
    return M;
}




//MARK: GET Functions

std::vector<ML_Lib::Matrix> ML_Lib::Conv_Layer::GetInputImages()
{
    
    return InputImages;
}


std::vector<ML_Lib::Matrix> ML_Lib::Conv_Layer::GetFilters()
{
    
    return std::vector<ML_Lib::Matrix>();
}


std::vector<ML_Lib::Matrix> ML_Lib::Conv_Layer::GetOutputArray()
{
    
    return std::vector<ML_Lib::Matrix>();
}
