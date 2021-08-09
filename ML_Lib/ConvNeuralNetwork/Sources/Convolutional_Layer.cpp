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
    for (int run = 0; run < runs; run++)
    {
        for (int imageindex = 0; imageindex < InputImages.size(); imageindex++) {
            for (int filterindex = run * int(InputImages.size()); filterindex < (run + 1) * int(InputImages.size()); filterindex++) {
                ActivationMaps.push_back(CalculateSingleFeatureMap(Filters[filterindex], InputImages[imageindex], this->Stride));
                //ActivationMaps[imageindex].ActivateNeurons(Ac);
            }
        }
        
        
    }
    
}


ML_Lib::Matrix ML_Lib::Conv_Layer::CalculateSingleFeatureMap(ML_Lib::Matrix &Filter, ML_Lib::Matrix &ChannelImage, int Stride)
{
    
    //CoreProperties for Calculation
    int FilterRowIndex = 0;
    int FilterColumnIndex = 0;
    int FilterBiggestRowIndex  = Filter.getRows();
    int FilterBiggestColumnIndex = Filter.getCols();
    bool needtocalculate = true;
    
    //Stores FeatureMap as vector and returns in the End as Matrix
    std::vector<std::vector<float>> FeatureMap;
    
    //Runs as long calculation is necessary
    while(needtocalculate){
        
        std::vector<float> Temp;

        //Shifts Start and End Indeces to get needed Imageportion
        for (int n = FilterRowIndex; n < FilterBiggestRowIndex; n++) {
            for (int j = FilterColumnIndex; j < FilterBiggestColumnIndex; j++) {
                Temp.push_back(ChannelImage(n, j));
            }
        }
        
        
        std::vector<std::vector<float>> ImagePart = ML_Lib::Matrix::ReturnTwoDimensionalVector(Temp, Filter.getRows(), Filter.getCols()); //Converts One Dimensional Vector into Two
        
        //Calculation of Dot Product and
        Matrix DotProduct = Matrix(ImagePart) * Filter;
        float AddedValue = 0.f;
        
        for (int n = 0; n < DotProduct.getRows(); n++) {
            for (int j = 0; j < DotProduct.getCols(); j++) {
                AddedValue += DotProduct(n, j);
            }
        }
        
        Temp.clear(); //Clears Object which isn't necessary
 
        
        //Looking if further Calculation is necessary
        if(FilterBiggestRowIndex == ChannelImage.getRows() && FilterBiggestColumnIndex == ChannelImage.getCols())
        {
            //Changes the boolean to false to stop the While Loop
            needtocalculate = false;
            
            //pushes the Last Value and Returns the full matrix
            FeatureMap[FilterRowIndex].push_back(AddedValue);
            return Matrix(FeatureMap);
        }
        
        
        //Moves Horizontally if the BiggestColumn Index is Smaller than the Column number of the Image
        if(FilterBiggestColumnIndex < ChannelImage.getCols())
        {
            //checks if its the first run and adds vectors to the vector of vectors
            if(FeatureMap.size() == 0)
            {
                FeatureMap.push_back(std::vector<float>());
            }
            
            //Pushes added Value of Matrix to the FeatureMap
            FeatureMap[FilterRowIndex].push_back(AddedValue);
            
            //Increases Indeces to move the filter further through the Image
            FilterColumnIndex += Stride;
            FilterBiggestColumnIndex +=  Stride;
        }
        //Checks if moving Horizontally is not an option anymore
        else if(FilterBiggestColumnIndex == ChannelImage.getCols())
        {
            //pushes Value to FeatureMap
            FeatureMap[FilterRowIndex].push_back(AddedValue);
            
            //Moves on Pixel down now and starts again at Column 0
            FilterColumnIndex = 0;
            FilterBiggestColumnIndex = FilterColumnIndex + Filter.getCols();
            FilterRowIndex += Stride;
            FilterBiggestRowIndex += Stride;
            
            //Pushes another Vector to move on row down
            FeatureMap.push_back(std::vector<float>());
        }
        
    }
    
    
    return Matrix(FeatureMap);
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
