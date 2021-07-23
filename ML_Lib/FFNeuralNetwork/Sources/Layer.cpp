//
//  Layer.cpp
//  ML_Lib
//
//  Created by Thiviyan Saravanamuthu on 23.07.21.
//

#include "Layer.hpp"


ML_Lib::Layer::Layer(int NeuronNum, int topologyNextElement, Layer* PreviousLayer)
{
    this->NeuronNum = NeuronNum;
    this->topologyNext = topologyNextElement;
    ValMatrix = new Matrix(NeuronNum, 1);
    
    if(PreviousLayer != nullptr)
    {
        this->PreviousLayer = PreviousLayer;
    }
    
    
    if(topologyNextElement != 0)
    {
        WeightMatrix = new Matrix(topologyNextElement, NeuronNum);
        WeightMatrix->RandomWeightInit();
    }
   
    
}

void ML_Lib::Layer::OverrideValMatrix(Matrix *InputValMatrix)
{
 
   for (int row = 0; row < ValMatrix->getRows(); row++) {
            for (int col = 0; col < ValMatrix->getCols(); col++) {
                ValMatrix->operator()(row, col) = InputValMatrix->operator()(row, col);
            }
            
    }
    
    
}

void ML_Lib::Layer::OverrideWeightMatrix(Matrix *NewWeights)
{
    Matrix OldWeights = *WeightMatrix;
    delete WeightMatrix;
    
    WeightMatrix = new Matrix(OldWeights.getRows(), OldWeights.getCols());
    for (int row = 0; row < WeightMatrix->getRows(); row++) {
             for (int col = 0; col < WeightMatrix->getCols(); col++) {
                 WeightMatrix->operator()(row, col) = OldWeights(row, col) + NewWeights->operator()(row, col);
             }
             
     }
}

void ML_Lib::Layer::feedforwardValues(ActivationFunction Ac)
{
    if(PreviousLayer != nullptr)
    {
        Matrix WeightM = PreviousLayer->GetWeightMatrix();
        Matrix ValM = PreviousLayer->GetValMatrix();

        Matrix NewValMatrix = WeightM * ValM;
        NewValMatrix.ActivateNeurons(Ac);
        
        OverrideValMatrix(&NewValMatrix);
        
    }
    
}



ML_Lib::Matrix ML_Lib::Layer::GetValMatrix()
{
    return *ValMatrix;
}


ML_Lib::Matrix ML_Lib::Layer::GetWeightMatrix()
{
    return *WeightMatrix;
}

ML_Lib::Matrix ML_Lib::Layer::GetSumMatrix()
{
    return (*WeightMatrix) * (*ValMatrix);
}
