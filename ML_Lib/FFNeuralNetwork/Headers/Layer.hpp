//
//  Layer.hpp
//  ML_Lib
//
//  Created by Thiviyan Saravanamuthu on 23.07.21.
//

#ifndef Layer_hpp
#define Layer_hpp

#include <iostream>
#include "Matrix.hpp"

namespace ML_Lib
{

    //MARK: Layer Class-> Contains Neurons -> Each have weights
    class Layer
    {
    public:
        Layer(int NeuronNum, int topologyNextElement, Layer* PreviousLayer);
        
        //Feedforwards layer
        void feedforwardValues(ActivationFunction Ac);
            
        //Override Val Matrix -> Only for Input and feedforward
        void OverrideValMatrix(Matrix* InputValMatrix);
        
        //Override Weight Matrix -> After Backprop
        void OverrideWeightMatrix(Matrix* NewWeights);
        
        //Get Val or Weight Matrix
        Matrix GetValMatrix();
        Matrix GetWeightMatrix();
        Matrix GetSumMatrix();
        
        
    private:
        
        //MARK: Properties and Matrices necessary to calculate
        Matrix* ValMatrix;
        Matrix* WeightMatrix;
        Layer* PreviousLayer;
        
        
        Matrix* ErrorValMatrix;
        
        int NeuronNum;
        int topologyNext;
    };

}


#endif /* Layer_hpp */
