//
//  FFNN.hpp
//  ML_Lib
//
//  Created by Thiviyan Saravanamuthu on 23.07.21.
//

#ifndef FFNN_hpp
#define FFNN_hpp

#include <iostream>
#include <vector>
#include "Layer.hpp"

namespace ML_Lib
{
    
    class FFNN
    {
    public:
        
        //MARK: Constructor/Destructor
        FFNN(std::vector<int> topology, ActivationFunction Ac, float LearningRate);
        ~FFNN();
        
        
        //MARK: Different kinds of excectution for the Network
        
        //Training, Test; Input is all the numbers(separated by layernum)
        void TrainNetwork(std::vector<float> Trainingsset, std::vector<float> Targets, size_t iterations);
        void TestNetwork(std::vector<float> Testingset, std::vector<float> Targets);
        
        //Just To run without any backpropagation
        std::vector<float> RunNetwork(std::vector<float> InputSet);
        
        //MARK: The fundamental algorithms to run the network
        void feedforward();
        void backpropagate(std::vector<float> CurrentTargets);
        
        float CalculateCost(std::vector<float> CurrentTargets);
        
        void PrintAll();
        
    private:
        
        std::vector<int> topology;
        int LayerNum;
        Layer* Layers;
        
        float Cost;
        int NumBatch;
        
        float LearningRate;
        ActivationFunction Ac;
        
    };


}



#endif /* FFNN_hpp */
