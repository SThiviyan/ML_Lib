//
//  main.cpp
//  ML_Lib_UseCases
//
//  Created by Thiviyan Saravanamuthu on 06.10.21.
//

#include <iostream>
#include "ML_Lib.hpp"
#include "ImageReader.hpp"

int main(int argc, const char * argv[]) {
    // insert code here...
  
   
    
    std::vector<int> topology = {10, 1};
    ML_Lib::LSTMNN LSTMNN = ML_Lib::LSTMNN(topology, 3);
    
  
    
    std::vector<float> Input = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
    std::vector<float> Output = {0.23};
    
    

    LSTMNN.trainNetwork(Input, Output);
    

    return 0;
}


/*
 
 std::vector<int> topology = {2, 3, 1};
 

 ML_Lib::FFNN NN(topology, ML_Lib::SIGMOID, 0.1f);
 
 
 std::vector<float> TrainingSet = {1,0,0,1,1,1,0,0};
 std::vector<float> Targets = {1,1,0,0};
 
 //NN.TrainNetwork(TrainingSet, Targets, 100);
 
 
 std::vector<std::vector<float>> Image;
 std::vector<float> Labels;
 
 ML_Lib::ImageReader::return_mnist_dataset("/Users/thiviyansaravanamuthu/Nextcloud/Programming/Development/Cpp-Projects/ML/ML_Lib/ML_Lib_UseCases/train-images-idx3-ubyte", "/Users/thiviyansaravanamuthu/Nextcloud/Programming/Development/Cpp-Projects/ML/ML_Lib/ML_Lib_UseCases/train-labels-idx1-ubyte", Image, Labels, false);
 
 */
