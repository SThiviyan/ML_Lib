//
//  LSTMNN.hpp
//  ML_Lib
//
//  Created by Thiviyan Saravanamuthu on 25.12.21.
//

#ifndef LSTMNN_hpp
#define LSTMNN_hpp

#include <iostream>
#include "Matrix.hpp"
#include "LSTMCell.hpp"

namespace ML_Lib
{
    class LSTMNN
    {
    public:
        //MARK: Initalization
        LSTMNN(std::vector<int> topology, int SequenceLength);
        ~LSTMNN();
        
        //MARK: Usable Functions
        void trainNetwork(std::vector<float> InputM, std::vector<float> ExpectedOutput);
       
        //MARK: GET functions
        Matrix getOutput();
        
    private:
        //MARK: Operations
        void feedforward(std::vector<Matrix> Input);
        void backpropagate(Matrix& Output, Matrix& Targets);
        
        Matrix calcLoss(Matrix& Output, Matrix& ExpectedOutput);
        
        Matrix Input;
        Matrix Output;
        std::vector<std::vector<LSTMCell>> LSTMCells;
        
        Matrix DenseLayer;
        Matrix DenseWeights;
        
        std::vector<int> topology;
        int SequenceLength;
        
        
    };

}

#endif /* LSTMNN_hpp */
