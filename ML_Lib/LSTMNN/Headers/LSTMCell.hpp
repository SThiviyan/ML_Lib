//
//  LSTMCell.hpp
//  ML_Lib
//
//  Created by Thiviyan Saravanamuthu on 05.01.22.
//

#ifndef LSTMCell_hpp
#define LSTMCell_hpp

#include <iostream>
#include "Matrix.hpp"

namespace ML_Lib {
        
    class LSTMCell
    {
    public:
        
        //MARK: Initialization
        LSTMCell(int InputDim, int OutputDim);
        ~LSTMCell();
        
        void initWeightMatrices();
        
        //MARK: Operations
        void FeedForward(Matrix& Input, Matrix& PrevCellState, Matrix& PrevHiddenState); // Passing Input, Previous Cell State(timestep t-1) and Previous Hidden State(timestep t-1)
        void Backprop(Matrix& Targets);
        
        //Calculations
        Matrix CalcForgetGate(Matrix& PrevHiddenState);
        Matrix CalcNewCandidate(Matrix& PrevHiddenState);
        Matrix CalcOutputGate(Matrix CellState, Matrix& PrevHiddenState);
        Matrix CalcUpdateGate(Matrix& PrevHiddenState);
        
        Matrix CalcCellState(Matrix ForgetGate, Matrix UpdateGate, Matrix newCandidate, Matrix PastCellState);
        
        
        //GetFunctions
        Matrix getCellState();
        Matrix getHiddenState();
        Matrix getOutput();
        
        
        void SetInput(Matrix NewInput); // Only for Overriding Input
        
        void PrintVals();
        
        
    private:
        Matrix CurrentCellState;
        Matrix CurrentHiddenState;
        Matrix Output;
        Matrix Input;
        
        
        std::vector<Matrix> FWeights; // Forget Gate Weights
        std::vector<std::vector<Matrix>> CWeights; // Cell State Weights
        std::vector<Matrix> OWeights; // Output Gate Weights
        
    };

    
}


#endif /* LSTMCell_hpp */
