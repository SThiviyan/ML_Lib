//
//  LSTMCell.cpp
//  ML_Lib
//
//  Created by Thiviyan Saravanamuthu on 05.01.22.
//

#include "LSTMCell.hpp"


ML_Lib::LSTMCell::LSTMCell(int InputDim, int OutputDim)
{
    this->Input = Matrix(InputDim, 1);
    this->Output = Matrix(InputDim, 1);
    
    initWeightMatrices();
    
}

ML_Lib::LSTMCell::~LSTMCell()
{
   
}


void ML_Lib::LSTMCell::initWeightMatrices()
{
    
    //Constructor
    FWeights.push_back(Matrix(Output.getRows(), Input.getRows()));
    OWeights.push_back(Matrix(Output.getRows(), Input.getRows()));
    FWeights.push_back(Matrix(Input.getRows(), Input.getRows()));
    OWeights.push_back(Matrix(Input.getRows(), Input.getRows()));
  
    
    for(int n = 0; n < 2; n++)
    {
        CWeights.push_back(std::vector<Matrix>());
        
      
        CWeights[n].push_back(Matrix(Output.getRows(), Input.getRows()));
        CWeights[n].push_back(Matrix(Input.getRows(), Input.getRows()));
        
        //Random initalization
        CWeights[n][0].RandomWeightInit();
        CWeights[n][1].RandomWeightInit();
        FWeights[n].RandomWeightInit();
        OWeights[n].RandomWeightInit();
    }
    
}


void ML_Lib::LSTMCell::FeedForward(Matrix& Input, Matrix& PrevCellState, Matrix& PrevHiddenState)
{
    this->Input = Input;
     CurrentCellState = CalcCellState(CalcForgetGate(PrevHiddenState), CalcUpdateGate(PrevHiddenState), CalcNewCandidate(PrevHiddenState), PrevCellState);
       
    
     Output = CalcOutputGate(CurrentCellState, PrevHiddenState);
     CurrentHiddenState = Output;
    
}


void ML_Lib::LSTMCell::Backprop(Matrix& Targets)
{
    Matrix DerivativeErrorOutput = Output - Targets;
    Matrix DerivativeOutputHiddenState = Matrix(4, 4);
    
}

ML_Lib::Matrix ML_Lib::LSTMCell::CalcCellState(Matrix ForgetGate,
                                               Matrix UpdateGate,
                                               Matrix newCandidate,
                                               Matrix PastCellState)
{
    Matrix CellState = PastCellState;
    
    CellState = CellState * ForgetGate;
    
    Matrix Temp = UpdateGate * newCandidate;
    
    return CellState + Temp;
}


ML_Lib::Matrix ML_Lib::LSTMCell::CalcForgetGate(Matrix& PrevHiddenState)
{
    Matrix Temp1 = (FWeights[0]) * (Input);
    Matrix Temp2 =  FWeights[1] * PrevHiddenState;
    
    
    Matrix NewBatch = Temp1;
    if(Temp1 == Temp2)
    {
        NewBatch += Temp2;
    }
    
    NewBatch.ActivateNeurons(SIGMOID);
    
    return NewBatch;
}


ML_Lib::Matrix ML_Lib::LSTMCell::CalcUpdateGate(Matrix& PrevHiddenState)
{
    Matrix Temp1 = CWeights[0][0] * Input;
    Matrix Temp2 = CWeights[0][1] * PrevHiddenState;
    
    Matrix NewBatch = Temp1;
    if(Temp1 == Temp2) //Just checks if the dimensions are the same
    {
        NewBatch += Temp2;
    }
    
    NewBatch.ActivateNeurons(SIGMOID);
    
    return NewBatch;
}

ML_Lib::Matrix ML_Lib::LSTMCell::CalcNewCandidate(Matrix& PrevHiddenState)
{
    Matrix Temp1 = CWeights[1][0] * Input;
    Matrix Temp2 =  CWeights[1][1] * PrevHiddenState;
    
    Matrix NewBatch = Temp1;
    if(Temp1 == Temp2) //Just checks if the dimensions are the same
    {
        NewBatch += Temp2;
    }
    
    NewBatch.ActivateNeurons(TANH);
    
    return NewBatch;
}


ML_Lib::Matrix ML_Lib::LSTMCell::CalcOutputGate(Matrix CellState, Matrix& PrevHiddenState)
{
    Matrix Temp = CellState;
    Temp.ActivateNeurons(TANH);
        
    Matrix Temp1 = OWeights[0] * Input;
    Matrix Temp2 = OWeights[1] * PrevHiddenState;
    
    Matrix NewBatch = Temp1;
    if(Temp1 == Temp2) //Just checks if the dimensions are the same
    {
        NewBatch += Temp2;
    }
    
    NewBatch.ActivateNeurons(SIGMOID);
    
    return Temp * NewBatch;
}



ML_Lib::Matrix ML_Lib::LSTMCell::getCellState()
{
    return CurrentCellState;
}

ML_Lib::Matrix ML_Lib::LSTMCell::getHiddenState()
{
    return CurrentHiddenState;
}

ML_Lib::Matrix ML_Lib::LSTMCell::getOutput()
{
    return Output;
}


void ML_Lib::LSTMCell::SetInput(Matrix NewInput)
{
    Input = NewInput;
}


void ML_Lib::LSTMCell::PrintVals()
{
    
    
}
