//
//  LSTMNN.cpp
//  ML_Lib
//
//  Created by Thiviyan Saravanamuthu on 25.12.21.
//

#include "LSTMNN.hpp"


ML_Lib::LSTMNN::LSTMNN(std::vector<int> topology, int SequenceLength)
{
    this->topology = topology;
    this->SequenceLength = SequenceLength;
    
    std::vector<LSTMCell> BaseCells;
    
    for(int n = 0; n < topology.size() - 1; n++)
    {
       BaseCells.push_back(LSTMCell(topology[n],
                                    topology[n+1]));
    }
    
    for(int j = 0; j < SequenceLength; j++)
    {
        LSTMCells.push_back(BaseCells);
    }
    
    if(topology.size() > 1)
    {
    DenseLayer = Matrix(topology[topology.size() - 1], 1);
    DenseWeights = Matrix(topology[topology.size() - 1], topology[topology.size() - 2]);
    }
   
}

ML_Lib::LSTMNN::~LSTMNN()
{
   
    
}

void ML_Lib::LSTMNN::trainNetwork(std::vector<float> InputM, std::vector<float> ExpectedOutput)
{
    int index = 0;
    std::vector<Matrix> s_InputM;
    std::vector<float> Temp;
    
    for(int t = 0; t < 1; t++)
    {
        for(int j = 0; j < SequenceLength; j++)
        {
            for(int n = index; n < (index + topology[0]); n++)
            {
                Temp.push_back(InputM[n]);
            }
            
            s_InputM.push_back(Temp);
            
            if((index + 1 + topology[0]) > InputM.size())
            {
                index = 0;
            }
            else {
                index++;
            }
            Temp.clear();
         }
        
        feedforward(s_InputM);
        
    }
    
}


void ML_Lib::LSTMNN::feedforward(std::vector<Matrix> Input)
{
    for(int t = 0; t < SequenceLength; t++)
    {
        LSTMCells[t][0].SetInput(Input[t]);
        
        for(int n = 0; n < LSTMCells[t].size(); n++)
        {
            Matrix Input_t;
            if(n == 0)
            {
                Input_t = Input[t];
            }
            else
            {
                Input_t = LSTMCells[t][n - 1].getOutput();
            }
            
            if(t == 0 && n == 0)
            {
                Matrix EmptyM = Matrix(Input[t].getRows(), Input[t].getCols());
                LSTMCells[0][0].FeedForward(Input[t], EmptyM, EmptyM);
            }
            else if(t == 0 && n > 0)
            {
                Matrix EmptyM = Matrix(Input[t].getRows(), Input[t].getCols());
                Matrix NewInput = LSTMCells[t][n - 1].getOutput();
                
                LSTMCells[0][n].FeedForward(NewInput, EmptyM, EmptyM);
            }
            else if(t > 0 && n == 0)
            {
                Matrix PrevCellState = LSTMCells[t - 1][n].getCellState();
                Matrix PrevHiddenState = LSTMCells[t - 1][n].getHiddenState();
                
                LSTMCells[t][0].FeedForward(Input[t],
                                            PrevCellState,
                                            PrevHiddenState);
            }
            else if(t > 0 && n > 0)
            {
                Matrix PrevCellstate = LSTMCells[t - 1][n].getCellState();
                Matrix PrevHiddenState = LSTMCells[t - 1][n].getHiddenState();
                Matrix NewInput = LSTMCells[t][n - 1].getOutput();
                
                LSTMCells[t][n].FeedForward(NewInput,
                                            PrevCellstate,
                                            PrevHiddenState);
            }
                        
        }
        
        
    }
}

void ML_Lib::LSTMNN::backpropagate(ML_Lib::Matrix& Output, ML_Lib::Matrix& Targets)
{
    
    
}


ML_Lib::Matrix calcLoss(ML_Lib::Matrix& Output, ML_Lib::Matrix& Targets)
{
    ML_Lib::Matrix Loss = Output - Targets;
    Loss = Loss * Loss;
    Loss.MultiplyByScalar(0.5);
    
    return Loss;
}



ML_Lib::Matrix ML_Lib::LSTMNN::getOutput()
{
    return Output;
}

