//
//  Matrix.cpp
//  ML_Lib
//
//  Created by Thiviyan Saravanamuthu on 23.07.21.
//

#include "Matrix.hpp"
#include "ActivationFunctions.hpp"
#include <stdio.h>
#include <time.h>
#include <cmath>


//MARK: Constructor and Destructor

ML_Lib::Matrix::Matrix(int rows, int cols)
{
    this->cols = cols;
    this->rows = rows;
    
    
    //Allocating Memory for Columns

    for(int n = 0; n < rows; n++)
    {
        Vals.push_back(std::vector<float>());
        for (int j = 0; j < cols; j++) {
            Vals[n].push_back(0);
        }
    }
    
    
}


ML_Lib::Matrix::Matrix(std::vector<float> &Array)
{
    Vals.clear();
    
    for(int n = 0; n < Array.size(); n++)
    {
        Vals.push_back(std::vector<float>());
        Vals[n].push_back(Array[n]);
    }
    
    this->cols = 1;
    this->rows = int(Array.size());
}


ML_Lib::Matrix::Matrix(std::vector<std::vector<float>> &Array)
{
    Vals.clear();
    
    for(int n = 0; n < Array.size(); n++)
    {
        Vals.push_back(std::vector<float>());
        for (int j = 0; j < Array[0].size(); j++) {
            Vals[n].push_back(Array[n][j]);
        }
    }
    
    this->cols = int(Array[0].size());
    this->rows = int(Array.size());
}

ML_Lib::Matrix::Matrix()
{
    this->cols = 0;
    this->rows = 0;
}


ML_Lib::Matrix::~Matrix()
{
    
}


//MARK: Mathematical Stuff

void ML_Lib::Matrix::MultiplyByScalar(float Scalar)
{
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            Vals[row][col] = Vals[row][col] * Scalar;
        }
    }
    
}

void ML_Lib::Matrix::DivideByScalar(float Scalar)
{
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            Vals[row][col] = Vals[row][col] / Scalar;
        }
    }
    
}


float GetRandomNum()
{
    float x = float(rand() % 100 + 1) / 100;
    
  
    return x;
}

void ML_Lib::Matrix::RandomWeightInit()
{
    srand(int(time(NULL)));
   
    /*
    std::vector<float> RandomWeightSequence;
   
    for (int n = 0; n < rows * cols; n++) {
        RandomWeightSequence.push_back(GetRandomNum());
    }
    
    for (int n = 0; n < RandomWeightSequence.size(); n++) {
        for(int j = n + 1; j < RandomWeightSequence.size(); j++)
        {
            bool SameNumber = RandomWeightSequence[n] == RandomWeightSequence[j];
            while(SameNumber)
            {
                RandomWeightSequence[n] = GetRandomNum();
                
                if(RandomWeightSequence[n] != RandomWeightSequence[j])
                {
                    SameNumber = false;
                }
                
            }
        }
    }
     */
    
    for(int n = 0; n < rows; n++)
    {
        for (int j = 0; j < cols; j++) {
            
            Vals[n][j] = GetRandomNum();
            
        }
    }
   
}


void ML_Lib::Matrix::RandonWeightInitwithRange(int start, int end)
{
    srand(int(time(NULL)));
    
    for (int n = 0; n < rows; n++) {
        for (int j = 0; j < cols; j++) {
           if(start < 0)
           {
            Vals[n][j] = float(rand() % end) - start;
           }
        }
    }
}



ML_Lib::Matrix ML_Lib::Matrix::GetTransposedMatrix()
{
    Matrix TransposedMatrix = Matrix(cols, rows);
    
    for (int n = 0; n < TransposedMatrix.rows; n++) {
        for (int j = 0; j < TransposedMatrix.cols; j++) {
            TransposedMatrix(n, j) = Vals[j][n];
        }
    }
    
    return TransposedMatrix;
}


//MARK: ActivationFunction Stuff (Seperate from Math because...)

void ML_Lib::Matrix::ActivateNeurons(ActivationFunction AF)
{
    switch (AF) {
        
        case SIGMOID:
           
            for(int row = 0; row < rows; row++)
            {
                for (int col = 0; col < cols; col++) {
                    Vals[row][col] = Sigmoid(Vals[row][col]);
                }
            }
            
            break;
            
        case D_SIGMOID:
            
            for(int row = 0; row < rows; row++)
            {
                for (int col = 0; col < cols; col++) {
                    Vals[row][col] = D_Sigmoid(Vals[row][col]);
                }
            }
            
            break;
            
        case RELU:
            
            for(int row = 0; row < rows; row++)
            {
                for (int col = 0; col < cols; col++) {
                    Vals[row][col] = Relu(Vals[row][col]);
                }
            }
            
            break;
            
        case D_RELU:
            
            for(int row = 0; row < rows; row++)
            {
                for (int col = 0; col < cols; col++) {
                    Vals[row][col] = D_Relu(Vals[row][col]);
                }
            }
            
            break;
      
        case TANH:
            
            for(int row = 0; row < rows; row++)
            {
                for (int col = 0; col < cols; col++) {
                    Vals[row][col] = tanh(Vals[row][col]);
                }
            }
            
            break;
            
        case D_TANH:
            
            for(int row = 0; row < rows; row++)
            {
                for (int col = 0; col < cols; col++) {
                    Vals[row][col] = dtanhf(Vals[row][col]);
                }
            }
            
            break;
        default:
            break;
    }
}


void ML_Lib::Matrix::TakeDerivative(ActivationFunction AF)
{
    switch (AF) {
        case RELU:
            ActivateNeurons(D_RELU);
            break;
        case SIGMOID:
            ActivateNeurons(D_SIGMOID);
            break;
        case TANH:
            ActivateNeurons(D_TANH);
            break;
        default:
            std::cout << "Derivative not taken. Supplied Activationfunction is non existant." << std::endl;
            break;
    }
}

//2 Activation functions and their derivatives

float ML_Lib::Matrix::Sigmoid(float x)
{
    return 1 / (1 + exp(-x));
}

float ML_Lib::Matrix::D_Sigmoid(float x)
{
    return x * (1 - x);
}

float ML_Lib::Matrix::Relu(float x)
{
    if(x > 0.f)
        return x;
    return 0.01 * x;
}

float ML_Lib::Matrix::D_Relu(float x)
{
    if(x > 0.f)
        return 1.f;
    return 0.5f;
}


float ML_Lib::Matrix::tanhf(float x)
{
    return tanh(x);
}

float ML_Lib::Matrix::dtanhf(float x)
{
    return 1 - (x*x);
}




void ML_Lib::Matrix::PrintMatrix()
{
    for (int n = 0; n < rows; n++) {
        for (int j = 0; j < cols; j++) {
            std::cout << Vals[n][j] << " ";
        }
        std::cout << std::endl;
    }
}


std::vector<std::vector<float>> ML_Lib::Matrix::ReturnTwoDimensionalVector(std::vector<float> &OneDimVector,int rows, int cols)
{
    std::vector<std::vector<float>> TwoDimVector;
    
    if(OneDimVector.size() == rows*cols)
    {
        std::vector<float> Temp;
        for (int l = 1; l <= OneDimVector.size(); l++) {
            Temp.push_back(OneDimVector[l - 1]);
            
            if(l % rows == 0)
            {
                TwoDimVector.push_back(Temp);
                Temp.clear();
            }
            
        }
        
    }
    
    return TwoDimVector;
}



bool ML_Lib::Matrix::isEmpty()
{
    for(int n = 0; n < rows; n++)
    {
        for(int j = 0; j < cols; j++)
        {
            if(Vals[n][j] != 0)
            {
                return false;
            }
        }
    }
    
    return true;
}
