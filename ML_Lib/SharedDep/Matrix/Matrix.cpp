//
//  Matrix.cpp
//  ML_Lib
//
//  Created by Thiviyan Saravanamuthu on 23.07.21.
//

#include "Matrix.hpp"
#include "ActivationFunctions.h"
#include <stdio.h>
#include <time.h>
#include <math.h>


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


ML_Lib::Matrix::Matrix(std::vector<float> Array)
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


ML_Lib::Matrix::Matrix(std::vector<std::vector<float>> Array)
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
                    Vals[row][col] = dtanh(Vals[row][col]);
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
    {
        return x;
    }
    else
    {
        return 0.f;
    }
}

float ML_Lib::Matrix::D_Relu(float x)
{
    if(x < 0.0f)
    {
        return 0.f;
    }
    else
    {
        return 1.f;
    }
}


float ML_Lib::Matrix::tanh(float x)
{
    return tanhf(x);
}

float ML_Lib::Matrix::dtanh(float x)
{
    return 1 - (x*x);
}





