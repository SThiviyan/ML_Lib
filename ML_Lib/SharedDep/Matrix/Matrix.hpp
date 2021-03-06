//
//  Matrix.hpp
//  ML_Lib
//
//  Created by Thiviyan Saravanamuthu on 23.07.21.
//

#ifndef Matrix_hpp
#define Matrix_hpp

#include <iostream>
#include <vector>
#include "ActivationFunctions.hpp"

namespace ML_Lib
{
    class Matrix
    {
    public:
    
        //MARK: Constructor and Destructor
        Matrix(int rows, int cols);
        Matrix(std::vector<float> &Array);
        Matrix(std::vector<std::vector<float>> &Array);
        Matrix();
        ~Matrix();
        
        static std::vector<std::vector<float>> ReturnTwoDimensionalVector(std::vector<float> &OneDimVector,int rows, int cols);
 
        //MARK: Acces to Elements / Mathematical stuff
    
        //Operators
        float &operator () (int m, int n) {
    
            if(m >= rows || n >= cols)
            {
                std::cout << "Index Out of Bounds!!" << std::endl;
                std::cout << "Requested Index:" << rows << " and " << cols << std::endl;
                return Vals[0][0];
            }
            else
            {
                return Vals[m][n];
            }
    
        }
        
        bool operator == (Matrix M) //MARK: ONLY COMPARES DIMENSIONS NOT VALUES!!!
        {
            if(rows == M.getRows() && cols == M.getCols())
            {
                return true;
            }
            return false;
        }
    
        void operator = (Matrix M){
    
            Vals.clear();
    
            for(int n = 0; n < M.getRows(); n++)
            {
                Vals.push_back(std::vector<float>());
                for(int j = 0; j < M.getCols(); j++)
                {
                        //Vals[n].push_back(float());
                    Vals[n].push_back(M(n, j));
                }
            }
    
            this->rows = M.getRows();
            this->cols = M.getCols();
    
        }
    
        void operator = (const std::vector<float> M)
        {
            if(this->rows == M.size())
            {
                for(int n = 0; n < this->rows; n++)
                {
                  Vals[n][0] = M[n];
                }
            }
        }
    
        Matrix operator * ( Matrix& SecondMatrix)
        {
            Matrix MultipliedMatrix(this->rows, SecondMatrix.cols);
    
            if(this->cols == SecondMatrix.cols && this->rows == SecondMatrix.rows)
            {
                for(int row = 0; row < this->rows; row++)
                {
                    for(int col = 0; col < this->cols; col++)
                    {
                        MultipliedMatrix(row, col) = Vals[row][col] * SecondMatrix(row, col);
                    }
                }
            }
            else if(this->cols == 1 && SecondMatrix.rows == 1 && SecondMatrix.cols == 1)
            {
                for(int row = 0; row < this->rows; row++)
                {
                    for(int col = 0; col < 1; col++)
                    {
                            MultipliedMatrix(row, col) = Vals[row][col] * SecondMatrix(0, 0);
                    }
                }
    
            }
            else if(this->cols == SecondMatrix.rows)
            {
    
                for(int RowMatrixOne = 0; RowMatrixOne < this->rows; RowMatrixOne++)
                {
                    for(int SharedDimension = 0; SharedDimension < this->cols; SharedDimension++)
                    {
    
                        for(int ColMatrixTwo = 0; ColMatrixTwo < SecondMatrix.getCols(); ColMatrixTwo++)
                        {
                            MultipliedMatrix(RowMatrixOne, ColMatrixTwo) += Vals[RowMatrixOne][SharedDimension] *   SecondMatrix(SharedDimension, ColMatrixTwo);
                        }
                    }
                }
            }
    
    
    
            return MultipliedMatrix;
        }
    
        Matrix operator + ( Matrix& SecondMatrix)
        {
            Matrix NewMatrix = Matrix(this->rows, this->cols);
    
            if(this->rows == SecondMatrix.rows && this->cols == SecondMatrix.cols)
            {
                for (int n = 0; n < rows; n++) {
                    for (int j = 0; j < cols; j++) {
                        NewMatrix(n, j) = Vals[n][j] + SecondMatrix(n, j);
                    }
                }
    
            }
            else
            {
                std::cout << "Can't Add them together " << rows << "x" << cols << " and " << SecondMatrix.rows << "x" << SecondMatrix.cols << std::endl;
            }
    
    
            return NewMatrix;
        }
    
        Matrix operator += ( Matrix& SecondMatrix)
        {
            Matrix NewMatrix = Matrix(this->rows, this->cols);
    
            if(this->rows == SecondMatrix.rows && this->cols == SecondMatrix.cols)
            {
                for (int n = 0; n < rows; n++) {
                    for (int j = 0; j < cols; j++) {
                        NewMatrix(n, j) = Vals[n][j] + SecondMatrix(n, j);
                    }
                }
    
            }
            else
            {
                std::cout << "Can't Add them together" << std::endl;
            }
    
    
            return NewMatrix;
        }
    
        Matrix operator - (Matrix& SecondMatrix)
        {
            Matrix NewMatrix = Matrix(this->rows, this->cols);
    
            if(this->rows == SecondMatrix.rows && this->cols == SecondMatrix.cols)
            {
                for (int n = 0; n < rows; n++) {
                    for (int j = 0; j < cols; j++) {
                        NewMatrix(n, j) = Vals[n][j] - SecondMatrix(n, j);
                    }
                }
    
            }
            else
            {
                std::cout << "Can't Subtract them" << std::endl;
            }
    
    
            return NewMatrix;
        }
    
        //Scalar Multiplication
        void MultiplyByScalar(float Scalar);
        void DivideByScalar(float Scalar);
    
    
        //Return Transposed Matrix; ex. 3 x 2 -> 2 x 3
        Matrix GetTransposedMatrix();
    
    
        //Random Weight Initalization
        void RandomWeightInit();
        void RandonWeightInitwithRange(int start, int end);
    
    
        //Activations
        void ActivateNeurons(ActivationFunction AF);
        void TakeDerivative(ActivationFunction AF);
        float Sigmoid(float x);
        float D_Sigmoid(float x);
        float Relu(float x);
        float D_Relu(float x);
        float tanhf(float x);
        float dtanhf(float x);
    
    
        //MARK: GET functions
    
        int getCols(){return this->cols;};
        int getRows(){return this->rows;};
        
        
        //Returns true if all values are 0
        bool isEmpty();
    
        
        void PrintMatrix();
    
    private:
    
        //MARK: Matrix Properties
        std::vector<std::vector<float>> Vals;
        int rows;
        int cols;
    
    };
    






}



#endif /* Matrix_hpp */
