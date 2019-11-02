#ifndef Matrix_H_INCLUDED
#define Matrix_H_INCLUDED

#include <iostream>
#include <stdio.h>



using namespace std;


class Matrix
{
    //public:
    int row,col;
    inline void destruct();
    inline void equality(const Matrix &mtx);
    public:
    double **data;
    Matrix(int row = 1,int col = 1);
    ~Matrix();
    Matrix(const Matrix& mtx);
    Matrix operator* (const Matrix& mtx);
    Matrix operator* (double d);
    Matrix operator= (const Matrix& mtx);
    void operator+= (const Matrix& mtx);
    void operator+= (double d);
    Matrix operator+ (const Matrix &mtx);
    friend Matrix hadamart_product(Matrix &mtx1, Matrix &mtx2);
    Matrix transpose();
    Matrix rot180();
    Matrix zero_padd(int top, int right, int bottom, int left);
    friend void convolution(Matrix &input, Matrix &kernel, Matrix &output, int stride = 1);
    friend void cross_correlation(Matrix &input, Matrix &kernel, Matrix &output, int stride = 1);
    friend void print_mtx_list(Matrix **mtx, int list_len);
    friend void print_mtx(Matrix &mtx);
    int get_row();
    int get_col();
    void zero();
};


#endif // Matrix_H_INCLUDED
