#ifndef Matrice_H_INCLUDED
#define Matrice_H_INCLUDED

#include <iostream>
#include <stdio.h>



using namespace std;


class Matrice
{
    int row,col;
    inline void destruct();
    inline void equality(const Matrice &mtx);
    public:
    double **data;
    Matrice(int row = 1,int col = 1);
    ~Matrice();
    Matrice(const Matrice& mtx);
    Matrice operator* (const Matrice& mtx);
    Matrice operator= (const Matrice& mtx);
    friend Matrice hadamart_product(Matrice &mtx1, Matrice &mtx2);
    Matrice transpose();
    friend void print_mtx_list(Matrice **mtx, int list_len);
};

#endif // Matrice_H_INCLUDED
