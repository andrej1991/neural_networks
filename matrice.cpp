#include "matrice.h"
Matrice::Matrice(int row,int col) : data(NULL)
{
    if(row > 0)
        this->row = row;
    else
        this->row = 1;
    if(col>0)
        this->col = col;
    else
        this->col = 1;
    try
        {
            data = new double* [row];
            for(int i = 0; i < row; i++)
                {
                    data[i] = new double [col];
                    for(int j = 0; j < col; j++)
                        {
                            data[i][j] = 0;
                        }
                }
        }
    catch(bad_alloc& ba)
        {
            cerr << "Matrice::constructor: bad_alloc caught: " << ba.what() << endl;
            throw;
        }
}

inline void Matrice::destruct()
{
    //cout << this->data << endl;
    if(this->data != NULL)
        {
            for(int i = 0; i < this->row; i++)
                {
                    delete this->data[i];
                }
            delete[] this->data;
            this->data = NULL;
        }
}

inline void Matrice::equality(const Matrice &mtx)
{
    row = mtx.row;
    col = mtx.col;
    try
       {
            data = new double*[row];
            for(int i = 0; i < row; i++)
                data[i] = new double [col];
       }
    catch(bad_alloc& ba)
        {
            cerr << "Matrice::copyconstructor: bad_alloc caught: " << ba.what() << endl;
            throw;
        }
    for(int i = 0; i < row; i++)
    {
        for(int j = 0; j < col; j++)
            data[i][j] = mtx.data[i][j];
    }
}

Matrice::~Matrice()
    {
        this->destruct();
    }

Matrice::Matrice (const Matrice& mtx)
{
    //this->destruct();
    this->equality(mtx);

}
Matrice Matrice::operator= (const Matrice& mtx)
{
    if(this==&mtx) return *this;
    this->destruct();
    this->equality(mtx);
    return *this;

}

Matrice Matrice::operator* (const Matrice& other)
{
    if(col != other.row)
        {
            std::cerr << "the condition of the if statement is fales in the operator Matrice::operator*\n";
            throw std::exception();
        }
    else
        {
            Matrice mtx(row, other.col);
            double c = 0;
            for(int k = 0; k < row; k++)
                {
                    for(int l = 0; l < other.col; l++)
                        {
                            for(int i = 0; i < col; i++)
                                {
                                    c += data[k][i] * other.data[i][l];
                                }
                            mtx.data[k][l] = c;
                            c = 0;
                        }
                }
            return mtx;
        }
}

void Matrice::operator+=(const Matrice& mtx)
{
    if((this->col != mtx.col) + (this->row != mtx.row))
        {
            std::cerr << "the matrices cannot be added!\n";
            throw std::exception();
        }
    for(int i = 0; i < this->row; i++)
        {
            for(int j = 0; j < this->col; j++)
                {
                    this->data[i][j] += mtx.data[i][j];
                }
        }
}

void Matrice::operator+=(double d)
{
    for(int i = 0; i < this->row; i++)
        {
            for(int j = 0; j < this->col; j++)
                {
                    this->data[i][j] += d;
                }
        }
}

Matrice Matrice::operator+(const Matrice &mtx)
{
    if((this->col != mtx.col) + (this->row != mtx.row))
        {
            std::cerr << "the matrices cannot be added!\n";
            throw std::exception();
        }
    Matrice sum(this->row, this->col);
    for(int i = 0; i < this->row; i++)
        {
            for(int j = 0; j < this->col; j++)
                {
                    sum.data[i][j] = this->data[i][j] + mtx.data[i][j];
                }
        }
    return sum;
}

/*Matrice Matrice::operator-(const Matrice& mtx)
{
    if((this->col != mtx.col) + (this->row != mtx.row))
        {
            std::cerr << "the matrices cannot be substracted!\n";
            throw std::exception();
        }
    Matrice difference(this->row, this->col);
    for(int i = 0; i < this->row; i++)
        {
            for(int j = 0; j < this->col; j++)
                {
                    difference.data[i][j] = this->data[i][j] - mtx.data[i][j];
                }
        }
    return dufference;
}

Matrice Matrice::operator-(double** mtx)
{
    Matrice difference(this->row, this->col);
    for(int i = 0; i < this->row; i++)
        {
            for(int j = 0; j < this->col; j++)
                {
                    difference.data[i][j] = this->data[i][j] - mtx[i][j];
                }
        }
}*/

int Matrice::get_row()
{
    return this->row;
}

int Matrice::get_col()
{
    return this->col;
}

Matrice hadamart_product(Matrice &mtx1, Matrice &mtx2)
{
    if((mtx1.row == mtx2.row) * (mtx1.col == mtx2.col))
        {
            Matrice result(mtx1.row, mtx1.col);
            for(int i = 0; i < mtx1.row; i++)
                for(int j = 0; j < mtx1.col; j++)
                    result.data[i][j] = mtx1.data[i][j] * mtx2.data[i][j];
            return result;
        }
    else
        {
            std::cerr << "the condition of the if statement is fales in the function Matrice::hadamart_product\n";
            throw std::exception();
        }
}

Matrice Matrice::transpose()
{
    Matrice tr_mtx(this->col,this->row);
    for(int i = 0; i < this->row; i++)
        {
            for(int  j= 0; j < this->col; j++)
                {
                    tr_mtx.data[j][i]=this->data[i][j];
                }
        }
    return tr_mtx;
}

Matrice Matrice::rot180()
{
    Matrice ret(this->row, this->col);
    int i2 = 0;
    int j2 = 0;
    for(int i = this->row - 1; i >= 0; i--)
        {
            for(int j = this->col - 1; j >= 0; j--)
                {
                    ret.data[i][j] = this->data[i2][j2];
                    j2++;
                }
            i2++;
            j2 = 0;
        }
    return ret;
}

Matrice Matrice::zero_padd(int top, int right, int bottom, int left)
{
    Matrice ret(this->row + top + bottom, this->col + left + right);
    for(int i = 0; i < this->row; i++)
        {
            for(int j = 0; j < this->col; j++)
                {
                    ret.data[top + i][left + j] = this->data[i][j];
                }
        }
    return ret;
}

void convolution(Matrice &input, Matrice &kernel, Matrice &output, int stride)
{
    double helper;
    int r, c;
    r = c = 0;
    int vertical_step = (input.row - kernel.row) / stride + 1;
    int horizontal_step = (input.col - kernel.col) / stride + 1;
    for(int i = 0; i < vertical_step; i += stride)
        {
            for(int j = 0; j < horizontal_step; j += stride)
                {
                    helper = 0;
                    for(int k = 0; k < kernel.row; k++)
                        {
                            for(int l = 0; l < kernel.col; l++)
                                {
                                    helper += kernel.data[k][l] * input.data[i + k][j + l];
                                }
                        }
                    output.data[r][c] = helper;
                    c++;
                }
            r++;
            c = 0;
        }
}

void Matrice::zero()
{
    for(int i = 0; i < this->row; i++)
        for(int j = 0; j < this->col; j++)
            this->data[i][j] = 0;
}

void print_mtx_list(Matrice **mtx, int list_len)
{
    for(int i = 0; i < list_len; i++)
    {
        cout << "[";
        for(int j = 0; j < mtx[i][0].row; j++)
        {
            cout << "[";
            for(int k = 0; k < mtx[i][0].col; k++)
            {
                cout << mtx[i][0].data[j][k] << "; ";
            }
            cout << "]\n";
        }
        cout << "]\n";
    }

}

void print_mtx(Matrice &mtx)
{
    cout << "[";
    for(int j = 0; j < mtx.row; j++)
    {
        cout << "[";
        for(int k = 0; k < mtx.col; k++)
        {
            cout << mtx.data[j][k] << "; ";
        }
        cout << "]\n";
    }
    cout << "]\n";
}
