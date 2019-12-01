#include <math.h>
#include "matrix.h"

Matrix::Matrix(int r,int c) : data(NULL)
{
    if(r > 0)
        this->row = r;
    else
        {
            cerr << "The matrix must contain more than 0 rows!\n";
            throw exception();
        }
    if(c >= 0)
        this->col = c;
    else
        {
            cerr << "The matrix must contain more than 0 colums!\n";
            throw exception();
        }
    try
        {
            this->data = new double* [r];
            for(int i = 0; i < r; i++)
                {
                    this->data[i] = new double [c];
                    for(int j = 0; j < c; j++)
                        {
                            this->data[i][j] = 0;
                        }
                }
        }
    catch(bad_alloc& ba)
        {
            cerr << "Matrix::constructor: bad_alloc caught: " << ba.what() << endl;
            throw;
        }
}

inline void Matrix::destruct()
{
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

inline void Matrix::equality(const Matrix &mtx)
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
            cerr << "Matrix::copyconstructor: bad_alloc caught: " << ba.what() << endl;
            throw;
        }
    for(int i = 0; i < row; i++)
    {
        for(int j = 0; j < col; j++)
            data[i][j] = mtx.data[i][j];
    }
}

Matrix::~Matrix()
    {
        this->destruct();
    }

Matrix::Matrix (const Matrix& mtx)
{
    //this->destruct();
    this->equality(mtx);

}
Matrix & Matrix::operator= (const Matrix& mtx)
{
    this->destruct();
    this->equality(mtx);
    return *this;

}

Matrix Matrix::operator* (const Matrix& other)
{
    int debug1 = col;
    if(col != other.row)
        {
            std::cerr << "the condition of the if statement is fales in the operator Matrix::operator*\n";
            throw std::exception();
        }
    else
        {
            Matrix mtx(row, other.col);
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

Matrix Matrix::operator/(const Matrix &mtx)
{
    if((this->col != mtx.col) && (this->row != mtx.row))
    {
        cerr << "The row or the colum is not equal of the matrixes in the division operator\n";
        throw exception();
    }
    Matrix ret(this->row, this->col);
    for(int i = 0; i < this->row; i++)
    {
        for(int j = 0; j < this->col; j++)
        {
            ret.data[i][j] = this->data[i][j] / mtx.data[i][j];
        }
    }

    return ret;
}

Matrix Matrix::operator*(double d)
{
    Matrix mtx(this->row, this->col);
    for(int i = 0; i < this->row; i++)
    {
        for(int j = 0; j < this->col; j++)
        {
            mtx.data[i][j] = this->data[i][j] * d;
        }
    }
    return mtx;
}

void Matrix::operator+=(const Matrix& mtx)
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

void Matrix::operator+=(double d)
{
    for(int i = 0; i < this->row; i++)
        {
            for(int j = 0; j < this->col; j++)
                {
                    this->data[i][j] += d;
                }
        }
}

Matrix Matrix::operator+(double d)
{
    Matrix sum(this->row, this->col);
    for(int i = 0; i < this->row; i++)
    {
        for(int j = 0; j < this->col; j++)
        {
            sum.data[i][j] = this->data[i][j] + d;
        }
    }
    return sum;
}

Matrix Matrix::operator+(const Matrix &mtx)
{
    if((this->col != mtx.col) + (this->row != mtx.row))
        {
            std::cerr << "the matrices cannot be added!\n";
            throw std::exception();
        }
    Matrix sum(this->row, this->col);
    for(int i = 0; i < this->row; i++)
        {
            for(int j = 0; j < this->col; j++)
                {
                    sum.data[i][j] = this->data[i][j] + mtx.data[i][j];
                }
        }
    return sum;
}

int Matrix::get_row()
{
    return this->row;
}

int Matrix::get_col()
{
    return this->col;
}

Matrix hadamart_product(Matrix &mtx1, Matrix &mtx2)
{
    if((mtx1.row == mtx2.row) * (mtx1.col == mtx2.col))
        {
            Matrix result(mtx1.row, mtx1.col);
            for(int i = 0; i < mtx1.row; i++)
                for(int j = 0; j < mtx1.col; j++)
                    result.data[i][j] = mtx1.data[i][j] * mtx2.data[i][j];
            return result;
        }
    else
        {
            std::cerr << "the condition of the if statement is fales in the function Matrix::hadamart_product\n";
            throw std::exception();
        }
}

Matrix Matrix::transpose()
{
    Matrix tr_mtx(this->col,this->row);
    for(int i = 0; i < this->row; i++)
        {
            for(int  j= 0; j < this->col; j++)
                {
                    tr_mtx.data[j][i]=this->data[i][j];
                }
        }
    return tr_mtx;
}

Matrix Matrix::rot180()
{
    Matrix ret(this->row, this->col);
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

Matrix Matrix::sqroot()
{
    Matrix ret(this->row, this->col);
    for(int i = 0; i < this->row; i++)
    {
        for(int j = 0; j < this->col; j++)
        {
            ret.data[i][j] = sqrt(this->data[i][j]);
        }
    }
    return ret;
}

Matrix Matrix::square_element_by()
{
    Matrix ret(this->row, this->col);
    for(int i = 0; i < this->row; i++)
    {
        for(int j = 0; j < this->col; j++)
        {
            ret.data[i][j] = this->data[i][j] * this->data[i][j];
        }
    }
    return ret;
}

Matrix Matrix::zero_padd(int top, int right, int bottom, int left)
{
    int padded_row = this->row + top + bottom;
    int padded_col = this->col + left + right;
    Matrix ret(padded_row, padded_col);
    for(int i = 0; i < this->row; i++)
        {
            for(int j = 0; j < this->col; j++)
                {
                    ret.data[top + i][left + j] = this->data[i][j];
                }
        }
    return ret;
}

void cross_correlation(Matrix &input, Matrix &kernel, Matrix &output,  int vertical_stride, int horizontal_stride)
{
    double helper;
    int r, c;
    r = c = 0;
    int vertical_step = (input.row - kernel.row) / vertical_stride + 1;
    int horizontal_step = (input.col - kernel.col) / horizontal_stride + 1;
    for(int i = 0; i < vertical_step; i += vertical_stride)
        {
            for(int j = 0; j < horizontal_step; j += horizontal_stride)
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

void convolution(Matrix &input, Matrix &kernel, Matrix &output, int vertical_stride, int horizontal_stride)
{
    double helper;
    int r, c;
    r = c = 0;
    for(int i = kernel.row-1; i < input.row; i += vertical_stride)
        {
            for(int j = kernel.col-1; j < input.col; j += horizontal_stride)
                {
                    helper = 0;
                    for(int k = kernel.row-1; k >= 0; k--)
                        {
                            for(int l = kernel.col-1; l >= 0; l--)
                                {
                                    helper += kernel.data[k][l] * input.data[i - k][j - l];
                                }
                        }
                    output.data[r][c] = helper;
                    c++;
                }
            r++;
            c = 0;
        }
}

Matrix Matrix::dilate(int vertical_stride, int horizontal_stride)
{
    int new_row, new_col;
    new_row = this->row + ((this->row - 1) * (vertical_stride - 1));
    new_col = this->col + ((this->col - 1) * (horizontal_stride - 1));
    horizontal_stride;
    vertical_stride;
    Matrix ret(new_row, new_col);
    ret.zero();
    int r, c;
    r = 0;
    for(int i = 0; i < this->row; i++)
    {
        c = 0;
        for(int j = 0; j < this->col; j++)
        {
            ret.data[r][c] = this->data[i][j];
            c += horizontal_stride;
        }
        r += vertical_stride;
    }
    return ret;
}

void Matrix::zero()
{
    for(int i = 0; i < this->row; i++)
        for(int j = 0; j < this->col; j++)
            this->data[i][j] = 0;
}

double Matrix::squared_sum_over_elements()
{
    double squared_sum = 0;
    for(int i = 0; i < this->row; i++)
    {
        for(int j = 0; j < this->col; j++)
        {
            squared_sum += this->data[i][j]*this->data[i][j];
        }
    }
    return squared_sum;
}

double Matrix::sum_over_elements()
{
    double sum = 0;
    for(int i = 0; i < this->row; i++)
    {
        for(int j = 0; j < this->col; j++)
        {
            sum += this->data[i][j];
        }
    }
    return sum;
}

Matrix* Matrix::remove_rows(Matrix &rows_to_remove)
{
    if(rows_to_remove.row != this->row)
    {
        cerr << "Not decided about each row if it needs to be deleted or not\n";
        throw exception();
    }
    int count_of_remaining = 0;
    for(int i = 0; i < rows_to_remove.row; i++)
    {
        if(rows_to_remove.data[i][0] != 1)
            count_of_remaining++;
    }
    Matrix *ret = new Matrix(count_of_remaining, this->col);
    int k = 0;
    for(int i = 0; i < this->row; i++)
    {
        if(rows_to_remove.data[i][0] != 1)
        {
            for(int j = 0; j < this->col; j++)
            {
                ret->data[k][j] = this->data[i][j];
            }
            k++;
        }
    }
    return ret;
}

Matrix* Matrix::remove_colums(Matrix &colums_to_remove)
{
    if(colums_to_remove.row != this->col)
    {
        cerr << "Not decided about each colum if it needs to be deleted or not\n";
        throw exception();
    }
    int count_of_remaining = 0;
    for(int i = 0; i < colums_to_remove.row; i++)
    {
        if(colums_to_remove.data[i][0] != 1)
            count_of_remaining++;
    }
    Matrix *ret = new Matrix(this->row, count_of_remaining);
    int k = 0;
    for(int i = 0; i < this->row; i++)
    {
        for(int j = 0; j < this->col; j++)
        {
            if(colums_to_remove.data[j][0] != 1)
            {
                ret->data[i][k] = this->data[i][j];
                k++;
            }
        }
        k = 0;
    }
    return ret;
}

void print_mtx_list(Matrix **mtx, int list_len)
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

void print_mtx(Matrix &mtx)
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
