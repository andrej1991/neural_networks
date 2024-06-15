#include <gtest/gtest.h>
#include <iostream>
#include "../matrix/matrix.h"


using namespace std;

bool test_matrixes_for_equality(Matrix A, Matrix B, double epsilon = 1E-10){
    int rowA = A.get_row();
    int rowB = B.get_row();
    int colA = A.get_col();
    int colB = B.get_col();
    if(rowA != rowB)
        return false;
    if(colA != colB)
        return false;
    for(int r = 0; r < rowA; r++){
        for(int c = 0; c < colA; c++){
            if( abs(A.data[r][c] - B.data[r][c]) > epsilon){
                cerr << "[r:" << r << "  " << "c:" << c << "] [ERROR]\n"; 
                cerr << "[ result: " << A.data[r][c] << "] [ERROR]\n";
                cerr << "[ expected result: " << B.data[r][c] << "] [ERROR]\n"; 
                return false;
            }
        }
    }
    return true;
}

TEST(MatrixBasicOperationsTest, test_mtx_to_mtx_multyply_pass){
    Matrix A(4,5), B(5,7), Result(1,1), ExpectedResult(4,7);
    double helper1[A.get_row()][A.get_col()] = {{5, 2, 7, 9, 1.1},
                                                {11, 7, 8, 2.3, 57},
                                                {0, 4, 1, 3, 5},
                                                {7, 1, 2, 6, 6}};
    for (int row = 0; row < A.get_row(); row++){
        for (int col = 0; col < A.get_col(); col++){
            A.data[row][col] = helper1[row][col];
        }
    }
    double helper2[B.get_row()][B.get_col()] = {{2.71, 8, 1, 5, 4, 5, 3.2},
                                                {6, 3, 2, 5, 1.5, 3, 5},
                                                {4, 9, 8, 1, 8.2, 7, 8},
                                                {7, 2, 11, 41, 1, 1, 3},
                                                {8, 4, 1, 9, 7, 3, 4}};
    for (int row = 0; row < B.get_row(); row++){
        for (int col = 0; col < B.get_col(); col++){
            B.data[row][col] = helper2[row][col];
        }
    }
    double helper3[ExpectedResult.get_row()][ExpectedResult.get_col()] = {{125.35, 131.4, 165.1, 420.9, 97.1, 92.3, 113.4},
                                                                         {575.91, 413.6, 171.3, 705.3, 521.4, 305.3, 369.1},
                                                                         {89, 47, 54, 189, 52.2, 37, 57},
                                                                         {122.97, 113, 97, 342, 93.9, 76, 85.4}};
    for (int row = 0; row < ExpectedResult.get_row(); row++){
        for (int col = 0; col < ExpectedResult.get_col(); col++){
            ExpectedResult.data[row][col] = helper3[row][col];
        }
    }
    Result = A * B;
    ASSERT_TRUE(test_matrixes_for_equality(Result, ExpectedResult));
}

TEST(MatrixBasicOperationsTest, test_mtx_to_double_multyply_pass){
    Matrix A(2,3), Result(2,3), ExpectedResult(2,3);
    double x = 3;
    double helper1[A.get_row()][A.get_col()] = {{31.28, 19.5, 4},
                                                {9, 87.1, 8}};
    for (int row = 0; row < A.get_row(); row++){
        for (int col = 0; col < A.get_col(); col++){
            A.data[row][col] = helper1[row][col];
        }
    }
    double helper2[ExpectedResult.get_row()][ExpectedResult.get_col()] = {{93.84, 58.5, 12},
                                                                          {27, 261.3, 24}};
    for (int row = 0; row < ExpectedResult.get_row(); row++){
        for (int col = 0; col < ExpectedResult.get_col(); col++){
            ExpectedResult.data[row][col] = helper2[row][col];
        }
    }
    Result = A*x;
    ASSERT_TRUE(test_matrixes_for_equality(Result, ExpectedResult));
}

TEST(MatrixBasicOperationsTest, test_divide_matrixes_pass){
    Matrix A(3,5), B(3,5), Result(3,5), ExpectedResult(3,5);
    double helper1[A.get_row()][A.get_col()] = {{3.2, 3.4, 4, 1, 3.14},
                                                {5.7, 7.7, 1, 39, 58},
                                                {0.18, 2, 4, 5, 1024}};
    for (int row = 0; row < A.get_row(); row++){
        for (int col = 0; col < A.get_col(); col++){
            A.data[row][col] = helper1[row][col];
        }
    }
    double helper2[B.get_row()][B.get_col()] = {{2, 8, 5, 50, 2},
                                                {6, 7, 89, 2, 4},
                                                {9, 10, 31, 40, 64}};
    for (int row = 0; row < B.get_row(); row++){
        for (int col = 0; col < B.get_col(); col++){
            B.data[row][col] = helper2[row][col];
        }
    }
    double helper3[ExpectedResult.get_row()][ExpectedResult.get_col()] = {{1.6, 0.425, 0.8, 0.02, 1.57},
                                                                          {0.95, 1.1, 0.011235955, 19.5, 14.5},
                                                                          {0.02, 0.2, 0.129032258, 0.125, 16}};
    for (int row = 0; row < ExpectedResult.get_row(); row++){
        for (int col = 0; col < ExpectedResult.get_col(); col++){
            ExpectedResult.data[row][col] = helper3[row][col];
        }
    }
    Result = A / B;
    ASSERT_TRUE(test_matrixes_for_equality(Result, ExpectedResult));
}

TEST(MatrixBasicOperationsTest, test_mtx_to_mtx_addition_pass){
    Matrix A(3,5), B(3,5), Result(3,5), ExpectedResult(3,5);
    double helper1[A.get_row()][A.get_col()] = {{3.2, 3.5, 4, 1, 3.14},
                                                {5.6, 7.8, 1, 39, 58},
                                                {0.001, 2, 4, 8, 1024}};
    for (int row = 0; row < A.get_row(); row++){
        for (int col = 0; col < A.get_col(); col++){
            A.data[row][col] = helper1[row][col];
        }
    }
    double helper2[B.get_row()][B.get_col()] = {{21, 49, 37.215, 51, 2},
                                                {6, 7, 89, 2, 3},
                                                {9, 10, 31, 49, 57.829}};
    for (int row = 0; row < B.get_row(); row++){
        for (int col = 0; col < B.get_col(); col++){
            B.data[row][col] = helper2[row][col];
        }
    }
    double helper3[ExpectedResult.get_row()][ExpectedResult.get_col()] = {{24.2, 52.5, 41.215, 52, 5.14},
                                                                          {11.6, 14.8, 90, 41, 61},
                                                                          {9.001, 12, 35, 57, 1081.829}};
    for (int row = 0; row < ExpectedResult.get_row(); row++){
        for (int col = 0; col < ExpectedResult.get_col(); col++){
            ExpectedResult.data[row][col] = helper3[row][col];
        }
    }
    Result = A + B;
    ASSERT_TRUE(test_matrixes_for_equality(Result, ExpectedResult));
}

TEST(MatrixBasicOperationsTest, test_mtx_to_double_addition_pass){
    Matrix A(4,3), Result(4,3), ExpectedResult(4,3);
    double x = 3.14159;
    double helper1[A.get_row()][A.get_col()] = {{3.2, 3.5, 4},
                                                {5.6, 7.8, 1},
                                                {0.001, 2, 4},
                                                {5, 7.2, 8}};
    for (int row = 0; row < A.get_row(); row++){
        for (int col = 0; col < A.get_col(); col++){
            A.data[row][col] = helper1[row][col];
        }
    }
    double helper2[ExpectedResult.get_row()][ExpectedResult.get_col()] = {{6.34159, 6.64159, 7.14159},
                                                                          {8.74159, 10.94159, 4.14159},
                                                                          {3.14259, 5.14159, 7.14159},
                                                                          {8.14159, 10.34159, 11.14159}};
    for (int row = 0; row < ExpectedResult.get_row(); row++){
        for (int col = 0; col < ExpectedResult.get_col(); col++){
            ExpectedResult.data[row][col] = helper2[row][col];
        }
    }
    Result = A+x;
    ASSERT_TRUE(test_matrixes_for_equality(Result, ExpectedResult));
}

TEST(MatrixBasicOperationsTest, test_mtx_plus_equal_mtx_pass){
    Matrix A(3,5), B(3,5), ExpectedResult(3,5);
    double helper1[A.get_row()][A.get_col()] = {{3.2, 3.5, 4, 1, 3.14},
                                                {5.6, 7.8, 15, 39, 58},
                                                {0.001, 2, 4, 8, 1024}};
    for (int row = 0; row < A.get_row(); row++){
        for (int col = 0; col < A.get_col(); col++){
            A.data[row][col] = helper1[row][col];
        }
    }
    double helper2[B.get_row()][B.get_col()] = {{21, 49, 37.215, 51, 2},
                                                {6, 7, 89, 2, 3},
                                                {9, 10, 31, 49, 51.8219}};
    for (int row = 0; row < B.get_row(); row++){
        for (int col = 0; col < B.get_col(); col++){
            B.data[row][col] = helper2[row][col];
        }
    }
    double helper3[ExpectedResult.get_row()][ExpectedResult.get_col()] = {{24.2, 52.5, 41.215, 52, 5.14},
                                                                          {11.6, 14.8, 104, 41, 61},
                                                                          {9.001, 12, 35, 57, 1075.8219}};
    for (int row = 0; row < ExpectedResult.get_row(); row++){
        for (int col = 0; col < ExpectedResult.get_col(); col++){
            ExpectedResult.data[row][col] = helper3[row][col];
        }
    }
    A += B;
    ASSERT_TRUE(test_matrixes_for_equality(A, ExpectedResult));
}

TEST(MatrixBasicOperationsTest, test_mtx_plus_equal_double_pass){
    Matrix A(3,2), ExpectedResult(3,2);
    double helper1[A.get_row()][A.get_col()] = {{1, 3.6},
                                                {5.6, 39},
                                                {5.301, 2}};
    for (int row = 0; row < A.get_row(); row++){
        for (int col = 0; col < A.get_col(); col++){
            A.data[row][col] = helper1[row][col];
        }
    }
    double d = 7.2;
    double helper2[ExpectedResult.get_row()][ExpectedResult.get_col()] = {{8.2, 10.8},
                                                                          {12.8, 46.2},
                                                                          {12.501, 9.2}};
    for (int row = 0; row < ExpectedResult.get_row(); row++){
        for (int col = 0; col < ExpectedResult.get_col(); col++){
            ExpectedResult.data[row][col] = helper2[row][col];
        }
    }
    A += d;
    ASSERT_TRUE(test_matrixes_for_equality(A, ExpectedResult));
}

TEST(MatrixBasicOperationsTest, test_hadamart_pass){
    Matrix A(2,3), B(2,3), Result(2,3), ExpectedResult(2,3);
    double helper1[A.get_row()][A.get_col()] = {{1, 2, 3},
                                                {18, 19, 31}};
    for (int row = 0; row < A.get_row(); row++){
        for (int col = 0; col < A.get_col(); col++){
            A.data[row][col] = helper1[row][col];
        }
    }
    double helper2[B.get_row()][B.get_col()] = {{4, 5, 37.215},
                                                {9, 8, 1.2}};
    for (int row = 0; row < B.get_row(); row++){
        for (int col = 0; col < B.get_col(); col++){
            B.data[row][col] = helper2[row][col];
        }
    }
    double helper3[ExpectedResult.get_row()][ExpectedResult.get_col()] = {{4, 10, 111.645},
                                                                          {162, 152, 37.2}};
    for (int row = 0; row < ExpectedResult.get_row(); row++){
        for (int col = 0; col < ExpectedResult.get_col(); col++){
            ExpectedResult.data[row][col] = helper3[row][col];
        }
    }
    Result = hadamart_product(A, B);
    ASSERT_TRUE(test_matrixes_for_equality(Result, ExpectedResult));
}

TEST(MatrixBasicOperationsTest, test_transpose_pass){
    Matrix A(3,2), Result(2,3), ExpectedResult(2,3);
    double helper1[A.get_row()][A.get_col()] = {{1, 2},
                                                {4, 8},
                                                {3, 4}};
    for (int row = 0; row < A.get_row(); row++){
        for (int col = 0; col < A.get_col(); col++){
            A.data[row][col] = helper1[row][col];
        }
    }
    double helper2[ExpectedResult.get_row()][ExpectedResult.get_col()] = {{1, 4, 3},
                                                                          {2, 8, 4}};
    for (int row = 0; row < ExpectedResult.get_row(); row++){
        for (int col = 0; col < ExpectedResult.get_col(); col++){
            ExpectedResult.data[row][col] = helper2[row][col];
        }
    }
    Result = A.transpose();
    ASSERT_TRUE(test_matrixes_for_equality(Result, ExpectedResult));
}

TEST(MatrixBasicOperationsTest, test_convolution_positive1){
    Matrix Input(7,8), Kernel(2,2), Result(6, 7), ExpectedResult(6,7);
    double helper1[Input.get_row()][Input.get_col()] = {{1, 2, 3, 4, 5, 6, 7, 8},
                                                        {9, 10, 11, 12, 13, 14, 15, 16},
                                                        {17, 18, 19, 20, 21, 22, 23, 24},
                                                        {25, 26, 27, 28, 29, 30, 31, 32},
                                                        {33, 34, 35, 36, 37, 38, 39, 40},
                                                        {41, 42, 43, 44, 45, 46, 47, 48},
                                                        {49, 50, 51, 52, 53, 54, 55, 56}};
    double helper2[Kernel.get_row()][Kernel.get_col()] = {{1.1, 2.2},
                                                          {3.3, 4.4}};
    double helper3[ExpectedResult.get_row()][ExpectedResult.get_col()] = {{41.8, 52.8, 63.8, 74.8, 85.8, 96.8, 107.8},
                                                                          {129.8, 140.8, 151.8, 162.8, 173.8, 184.8, 195.8},
                                                                          {217.8, 228.8, 239.8, 250.8, 261.8, 272.8, 283.8},
                                                                          {305.8, 316.8, 327.8, 338.8, 349.8, 360.8, 371.8},
                                                                          {393.8, 404.8, 415.8, 426.8, 437.8, 448.8, 459.8},
                                                                          {481.8, 492.8, 503.8, 514.8, 525.8, 536.8, 547.8}};
    for (int row=0; row<Input.get_row(); row++){
        for (int col=0; col < Input.get_col(); col++){
            Input.data[row][col] = helper1[row][col];
        }
    }
    for(int row=0; row<Kernel.get_row(); row++){
        for(int col=0; col<Kernel.get_col(); col++){
            Kernel.data[row][col] = helper2[row][col];
        }
    }
    for(int row=0; row<ExpectedResult.get_row(); row++){
        for(int col=0; col<ExpectedResult.get_col(); col++){
            ExpectedResult.data[row][col] = helper3[row][col];
        }
    }
    convolution(Input, Kernel, Result, 1, 1);
    ASSERT_TRUE(test_matrixes_for_equality(Result, ExpectedResult));
}

TEST(MatrixBasicOperationsTest, test_convolution_positive1_with_strides){
    Matrix Input(7,8), Kernel(2,2), Result(3, 4), ExpectedResult(3,4);
    double helper1[Input.get_row()][Input.get_col()] = {{1, 2, 3, 4, 5, 6, 7, 8},
                                                        {9, 10, 11, 12, 13, 14, 15, 16},
                                                        {17, 18, 19, 20, 21, 22, 23, 24},
                                                        {25, 26, 27, 28, 29, 30, 31, 32},
                                                        {33, 34, 35, 36, 37, 38, 39, 40},
                                                        {41, 42, 43, 44, 45, 46, 47, 48},
                                                        {49, 50, 51, 52, 53, 54, 55, 56}};
    double helper2[Kernel.get_row()][Kernel.get_col()] = {{1.1, 2.2},
                                                          {3.3, 4.4}};
    double helper3[ExpectedResult.get_row()][ExpectedResult.get_col()] = {{41.8, 63.8, 85.8, 107.8},
                                                                         {217.8, 239.8, 261.8, 283.8},
                                                                         {393.8, 415.8, 437.8, 459.8}};
    for (int row=0; row<Input.get_row(); row++){
        for (int col=0; col < Input.get_col(); col++){
            Input.data[row][col] = helper1[row][col];
        }
    }
    for(int row=0; row<Kernel.get_row(); row++){
        for(int col=0; col<Kernel.get_col(); col++){
            Kernel.data[row][col] = helper2[row][col];
        }
    }
    for(int row=0; row<ExpectedResult.get_row(); row++){
        for(int col=0; col<ExpectedResult.get_col(); col++){
            ExpectedResult.data[row][col] = helper3[row][col];
        }
    }
    convolution(Input, Kernel, Result, 2, 2);
    ASSERT_TRUE(test_matrixes_for_equality(Result, ExpectedResult));
}

TEST(MatrixBasicOperationsTest, test_cross_correlation_positive1){
    Matrix Input(7,8), Kernel(2,2), Result(6, 7), ExpectedResult(6,7);
    double helper1[Input.get_row()][Input.get_col()] = {{1, 2, 4, 8, 16, 32, 64, 128},
                                                        {3, 6, 9, 12, 18, 21, 24, 27},
                                                        {2, 3, 5, 7, 11, 13, 17, 19},
                                                        {256, 512, 1024, 2048, 4096, 8192, 16384, 32768},
                                                        {30, 33, 36, 39, 42, 45, 48, 51},
                                                        {23, 29, 31, 37, 41, 47, 53, 59},
                                                        {49, 50, 51, 52, 53, 54, 55, 56}};
    double helper2[Kernel.get_row()][Kernel.get_col()] = {{1.1, 2.2},
                                                          {3.3, 4.4}};
    double helper3[ExpectedResult.get_row()][ExpectedResult.get_col()] = {{41.8, 70.4, 104.5, 162.8, 239.8, 350.9, 550},
                                                                          {36.3, 58.3, 83.6, 124.3, 159.5, 193.6, 225.5},
                                                                          {3106.4, 6209.5, 12411.3, 24812.7, 49602.3, 99174.9, 198306.9},
                                                                          {1652.2, 3083.3, 5922.4, 11577.5, 22864.6, 45415.7, 90494.8},
                                                                          {309.1, 347.6, 390.5, 437.8, 487.3, 543.4, 599.5},
                                                                          {470.8, 489.5, 512.6, 535.7, 561, 588.5, 616}};
    for (int row=0; row<Input.get_row(); row++){
        for (int col=0; col < Input.get_col(); col++){
            Input.data[row][col] = helper1[row][col];
        }
    }
    for(int row=0; row<Kernel.get_row(); row++){
        for(int col=0; col<Kernel.get_col(); col++){
            Kernel.data[row][col] = helper2[row][col];
        }
    }
    for(int row=0; row<ExpectedResult.get_row(); row++){
        for(int col=0; col<ExpectedResult.get_col(); col++){
            ExpectedResult.data[row][col] = helper3[row][col];
        }
    }
    cross_correlation(Input, Kernel, Result, 1, 1);
    ASSERT_TRUE(test_matrixes_for_equality(Result, ExpectedResult));
}

TEST(MatrixBasicOperationsTest, test_cross_correlation_positive1_with_strides2){
    Matrix Input(7,8), Kernel(2,2), Result(3, 4), ExpectedResult(3,4);
    double helper1[Input.get_row()][Input.get_col()] = {{1, 2, 4, 8, 16, 32, 64, 128},
                                                        {3, 6, 9, 12, 18, 21, 24, 27},
                                                        {2, 3, 5, 7, 11, 13, 17, 19},
                                                        {256, 512, 1024, 2048, 4096, 8192, 16384, 32768},
                                                        {30, 33, 36, 39, 42, 45, 48, 51},
                                                        {23, 29, 31, 37, 41, 47, 53, 59},
                                                        {49, 50, 51, 52, 53, 54, 55, 56}};
    double helper2[Kernel.get_row()][Kernel.get_col()] = {{1.1, 2.2},
                                                          {3.3, 4.4}};
    double helper3[ExpectedResult.get_row()][ExpectedResult.get_col()] = {{41.8, 104.5, 239.8, 550},
                                                                          {3106.4, 12411.3, 49602.3, 198306.9},
                                                                          {309.1, 390.5, 487.3, 599.5}};
    for (int row=0; row<Input.get_row(); row++){
        for (int col=0; col < Input.get_col(); col++){
            Input.data[row][col] = helper1[row][col];
        }
    }
    for(int row=0; row<Kernel.get_row(); row++){
        for(int col=0; col<Kernel.get_col(); col++){
            Kernel.data[row][col] = helper2[row][col];
        }
    }
    for(int row=0; row<ExpectedResult.get_row(); row++){
        for(int col=0; col<ExpectedResult.get_col(); col++){
            ExpectedResult.data[row][col] = helper3[row][col];
        }
    }
    cross_correlation(Input, Kernel, Result, 2, 2);
    ASSERT_TRUE(test_matrixes_for_equality(Result, ExpectedResult));
}

TEST(MatrixBasicOperationsTest, test_cross_correlation_positive1_with_strides3){
    Matrix Input(7,8), Kernel(2,2), Result(6, 7), ExpectedResult(6,7);
    double helper1[Input.get_row()][Input.get_col()] = {{1, 2, 4, 8, 16, 32, 64, 128},
                                                        {3, 6, 9, 12, 18, 21, 24, 27},
                                                        {2, 3, 5, 7, 11, 13, 17, 19},
                                                        {256, 512, 1024, 2048, 4096, 8192, 16384, 32768},
                                                        {30, 33, 36, 39, 42, 45, 48, 51},
                                                        {23, 29, 31, 37, 41, 47, 53, 59},
                                                        {49, 50, 51, 52, 53, 54, 55, 56}};
    double helper2[Kernel.get_row()][Kernel.get_col()] = {{1.1, 2.2},
                                                          {3.3, 4.4}};
    double helper3[ExpectedResult.get_row()][ExpectedResult.get_col()] = {{41.8,162.8, 550},
                                                                          {1652.2,11577.5, 90494.8}};
    for (int row=0; row<Input.get_row(); row++){
        for (int col=0; col < Input.get_col(); col++){
            Input.data[row][col] = helper1[row][col];
        }
    }
    for(int row=0; row<Kernel.get_row(); row++){
        for(int col=0; col<Kernel.get_col(); col++){
            Kernel.data[row][col] = helper2[row][col];
        }
    }
    for(int row=0; row<ExpectedResult.get_row(); row++){
        for(int col=0; col<ExpectedResult.get_col(); col++){
            ExpectedResult.data[row][col] = helper3[row][col];
        }
    }
    cross_correlation(Input, Kernel, Result, 3, 3);
    ASSERT_TRUE(test_matrixes_for_equality(Result, ExpectedResult));
}


TEST(MatrixBasicOperationsTest, test_rot180_positive){
    Matrix A(4,3), Result(4,3), ExpectedResult(4,3);
    double helper1[A.get_row()][A.get_col()] = {{1, 2, 4},
                                                {8, 16, 32},
                                                {64, 128, 256},
                                                {512, 1024, 2048}};
    for (int row = 0; row < A.get_row(); row++){
        for (int col = 0; col < A.get_col(); col++){
            A.data[row][col] = helper1[row][col];
        }
    }
    double helper2[ExpectedResult.get_row()][ExpectedResult.get_col()] = {{2048, 1024, 512},
                                                                          {256, 128, 64},
                                                                          {32, 16, 8},
                                                                          {4, 2, 1}};
    for (int row = 0; row < ExpectedResult.get_row(); row++){
        for (int col = 0; col < ExpectedResult.get_col(); col++){
            ExpectedResult.data[row][col] = helper2[row][col];
        }
    }
    Result = A.rot180();
    ASSERT_TRUE(test_matrixes_for_equality(Result, ExpectedResult));
}

TEST(MatrixBasicOperationsTest, test_sqroot_positive){
    Matrix A(4,3), Result(4,3), ExpectedResult(4,3);
    double helper1[A.get_row()][A.get_col()] = {{0, 1, 4},
                                                {9, 16, 49},
                                                {81, 25, 256},
                                                {400, 100, 36}};
    for (int row = 0; row < A.get_row(); row++){
        for (int col = 0; col < A.get_col(); col++){
            A.data[row][col] = helper1[row][col];
        }
    }
    double helper2[ExpectedResult.get_row()][ExpectedResult.get_col()] = {{0, 1, 2},
                                                                          {3, 4, 7},
                                                                          {9, 5, 16},
                                                                          {20, 10, 6}};
    for (int row = 0; row < ExpectedResult.get_row(); row++){
        for (int col = 0; col < ExpectedResult.get_col(); col++){
            ExpectedResult.data[row][col] = helper2[row][col];
        }
    }
    Result = A.sqroot();
    ASSERT_TRUE(test_matrixes_for_equality(Result, ExpectedResult));
}

TEST(MatrixBasicOperationsTest, test_square_element_by_positive){
    Matrix A(4,3), Result(4,3), ExpectedResult(4,3);
    double helper1[A.get_row()][A.get_col()] = {{0, 1, 2},
                                                {3, 4, 7},
                                                {9, 5, 16},
                                                {20, 10, 6}};
    for (int row = 0; row < A.get_row(); row++){
        for (int col = 0; col < A.get_col(); col++){
            A.data[row][col] = helper1[row][col];
        }
    }
    double helper2[ExpectedResult.get_row()][ExpectedResult.get_col()] = {{0, 1, 4},
                                                                          {9, 16, 49},
                                                                          {81, 25, 256},
                                                                          {400, 100, 36}};
    for (int row = 0; row < ExpectedResult.get_row(); row++){
        for (int col = 0; col < ExpectedResult.get_col(); col++){
            ExpectedResult.data[row][col] = helper2[row][col];
        }
    }
    Result = A.square_element_by();
    ASSERT_TRUE(test_matrixes_for_equality(Result, ExpectedResult));
}

TEST(MatrixBasicOperationsTest, test_dilate_stride1_positive){
    Matrix A(4,3), Result(4,3), ExpectedResult(4,3);
    double helper1[A.get_row()][A.get_col()] = {{0, 1, 2},
                                                {3, 4, 7},
                                                {9, 5, 16},
                                                {20, 10, 6}};
    for (int row = 0; row < A.get_row(); row++){
        for (int col = 0; col < A.get_col(); col++){
            A.data[row][col] = helper1[row][col];
        }
    }
    double helper2[ExpectedResult.get_row()][ExpectedResult.get_col()] = {{0, 1, 2},
                                                                          {3, 4, 7},
                                                                          {9, 5, 16},
                                                                          {20, 10, 6}};
    for (int row = 0; row < ExpectedResult.get_row(); row++){
        for (int col = 0; col < ExpectedResult.get_col(); col++){
            ExpectedResult.data[row][col] = helper2[row][col];
        }
    }
    Result = A.dilate(1, 1);
    ASSERT_TRUE(test_matrixes_for_equality(Result, ExpectedResult));
}

TEST(MatrixBasicOperationsTest, test_dilate_wider_stride_positive){
    Matrix A(4,3), Result(7,7), ExpectedResult(7,7);
    double helper1[A.get_row()][A.get_col()] = {{0, 1, 2},
                                                {3, 4, 7},
                                                {9, 5, 16},
                                                {20, 10, 6}};
    for (int row = 0; row < A.get_row(); row++){
        for (int col = 0; col < A.get_col(); col++){
            A.data[row][col] = helper1[row][col];
        }
    }
    double helper2[ExpectedResult.get_row()][ExpectedResult.get_col()] = {{0, 0, 0, 1, 0, 0, 2},
                                                                          {0, 0, 0, 0, 0, 0, 0},
                                                                          {3, 0, 0, 4, 0, 0, 7},
                                                                          {0, 0, 0, 0, 0, 0, 0},
                                                                          {9, 0, 0, 5, 0, 0, 16},
                                                                          {0, 0, 0, 0, 0, 0, 0},
                                                                          {20, 0, 0, 10, 0, 0, 6}};
    for (int row = 0; row < ExpectedResult.get_row(); row++){
        for (int col = 0; col < ExpectedResult.get_col(); col++){
            ExpectedResult.data[row][col] = helper2[row][col];
        }
    }
    Result = A.dilate(2, 3);
    ASSERT_TRUE(test_matrixes_for_equality(Result, ExpectedResult));
}

TEST(MatrixBasicOperationsTest, test_zero_padd_positive){
    Matrix A(4,3), Result(8,9), ExpectedResult(8,9);
    double helper1[A.get_row()][A.get_col()] = {{0, 1, 2},
                                                {3, 4, 7},
                                                {9, 5, 16},
                                                {20, 10, 6}};
    for (int row = 0; row < A.get_row(); row++){
        for (int col = 0; col < A.get_col(); col++){
            A.data[row][col] = helper1[row][col];
        }
    }
    double helper2[ExpectedResult.get_row()][ExpectedResult.get_col()] = {{0, 0, 0, 0, 0, 0, 0, 0, 0},
                                                                          {0, 0, 0, 0, 0, 1, 2, 0, 0},
                                                                          {0, 0, 0, 0, 3, 4, 7, 0, 0},
                                                                          {0, 0, 0, 0, 9, 5, 16, 0, 0},
                                                                          {0, 0, 0, 0, 20, 10, 6, 0, 0},
                                                                          {0, 0, 0, 0, 0, 0, 0, 0, 0},
                                                                          {0, 0, 0, 0, 0, 0, 0, 0, 0},
                                                                          {0, 0, 0, 0, 0, 0, 0, 0, 0},};
    for (int row = 0; row < ExpectedResult.get_row(); row++){
        for (int col = 0; col < ExpectedResult.get_col(); col++){
            ExpectedResult.data[row][col] = helper2[row][col];
        }
    }
    Result = A.zero_padd(1, 2, 3, 4);
    ASSERT_TRUE(test_matrixes_for_equality(Result, ExpectedResult));
}

TEST(MatrixBasicOperationsTest, test_remove_rows_positive){
    ///TODO teardown
    Matrix A(7,8), *Result, ExpectedResult(4,8), RowsToRemove(7,1);
    double helper1[A.get_row()][A.get_col()] = {{1, 2, 4, 8, 16, 32, 64, 128},
                                                {3, 6, 9, 12, 18, 21, 24, 27},
                                                {2, 3, 5, 7, 11, 13, 17, 19},
                                                {256, 512, 1024, 2048, 4096, 8192, 16384, 32768},
                                                {30, 33, 36, 39, 42, 45, 48, 51},
                                                {23, 29, 31, 37, 41, 47, 53, 59},
                                                {49, 50, 51, 52, 53, 54, 55, 56}};
    for (int row = 0; row < A.get_row(); row++){
        for (int col = 0; col < A.get_col(); col++){
            A.data[row][col] = helper1[row][col];
        }
    }
    RowsToRemove.data[1][0] = 1;
    RowsToRemove.data[4][0] = 1;
    RowsToRemove.data[5][0] = 1;
    double helper2[ExpectedResult.get_row()][ExpectedResult.get_col()] = {{1, 2, 4, 8, 16, 32, 64, 128},
                                                                          {2, 3, 5, 7, 11, 13, 17, 19},
                                                                          {256, 512, 1024, 2048, 4096, 8192, 16384, 32768},
                                                                          {49, 50, 51, 52, 53, 54, 55, 56}};
    for (int row = 0; row < ExpectedResult.get_row(); row++){
        for (int col = 0; col < ExpectedResult.get_col(); col++){
            ExpectedResult.data[row][col] = helper2[row][col];
        }
    }
    Result = A.remove_rows(RowsToRemove);
    ASSERT_TRUE(test_matrixes_for_equality(*Result, ExpectedResult));
}

TEST(MatrixBasicOperationsTest, test_remove_colums_positive){
    ///TODO teardown
    Matrix A(7,8), *Result, ExpectedResult(7,5), ColumsToRemove(8,1);
    double helper1[A.get_row()][A.get_col()] = {{1, 2, 4, 8, 16, 32, 64, 128},
                                                {3, 6, 9, 12, 18, 21, 24, 27},
                                                {2, 3, 5, 7, 11, 13, 17, 19},
                                                {256, 512, 1024, 2048, 4096, 8192, 16384, 32768},
                                                {30, 33, 36, 39, 42, 45, 48, 51},
                                                {23, 29, 31, 37, 41, 47, 53, 59},
                                                {49, 50, 51, 52, 53, 54, 55, 56}};
    for (int row = 0; row < A.get_row(); row++){
        for (int col = 0; col < A.get_col(); col++){
            A.data[row][col] = helper1[row][col];
        }
    }
    ColumsToRemove.data[1][0] = 1;
    ColumsToRemove.data[4][0] = 1;
    ColumsToRemove.data[5][0] = 1;
    double helper2[ExpectedResult.get_row()][ExpectedResult.get_col()] = {{1, 4, 8, 64, 128},
                                                                          {3, 9, 12, 24, 27},
                                                                          {2, 5, 7, 17, 19},
                                                                          {256, 1024, 2048, 16384, 32768},
                                                                          {30, 36, 39, 48, 51},
                                                                          {23, 31, 37, 53, 59},
                                                                          {49, 51, 52, 55, 56}};
    for (int row = 0; row < ExpectedResult.get_row(); row++){
        for (int col = 0; col < ExpectedResult.get_col(); col++){
            ExpectedResult.data[row][col] = helper2[row][col];
        }
    }
    Result = A.remove_colums(ColumsToRemove);
    ASSERT_TRUE(test_matrixes_for_equality(*Result, ExpectedResult));
}

TEST(MatrixBasicOperationsTest, test_sum_over_elements_positive){
    Matrix A(4,3);
    double Result, ExpectedResult;
    double helper1[A.get_row()][A.get_col()] = {{0, 1, 2},
                                                {3, 4, 7},
                                                {9, 5, 16},
                                                {20, 10, 6}};
    for (int row = 0; row < A.get_row(); row++){
        for (int col = 0; col < A.get_col(); col++){
            A.data[row][col] = helper1[row][col];
        }
    }
    ExpectedResult = 83;
    Result = A.sum_over_elements();
    ASSERT_DOUBLE_EQ (ExpectedResult, Result);
}

TEST(MatrixBasicOperationsTest, test_squared_sum_over_elements_positive){
    Matrix A(4,3);
    double Result, ExpectedResult;
    double helper1[A.get_row()][A.get_col()] = {{0, 1, 2},
                                                {3, 4, 7},
                                                {9, 5, 16},
                                                {20, 10, 6}};
    for (int row = 0; row < A.get_row(); row++){
        for (int col = 0; col < A.get_col(); col++){
            A.data[row][col] = helper1[row][col];
        }
    }
    ExpectedResult = 977;
    Result = A.squared_sum_over_elements();
    ASSERT_DOUBLE_EQ (ExpectedResult, Result);
}

TEST(MatrixBasicOperationsTest, test_multiply_with_transpose_positive){
    Matrix A(2,3), B(3,3);
    double helper1[A.get_row()][A.get_col()] = {{0, 1, 2},
                                                {3, 4, 7}};
    for (int row = 0; row < A.get_row(); row++){
        for (int col = 0; col < A.get_col(); col++){
            A.data[row][col] = helper1[row][col];
        }
    }
    double helper2[B.get_row()][B.get_col()] = {{0, 1, 2},
                                                {3, 4, 7},
                                                {9, 5, 16}};
    for (int row = 0; row < B.get_row(); row++){
        for (int col = 0; col < B.get_col(); col++){
            B.data[row][col] = helper2[row][col];
        }
    }
    Matrix Result1 = A * B.transpose();
    Matrix Result2 = A.multiply_with_transpose(B);
    ASSERT_TRUE(test_matrixes_for_equality(Result1, Result2));
    
}


int main(int argc, char **argv){
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
