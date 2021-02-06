#include <gtest/gtest.h>
#include <iostream>
#include "../matrix/matrix.h"


using namespace std;

bool test_matrixes_for_equality(Matrix A, Matrix B)
{
    int rowA = A.get_row();
    int rowB = B.get_row();
    int colA = A.get_col();
    int colB = B.get_col();
    if(rowA != rowB)
        return false;
    if(colA != colB)
        return false;
    for(int r = 0; r < rowA; r++)
    {
        for(int c = 0; c < colA; c++)
        {
            if( abs(A.data[r][c] - B.data[r][c]) > 1E-10)
            {
                cerr << "[r:" << r << "  " << "c:" << c << "] [ERROR]\n"; 
                cerr << "[ result: " << A.data[r][c] << "] [ERROR]\n";
                cerr << "[ expected result: " << B.data[r][c] << "] [ERROR]\n"; 
                return false;
            }            
        }
    }
    return true;
}

void test_mtx_to_mtx_multyply_pass()
{
    Matrix A(4,5), B(5,7), Result(1,1), ExpectedResult(4,7);
    double helper1[4][5] = {{5, 2, 7, 9, 1.1},
                           {11, 7, 8, 2.3, 57},
                           {0, 4, 1, 3, 5},
                           {7, 1, 2, 6, 6}};
    for (int row = 0; row < 4; row++)
    {
        for (int col = 0; col < 5; col++)
        {
            A.data[row][col] = helper1[row][col];
        }
    }
    double helper2[5][7] = {{2.71, 8, 1, 5, 4, 5, 3.2},
                           {6, 3, 2, 5, 1.5, 3, 5},
                           {4, 9, 8, 1, 8.2, 7, 8},
                           {7, 2, 11, 41, 1, 1, 3},
                           {8, 4, 1, 9, 7, 3, 4}};
    for (int row = 0; row < 5; row++)
    {
        for (int col = 0; col < 7; col++)
        {
            B.data[row][col] = helper2[row][col];
        }
    }
    double helper3[4][7] = {{125.35, 131.4, 165.1, 420.9, 97.1, 92.3, 113.4},
                           {575.91, 413.6, 171.3, 705.3, 521.4, 305.3, 369.1},
                           {89, 47, 54, 189, 52.2, 37, 57},
                           {122.97, 113, 97, 342, 93.9, 76, 85.4}};
    for (int row = 0; row < 4; row++)
    {
        for (int col = 0; col < 7; col++)
        {
            ExpectedResult.data[row][col] = helper3[row][col];
        }
    }
    Result = A * B;
    ASSERT_EQ(true, test_matrixes_for_equality(Result, ExpectedResult));
}

void test_mtx_to_double_multyply_pass()
{
    Matrix A(4,3), Result(4,3), ExpectedResult(4,3);
    double x = 3;
    double helper1[4][3] = {{31.28, 19.5, 4},
                           {9, 87.1, 8}};
    for (int row = 0; row < 4; row++)
    {
        for (int col = 0; col < 3; col++)
        {
            A.data[row][col] = helper1[row][col];
        }
    }
    double helper2[4][3] = {{93.84, 58.5, 12},
                           {27, 261.3, 24}};
    for (int row = 0; row < 4; row++)
    {
        for (int col = 0; col < 3; col++)
        {
            ExpectedResult.data[row][col] = helper2[row][col];
        }
    }
    Result = A*x;
    ASSERT_EQ(true, test_matrixes_for_equality(Result, ExpectedResult));
}

void test_divide_matrixes_pass()
{
    Matrix A(3,5), B(3,5), Result(3,5), ExpectedResult(3,5);
    double helper1[3][5] = {{3.2, 3.4, 4, 1, 3.14},
                           {5.7, 7.7, 1, 39, 58},
                           {0.18, 2, 4, 5, 1024}};
    for (int row = 0; row < 3; row++)
    {
        for (int col = 0; col < 5; col++)
        {
            A.data[row][col] = helper1[row][col];
        }
    }
    double helper2[3][5] = {{2, 8, 5, 50, 2},
                           {6, 7, 89, 2, 4},
                           {9, 10, 31, 40, 64}};
    for (int row = 0; row < 3; row++)
    {
        for (int col = 0; col < 5; col++)
        {
            B.data[row][col] = helper2[row][col];
        }
    }
    double helper3[3][5] = {{1.6, 0.425, 0.8, 0.02, 1.57},
                           {0.95, 1.1, 0.011235955, 19.5, 14.5},
                           {0.02, 0.2, 0.129032258, 0.125, 16}};
    for (int row = 0; row < 3; row++)
    {
        for (int col = 0; col < 5; col++)
        {
            ExpectedResult.data[row][col] = helper3[row][col];
        }
    }
    Result = A / B;
    ASSERT_EQ(true, test_matrixes_for_equality(Result, ExpectedResult));
}

void test_mtx_to_mtx_addition_pass()
{
    Matrix A(3,5), B(3,5), Result(3,5), ExpectedResult(3,5);
    double helper1[3][5] = {{3.2, 3.5, 4, 1, 3.14},
                           {5.6, 7.8, 1, 39, 58},
                           {0.001, 2, 4, 8, 1024}};
    for (int row = 0; row < 3; row++)
    {
        for (int col = 0; col < 5; col++)
        {
            A.data[row][col] = helper1[row][col];
        }
    }
    double helper2[3][5] = {{21, 49, 37.215, 51, 2},
                           {6, 7, 89, 2, 3},
                           {9, 10, 31, 49, 57.829}};
    for (int row = 0; row < 3; row++)
    {
        for (int col = 0; col < 5; col++)
        {
            B.data[row][col] = helper2[row][col];
        }
    }
    double helper3[3][5] = {{24.2, 52.5, 41.215, 52, 5.14},
                           {11.6, 14.8, 90, 41, 61},
                           {9.001, 12, 35, 57, 1081.829}};
    for (int row = 0; row < 3; row++)
    {
        for (int col = 0; col < 5; col++)
        {
            ExpectedResult.data[row][col] = helper3[row][col];
        }
    }
    Result = A + B;
    ASSERT_EQ(true, test_matrixes_for_equality(Result, ExpectedResult));
}

void test_mtx_to_double_addition_pass()
{
    Matrix A(4,3), Result(4,3), ExpectedResult(4,3);
    double x = 3.14159;
    double helper1[4][3] = {{3.2, 3.5, 4},
                           {5.6, 7.8, 1},
                           {0.001, 2, 4},
                           {5, 7.2, 8}};
    for (int row = 0; row < 4; row++)
    {
        for (int col = 0; col < 3; col++)
        {
            A.data[row][col] = helper1[row][col];
        }
    }
    double helper2[4][3] = {{6.34159, 6.64159, 7.14159},
                           {8.74159, 10.94159, 4.14159},
                           {3.14259, 5.14159, 7.14159},
                           {8.14159, 10.34159, 11.14159}};
    for (int row = 0; row < 4; row++)
    {
        for (int col = 0; col < 3; col++)
        {
            ExpectedResult.data[row][col] = helper2[row][col];
        }
    }
    Result = A+x;
    ASSERT_EQ(true, test_matrixes_for_equality(Result, ExpectedResult));
}

void test_mtx_plus_equal_mtx_pass()
{
    Matrix A(3,5), B(3,5), ExpectedResult(3,5);
    double helper1[3][5] = {{3.2, 3.5, 4, 1, 3.14},
                           {5.6, 7.8, 15, 39, 58},
                           {0.001, 2, 4, 8, 1024}};
    for (int row = 0; row < 3; row++)
    {
        for (int col = 0; col < 5; col++)
        {
            A.data[row][col] = helper1[row][col];
        }
    }
    double helper2[3][5] = {{21, 49, 37.215, 51, 2},
                           {6, 7, 89, 2, 3},
                           {9, 10, 31, 49, 51.8219}};
    for (int row = 0; row < 3; row++)
    {
        for (int col = 0; col < 5; col++)
        {
            B.data[row][col] = helper2[row][col];
        }
    }
    double helper3[3][5] = {{24.2, 52.5, 41.215, 52, 5.14},
                           {11.6, 14.8, 104, 41, 61},
                           {9.001, 12, 35, 57, 1075.8219}};
    for (int row = 0; row < 3; row++)
    {
        for (int col = 0; col < 5; col++)
        {
            ExpectedResult.data[row][col] = helper3[row][col];
        }
    }
    A += B;
    ASSERT_EQ(true, test_matrixes_for_equality(A, ExpectedResult));
}

void test_mtx_plus_equal_double_pass()
{
    Matrix A(3,2), B(3,2), ExpectedResult(3,2);
    double helper1[3][2] = {{1, 3.6},
                           {5.6, 39},
                           {5.301, 2}};
    for (int row = 0; row < 3; row++)
    {
        for (int col = 0; col < 2; col++)
        {
            A.data[row][col] = helper1[row][col];
        }
    }
    double d = 7.2;
    double helper3[3][2] = {{8.2, 10.8},
                           {12.8, 46.2},
                           {12.501, 9.2}};
    for (int row = 0; row < 3; row++)
    {
        for (int col = 0; col < 2; col++)
        {
            ExpectedResult.data[row][col] = helper3[row][col];
        }
    }
    A += d;
    ASSERT_EQ(true, test_matrixes_for_equality(A, ExpectedResult));
}

void test_hadamart_pass()
{
    Matrix A(2,3), B(2,3), Result(2,3), ExpectedResult(2,3);
    double helper1[2][3] = {{1, 2, 3},
                           {18, 19, 31}};
    for (int row = 0; row < 2; row++)
    {
        for (int col = 0; col < 3; col++)
        {
            A.data[row][col] = helper1[row][col];
        }
    }
    double helper2[2][3] = {{4, 5, 37.215},
                           {9, 8, 1.2}};
    for (int row = 0; row < 2; row++)
    {
        for (int col = 0; col < 3; col++)
        {
            B.data[row][col] = helper2[row][col];
        }
    }
    double helper3[2][3] = {{4, 10, 111.645},
                           {162, 152, 37.2}};
    for (int row = 0; row < 2; row++)
    {
        for (int col = 0; col < 3; col++)
        {
            ExpectedResult.data[row][col] = helper3[row][col];
        }
    }
    Result = hadamart_product(A, B);
    ASSERT_EQ(true, test_matrixes_for_equality(Result, ExpectedResult));
}

void test_transpose_pass()
{
    Matrix A(3,2), Result(2,3), ExpectedResult(2,3);
    double helper1[3][2] = {{1, 2},
                           {4, 8},
                           {3, 4}};
    for (int row = 0; row < 3; row++)
    {
        for (int col = 0; col < 2; col++)
        {
            A.data[row][col] = helper1[row][col];
        }
    }
    double helper2[2][3] = {{1, 4, 3},
                           {2, 8, 4}};
    for (int row = 0; row < 2; row++)
    {
        for (int col = 0; col < 3; col++)
        {
            ExpectedResult.data[row][col] = helper2[row][col];
        }
    }
    Result = A.transpose();
    ASSERT_EQ(true, test_matrixes_for_equality(Result, ExpectedResult));
}

void test_convolution_positive1()
{
    Matrix Input(7,8), Kernel(2,2), Result(6, 7), ExpectedResult(6,7);
    double helper1[7][8] = {{1, 2, 3, 4, 5, 6, 7, 8},
                            {9, 10, 11, 12, 13, 14, 15, 16},
                            {17, 18, 19, 20, 21, 22, 23, 24},
                            {25, 26, 27, 28, 29, 30, 31, 32},
                            {33, 34, 35, 36, 37, 38, 39, 40},
                            {41, 42, 43, 44, 45, 46, 47, 48},
                            {49, 50, 51, 52, 53, 54, 55, 56}};
    double helper2[2][2] = {{1.1, 2.2},
                            {3.3, 4.4}};
    double helper3[6][7] = {{41.8, 52.8, 63.8, 74.8, 85.8, 96.8, 107.8},
                            {129.8, 140.8, 151.8, 162.8, 173.8, 184.8, 195.8},
                            {217.8, 228.8, 239.8, 250.8, 261.8, 272.8, 283.8},
                            {305.8, 316.8, 327.8, 338.8, 349.8, 360.8, 371.8},
                            {393.8, 404.8, 415.8, 426.8, 437.8, 448.8, 459.8},
                            {481.8, 492.8, 503.8, 514.8, 525.8, 536.8, 547.8}};
    for (int row=0; row<7; row++)
    {
        for (int col=0; col < 8; col++)
        {
            Input.data[row][col] = helper1[row][col];
        }
    }
    for(int row=0; row<2; row++)
    {
        for(int col=0; col<2; col++)
        {
            Kernel.data[row][col] = helper2[row][col];
        }
    }
    for(int row=0; row<6; row++)
    {
        for(int col=0; col<7; col++)
        {
            ExpectedResult.data[row][col] = helper3[row][col];
        }
    }
    convolution(Input, Kernel, Result, 1, 1);
    ASSERT_EQ(true, test_matrixes_for_equality(Result, ExpectedResult));
}

void test_convolution_positive1_with_strides()
{
    Matrix Input(7,8), Kernel(2,2), Result(3, 4), ExpectedResult(3,4);
    double helper1[7][8] = {{1, 2, 3, 4, 5, 6, 7, 8},
                            {9, 10, 11, 12, 13, 14, 15, 16},
                            {17, 18, 19, 20, 21, 22, 23, 24},
                            {25, 26, 27, 28, 29, 30, 31, 32},
                            {33, 34, 35, 36, 37, 38, 39, 40},
                            {41, 42, 43, 44, 45, 46, 47, 48},
                            {49, 50, 51, 52, 53, 54, 55, 56}};
    double helper2[2][2] = {{1.1, 2.2},
                            {3.3, 4.4}};
    double helper3[6][7] = {{41.8, 63.8, 85.8, 107.8},
                            {217.8, 239.8, 261.8, 283.8},
                            {393.8, 415.8, 437.8, 459.8}};
    for (int row=0; row<7; row++)
    {
        for (int col=0; col < 8; col++)
        {
            Input.data[row][col] = helper1[row][col];
        }
    }
    for(int row=0; row<2; row++)
    {
        for(int col=0; col<2; col++)
        {
            Kernel.data[row][col] = helper2[row][col];
        }
    }
    for(int row=0; row<3; row++)
    {
        for(int col=0; col<4; col++)
        {
            ExpectedResult.data[row][col] = helper3[row][col];
        }
    }
    convolution(Input, Kernel, Result, 2, 2);
    ASSERT_EQ(true, test_matrixes_for_equality(Result, ExpectedResult));
}

void test_cross_correlation_positive1()
{
    Matrix Input(7,8), Kernel(2,2), Result(6, 7), ExpectedResult(6,7);
    double helper1[7][8] = {{1, 2, 4, 8, 16, 32, 64, 128},
                            {3, 6, 9, 12, 18, 21, 24, 27},
                            {2, 3, 5, 7, 11, 13, 17, 19},
                            {256, 512, 1024, 2048, 4096, 8192, 16384, 32768},
                            {30, 33, 36, 39, 42, 45, 48, 51},
                            {23, 29, 31, 37, 41, 47, 53, 59},
                            {49, 50, 51, 52, 53, 54, 55, 56}};
    double helper2[2][2] = {{1.1, 2.2},
                            {3.3, 4.4}};
    double helper3[6][7] = {{41.8, 70.4, 104.5, 162.8, 239.8, 350.9, 550},
                            {36.3, 58.3, 83.6, 124.3, 159.5, 193.6, 225.5},
                            {3106.4, 6209.5, 12411.3, 24812.7, 49602.3, 99174.9, 198307},
                            {1652.2, 3083.3, 5922.4, 11577.5, 22864.6, 45415.7, 90494.8},
                            {309.1, 347.6, 390.5, 437.8, 487.3, 543.4, 599.5},
                            {470.8, 489.5, 512.6, 535.7, 561, 588.5, 616}};
    for (int row=0; row<7; row++)
    {
        for (int col=0; col < 8; col++)
        {
            Input.data[row][col] = helper1[row][col];
        }
    }
    for(int row=0; row<2; row++)
    {
        for(int col=0; col<2; col++)
        {
            Kernel.data[row][col] = helper2[row][col];
        }
    }
    for(int row=0; row<6; row++)
    {
        for(int col=0; col<7; col++)
        {
            ExpectedResult.data[row][col] = helper3[row][col];
        }
    }
    cross_correlation(Input, Kernel, Result, 1, 1);
    ASSERT_EQ(true, test_matrixes_for_equality(Result, ExpectedResult));
}

void test_cross_correlation_positive1_with_strides()
{
    Matrix Input(7,8), Kernel(2,2), Result(3, 4), ExpectedResult(3,4);
    double helper1[7][8] = {{1, 2, 4, 8, 16, 32, 64, 128},
                            {3, 6, 9, 12, 18, 21, 24, 27},
                            {2, 3, 5, 7, 11, 13, 17, 19},
                            {256, 512, 1024, 2048, 4096, 8192, 16384, 32768},
                            {30, 33, 36, 39, 42, 45, 48, 51},
                            {23, 29, 31, 37, 41, 47, 53, 59},
                            {49, 50, 51, 52, 53, 54, 55, 56}};
    double helper2[2][2] = {{1.1, 2.2},
                            {3.3, 4.4}};
    double helper3[3][4] = {{41.8, 104.5, 239.8, 550},
                            {3106.4, 12411.3, 49602.3, 198307},
                            {309.1, 390.5, 487.3, 599.5}};
    for (int row=0; row<7; row++)
    {
        for (int col=0; col < 8; col++)
        {
            Input.data[row][col] = helper1[row][col];
        }
    }
    for(int row=0; row<2; row++)
    {
        for(int col=0; col<2; col++)
        {
            Kernel.data[row][col] = helper2[row][col];
        }
    }
    for(int row=0; row<3; row++)
    {
        for(int col=0; col<4; col++)
        {
            ExpectedResult.data[row][col] = helper3[row][col];
        }
    }
    cross_correlation(Input, Kernel, Result, 2, 2);
    ASSERT_EQ(true, test_matrixes_for_equality(Result, ExpectedResult));
}


TEST(MatrixTest, BasicOperationsPass)
{
    test_mtx_to_mtx_multyply_pass();
    test_mtx_to_double_multyply_pass();
    test_divide_matrixes_pass();
    test_mtx_to_mtx_addition_pass();
    test_mtx_to_double_addition_pass();
    test_mtx_plus_equal_mtx_pass();
    test_mtx_plus_equal_double_pass();
    test_hadamart_pass();
    test_transpose_pass();
    test_convolution_positive1();
    //test_cross_correlation_positive1_with_strides();
    //test_convolution_positive1_with_strides();
}


int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
    /*Matrix Input(7,8), Kernel(2,2), Result(6, 7), ExpectedResult(6,7);
    double helper1[7][8] = {{1, 2, 4, 8, 16, 32, 64, 128},
                            {3, 6, 9, 12, 18, 21, 24, 27},
                            {2, 3, 5, 7, 11, 13, 17, 19},
                            {256, 512, 1024, 2048, 4096, 8192, 16384, 32768},
                            {30, 33, 36, 39, 42, 45, 48, 51},
                            {23, 29, 31, 37, 41, 47, 53, 59},
                            {49, 50, 51, 52, 53, 54, 55, 56}};
    double helper2[2][2] = {{1.1, 2.2},
                            {3.3, 4.4}};
    double helper3[6][7] = {{41.8, 70.4, 104.5, 162.8, 239.8, 350.9, 550},
                            {36.3, 58.3, 83.6, 124.3, 159.5, 193.6, 225.5},
                            {3106.4, 6209.5, 12411.3, 24812.7, 49602.3, 99174.9, 198307},
                            {1652.2, 3083.3, 5922.4, 11577.5, 22864.6, 45415.7, 90494.8},
                            {309.1, 347.6, 390.5, 437.8, 487.3, 543.4, 599.5},
                            {470.8, 489.5, 512.6, 535.7, 561, 588.5, 616}};
    for (int row=0; row<7; row++)
    {
        for (int col=0; col < 8; col++)
        {
            Input.data[row][col] = helper1[row][col];
        }
    }
    for(int row=0; row<2; row++)
    {
        for(int col=0; col<2; col++)
        {
            Kernel.data[row][col] = helper2[row][col];
        }
    }
    for(int row=0; row<6; row++)
    {
        for(int col=0; col<7; col++)
        {
            ExpectedResult.data[row][col] = helper3[row][col];
        }
    }
    cross_correlation(Input, Kernel, Result, 2, 2);
    print_mtx(Result);*/
}
