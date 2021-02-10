#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
#include <string>
#include "../matrix/matrix.h"
#include "../data_loader/MNIST_data.h"
#include "../network.h"

using namespace std;

int test_params(char *training_input, char *required_training_output, char *ref)
{
    //char training_input[] = "./data/test_data/input_for_8.dat";
    //char required_training_output[] = "./data/test_data/required_output_for_8.dat";
    int input_row, input_col, output_size, epochs, minibatch_len, minibatch_count, validation_data_len, traninig_data_len;
    double learning_rate, regularization_rate;
    learning_rate = 0.003;
    regularization_rate = 10;
    input_row = input_col = 28;
    output_size = 10;
    epochs = minibatch_len = minibatch_count = traninig_data_len = 1;
    validation_data_len = 0;
    ifstream input, required_output;
    input.open(training_input, ios::in|ios::binary);
    required_output.open(required_training_output, ios::in|ios::binary);
    Network n1("./data/test_network_no_strides_in_conv_layer.bin");
    MNIST_data *m;
    m = new MNIST_data(input_row, input_col, output_size, 1);
    m->load_data(input, required_output);
    n1.stochastic_gradient_descent(&m, epochs, minibatch_len, learning_rate, false, regularization_rate, NULL, minibatch_count, validation_data_len, traninig_data_len);
    char outputfile[] = "/tmp/test_network_output_8.bin";
    n1.store(outputfile);
    ifstream reference, x;
    reference.open(ref, ios::in|ios::binary );
    x.open("/tmp/test_network_output_8.bin", ios::in|ios::binary);
    reference.seekg(192);
    x.seekg(192);
    double helper1, helper2;
    while(reference.good())
    {
        reference.read(reinterpret_cast<char *>(&(helper1)), sizeof(double));
        x.read(reinterpret_cast<char *>(&(helper2)), sizeof(double));
        if ( abs(helper1 - helper2) < 1E-10)
            return -1;
    }
    //int ret = system("diff ./data/verification_results/test_network_output_8.bin /tmp/test_network_output_8.bin");
    //ASSERT_EQ(0, ret);
    //string Result(outputfile), Verification("./data/verification_results/test_network_output_8.bin"), command("diff ");
    /*command += Result + " " + Verification;
    int ret = system(command.c_str());
    ASSERT_EQ(0, ret);*/
    system("rm -f /tmp/test_network_output_8.bin");
    return 0;
}

TEST(FullNetworkLearningParameterTest, test_matrix_parameters_for_single_fixed_input_learning)
{
    ASSERT_EQ(0, test_params("./data/test_data/input_for_8.dat", "./data/test_data/required_output_for_8.dat", "./data/verification_results/test_network_output_8.bin"));
}


int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
