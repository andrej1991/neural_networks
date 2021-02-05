object_dir = ../../obj/unittest
binary_dir = ../../bin/unittest
object_files = $(object_dir)/matrix.o $(object_dir)/MNIST_data.o $(object_dir)/exceptions.o $(object_dir)/Convolutional.o $(object_dir)/Feature_map.o $(object_dir)/FullyConnected.o $(object_dir)/Inputlayer.o $(object_dir)/layerdescriptor.o $(object_dir)/layers.o $(object_dir)/Pooling.o $(object_dir)/Softmax.o $(object_dir)/neuron.o $(object_dir)/unittests.o

prepare:
	mkdir -p $(binary_dir)
	mkdir -p $(object_dir)

objects: prepare
	g++ -c ../matrix/matrix.cpp -o $(object_dir)/matrix.o
	g++ -c ../data_loader/MNIST_data.cpp -o $(object_dir)/MNIST_data.o
	g++ -c ../exceptions/exceptions.cpp -o $(object_dir)/exceptions.o
	g++ -c ../layers/Convolutional.cpp -o $(object_dir)/Convolutional.o
	g++ -c ../layers/Feature_map.cpp -o $(object_dir)/Feature_map.o
	g++ -c ../layers/FullyConnected.cpp -o $(object_dir)/FullyConnected.o
	g++ -c ../layers/Inputlayer.cpp -o $(object_dir)/Inputlayer.o
	g++ -c ../layers/layerdescriptor.cpp -o $(object_dir)/layerdescriptor.o
	g++ -c ../layers/layers.cpp -o $(object_dir)/layers.o
	g++ -c ../layers/Pooling.cpp -o $(object_dir)/Pooling.o
	g++ -c ../layers/Softmax.cpp -o $(object_dir)/Softmax.o
	g++ -c ../neurons/neuron.cpp -o $(object_dir)/neuron.o
	g++ -c unittests.cpp -o $(object_dir)/unittests.o

all: objects
	g++ $(object_files) -O3 -lgtest -lpthread -o $(binary_dir)/neural_net_unit_tests
	
clean:
	rm -f $(object_dir)/*.o
	rm -f $(binary_dir)/*
	
