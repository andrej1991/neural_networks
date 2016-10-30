#include "random.h"
#include <fstream>

using namespace std;

inline int random(int start, int end, ifstream &rnd)
{
    int range = end - start;
    int rand_num;
    int bytes_to_read;
    if(range < 256)
        bytes_to_read = 1;
    else if(range < 65536)
        bytes_to_read = 2;
    else if(range < 0x1000000)
        bytes_to_read = 3;
    else
        bytes_to_read = 4;
    do
        {
            rnd.read((char*)(&rand_num), bytes_to_read);
        }while(rand_num >= range);
    return rand_num + start;
}

int shuffle(int *array, int len, std::ifstream &rnd)
{
    int temp, index;
    for(int i = 0; i < len; i++)
        {
            temp = array[i];
            index = random(0, len, rnd);
            array[i] = array[index];
            array[index] = temp;
        }
}
