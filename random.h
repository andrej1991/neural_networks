#ifndef RANDOM_H_INCLUDED
#define RANDOM_H_INCLUDED
#include <fstream>

inline int random(int start, int end, std::ifstream &rnd);
int shuffle(int *array, int len, std::ifstream &rnd);

#endif // RANDOM_H_INCLUDED
