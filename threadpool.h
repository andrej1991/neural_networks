#ifndef THREADPOOL_H_INCLUDED
#define THREADPOOL_H_INCLUDED

#include <iostream>
#include <queue>
#include <thread>
#include <pthread.h>
#include <mutex>
#include <condition_variable>
#include "matrice.h"
//#include "layers/layers.h"

class JobInterface
{
    public:
    virtual void work() = 0;
    virtual ~JobInterface(){};
    virtual void set_params_for_deltahelper(int neuroncount, class Feature_map **next_l_fmap, Matrice **padded_delta);
};

class CalculatingNabla: public JobInterface
{
    int id;
    class Convolutional *convlay;
    public:
    Matrice **input;
    class Feature_map** nabla;
    public:
    CalculatingNabla(int id, class Convolutional *convlay);
    virtual void work();
};

class CalculatingDeltaHelperConv: public JobInterface
{
    int id;
    class Convolutional *convlay;
    Matrice helper, kernel;
    public:
    int next_layers_fmapcount;
    class Feature_map** next_layers_fmaps;
    Matrice **padded_delta;
    public:
    CalculatingDeltaHelperConv(int id, class Convolutional *convlay);
    virtual void work();
    virtual void set_params_for_deltahelper(int neuroncount, class Feature_map **next_l_fmap, Matrice **padded_delta);
};

class CalculatingDeltaHelperNonConv: public JobInterface
{
    int id;
    class Convolutional *convlay;
    Matrice helper, kernel;
    public:
    int next_layers_neuroncount;
    class Feature_map** next_layers_fmaps;
    Matrice **padded_delta;
    public:
    CalculatingDeltaHelperNonConv(int id, class Convolutional *convlay);
    virtual void work();
    virtual void set_params_for_deltahelper(int neuroncount, class Feature_map **next_l_fmap, Matrice **padded_delta);
};

/*class GetPaddedDeltaConv: public JobInterface
{
    int id, top, right, bottom, left;
    class Convolutional *convlay;
    Matrice **padded_delta, **delta;
    public:
    GetPaddedDeltaConv(int id, int top, int right, int bottom, int left, class Convolutional *convlay, Matrice **padded_delta, Matrice **delta);
    virtual void work();
};

class GetPaddedDeltaNonConv: public JobInterface
{
    int id, top, right, bottom, left;
    class Convolutional *convlay;
    Matrice **padded_delta, **delta;
    public:
    GetPaddedDeltaNonConv(int id, int top, int right, int bottom, int left, class Convolutional *convlay, Matrice **padded_delta, Matrice **delta);
    virtual void work();
};*/

class GetOutputJob: public JobInterface
{
    int id;
    class Convolutional *convlay;
    Matrice helper, convolved;
    public:
    Matrice **input;
    GetOutputJob(int id, class Convolutional *convlay);
    virtual void work();
};

class GetOutputDerivativeJob: public JobInterface
{
    int id;
    class Convolutional *convlay;
    Matrice helper, convolved;
    public:
    Matrice **input;
    GetOutputDerivativeJob(int id, class Convolutional *convlay);
    virtual void set_input(Matrice **input);
    virtual void work();
};


class ThreadPool
{
    pthread_mutex_t jobqueue_lock, remaining_work_lock;
    std::queue<class JobInterface*> jobqueue;
    int queue_len, thread_count;
    volatile int remaining_work;
    std::thread *t;
    std::mutex block_thread_lock;
    std::condition_variable block_thread;
public:
    ThreadPool(int threadcount);
    ~ThreadPool();
    void push(JobInterface*);
    void wait();
    static void run(ThreadPool *obj);
};



#endif // THREADPOOL_H_INCLUDED
