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
    virtual inline void set_params_for_deltahelper(int neuroncount, class Feature_map **next_l_fmap, Matrice **padded_delta);
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
    virtual inline void set_params_for_deltahelper(int neuroncount, class Feature_map **next_l_fmap, Matrice **padded_delta);
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
    virtual inline void set_params_for_deltahelper(int neuroncount, class Feature_map **next_l_fmap, Matrice **padded_delta);
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
    template<typename T>
    void push_many(JobInterface**, int, Matrice**);
    void push_many(JobInterface**, int, Matrice**, Feature_map**);
    template<typename T>
    void push_many(JobInterface**, int,int neuroncount, class Feature_map **next_l_fmap, Matrice **padded_delta);
    void wait();
    static void run(ThreadPool *obj);
};

void JobInterface::set_params_for_deltahelper(int neuroncount, class Feature_map **next_l_fmap, Matrice **padded_delta)
{
    cerr << "this function is not implemented\n";
    throw exception();
}

void CalculatingDeltaHelperNonConv::set_params_for_deltahelper(int neuroncount, Feature_map **next_l_fmap, Matrice **pdelta)
{
    this->next_layers_neuroncount = neuroncount;
    this->next_layers_fmaps = next_l_fmap;
    this->padded_delta = pdelta;
}

void CalculatingDeltaHelperConv::set_params_for_deltahelper(int neuroncount, Feature_map **next_l_fmap, Matrice **pdelta)
{
    this->next_layers_fmapcount = neuroncount;
    this->next_layers_fmaps = next_l_fmap;
    this->padded_delta = pdelta;
}

template<typename T>
void ThreadPool::push_many(JobInterface** jobs, int num, Matrice **input)
{
    pthread_mutex_lock(&(this->jobqueue_lock));
    pthread_mutex_lock(&(this->remaining_work_lock));
    for(int i = 0; i < num; i++)
    {
        dynamic_cast<T*>(jobs[i])->input = input;
        this->jobqueue.push(jobs[i]);
    }
    this->queue_len += num;
    this->remaining_work += num;
    pthread_mutex_unlock(&(this->jobqueue_lock));
    pthread_mutex_unlock(&(this->remaining_work_lock));
    this->block_thread.notify_all();
}


template<typename T>
void ThreadPool::push_many(JobInterface** jobs, int num, int neuroncount, class Feature_map **next_l_fmap, Matrice **padded_delta)
{
    pthread_mutex_lock(&(this->jobqueue_lock));
    pthread_mutex_lock(&(this->remaining_work_lock));
    for(int i = 0; i < num; i++)
    {
        dynamic_cast<T*>(jobs[i])->set_params_for_deltahelper(neuroncount, next_l_fmap, padded_delta);
        this->jobqueue.push(jobs[i]);
    }
    this->queue_len += num;
    this->remaining_work += num;
    pthread_mutex_unlock(&(this->jobqueue_lock));
    pthread_mutex_unlock(&(this->remaining_work_lock));
    this->block_thread.notify_all();
}


#endif // THREADPOOL_H_INCLUDED
