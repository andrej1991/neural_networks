#ifndef THREADPOOL_H_INCLUDED
#define THREADPOOL_H_INCLUDED

#include <iostream>
#include <queue>
#include <thread>
#include <pthread.h>
#include "../network.h"

class Job
{
    int id;
    Network *neunet;
    pthread_mutex_t nabla_lock;
    public:
    Job(int id, Network *n);
    virtual void work(int i);
    MNIST_data *training_data;
    Layers_features **deltanabla, **nabla;
    int costfunction_type;
};

class ThreadPool
{
    pthread_mutex_t jobqueue_lock, remaining_work_lock;
    std::queue<Job*> jobqueue;
    int queue_len, thread_count;
    volatile int remaining_work;
    std::thread *t;
public:
    ThreadPool(int threadcount);
    ~ThreadPool();
    void push(Job*);
    void wait();
    static void run(ThreadPool *obj,int i);
};

#endif // THREADPOOL_H_INCLUDED
