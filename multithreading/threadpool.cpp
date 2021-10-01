#include "threadpool.h"

using namespace std;

Job::Job(int id, Network *neunet):id(id)
{
    this->neunet = neunet;
    this->nabla_lock = PTHREAD_MUTEX_INITIALIZER;
}

void Job::work(int threadid)
{
    this->neunet->backpropagate(this->training_data, this->deltanabla, this->costfunction_type);
    pthread_mutex_lock(&(this->nabla_lock));
    for(int layer_index = 0; layer_index < this->neunet->layers_num; layer_index++)
    {
        this->nabla[layer_index][0] += this->deltanabla[layer_index][0];
    }
    pthread_mutex_lock(&(this->nabla_lock));
}


ThreadPool::ThreadPool(int threadcount): thread_count(threadcount), queue_len(0), remaining_work(0)
{
    this->jobqueue_lock = PTHREAD_MUTEX_INITIALIZER;
    this->remaining_work_lock = PTHREAD_MUTEX_INITIALIZER;
    this->t = new thread[threadcount];
    for(int i = 0; i < threadcount; i++)
    {
        this->t[i] = std::thread(&(ThreadPool::run), this, i);
    }
}

ThreadPool::~ThreadPool()
{
    this->wait();
    pthread_mutex_lock(&(this->jobqueue_lock));
    this->queue_len = -1;
    pthread_mutex_unlock(&(this->jobqueue_lock));
    for(int i = 0; i < this->thread_count; i++)
    {
        this->t[i].join();
    }
}

void ThreadPool::wait()
{
    while(this->remaining_work > 0);
}

void ThreadPool::push(Job* new_job)
{
    pthread_mutex_lock(&(this->jobqueue_lock));
    pthread_mutex_lock(&(this->remaining_work_lock));
    this->jobqueue.push(new_job);
    this->queue_len++;
    this->remaining_work++;
    pthread_mutex_unlock(&(this->jobqueue_lock));
    pthread_mutex_unlock(&(this->remaining_work_lock));
}

void ThreadPool::run(ThreadPool *obj, int i)
{
    Job *myjob = NULL;
    while(obj)
    {
        pthread_mutex_lock(&(obj->jobqueue_lock));
        if(obj->queue_len > 0)
        {
            myjob = obj->jobqueue.front();
            obj->jobqueue.pop();
            obj->queue_len -= 1;
            pthread_mutex_unlock(&(obj->jobqueue_lock));
            myjob->work(i);
            myjob = NULL;
            pthread_mutex_lock(&(obj->remaining_work_lock));
            obj->remaining_work--;
            pthread_mutex_unlock(&(obj->remaining_work_lock));
        }
        else if (obj->queue_len == 0)
        {
            pthread_mutex_unlock(&(obj->jobqueue_lock));
        }
        else
        {
            pthread_mutex_unlock(&(obj->jobqueue_lock));
            break;
        }
    }
}
