#include "threadpool.h"
#include "layers/layers.h"

using namespace std;


CalculatingNabla::CalculatingNabla(int id, class Convolutional *convlay, Matrice **input, class Feature_map** nabla):
                           id(id), convlay(convlay), input(input), nabla(nabla){};


void CalculatingNabla::work()
{
    convlay->layers_delta[id][0] = hadamart_product(convlay->layers_delta_helper[id][0], convlay->output_derivative[id][0]);
    for(int j = 0; j < convlay->fmap[id]->get_mapdepth(); j++)
    {
         convolution(input[j][0], convlay->layers_delta[id][0], nabla[id]->weights[j][0]);
    }
}

CalculatingDeltaHelperNonConv::CalculatingDeltaHelperNonConv(int id, int next_layers_neuroncount, Convolutional *convlay, Feature_map** next_layers_fmaps, Matrice **padded_delta):
                           id(id), next_layers_neuroncount(next_layers_neuroncount), convlay(convlay), next_layers_fmaps(next_layers_fmaps), padded_delta(padded_delta){};


void CalculatingDeltaHelperNonConv::work()
{
    Matrice kernel(convlay->output_row, convlay->output_col);
    Matrice helper(convlay->output_row, convlay->output_col);
    convlay->layers_delta_helper[id][0].zero();
    for(int j = 0; j < next_layers_neuroncount; j++)
    {
        convlay->get_2D_weights(j, id, kernel, next_layers_fmaps);
        convolution(padded_delta[j][0], kernel, helper);
        convlay->layers_delta_helper[id][0] += helper;
    }
}

CalculatingDeltaHelperConv::CalculatingDeltaHelperConv(int id, int next_layers_fmapcount, Convolutional *convlay, Feature_map** next_layers_fmaps, Matrice **padded_delta):
                           id(id), next_layers_fmapcount(next_layers_fmapcount), convlay(convlay), next_layers_fmaps(next_layers_fmaps), padded_delta(padded_delta){};


void CalculatingDeltaHelperConv::work()
{
    Matrice kernel(convlay->output_row, convlay->output_col);
    Matrice helper(convlay->output_row, convlay->output_col);
    convlay->layers_delta_helper[id][0].zero();
    for(int j = 0; j < next_layers_fmapcount; j++)
    {
        convolution(padded_delta[j][0], kernel, helper);
        convlay->layers_delta_helper[id][0] += helper;
    }
}


ThreadPool::ThreadPool(int threadcount): thread_count(threadcount), queue_len(0), remaining_work(0)
{
    this->jobqueue_lock = PTHREAD_MUTEX_INITIALIZER;
    this->remaining_work_lock = PTHREAD_MUTEX_INITIALIZER;
    this->t = new thread[threadcount];
    for(int i = 0; i < threadcount; i++)
    {
        this->t[i] = std::thread(&(ThreadPool::run), this);
    }
}

ThreadPool::~ThreadPool()
{
    this->wait();
    pthread_mutex_lock(&(this->jobqueue_lock));
    this->queue_len = -1;
    pthread_mutex_unlock(&(this->jobqueue_lock));
    this->block_thread.notify_all();
    for(int i = 0; i < this->thread_count; i++)
    {
        this->t[i].join();
    }
}

void ThreadPool::wait()
{
    while(this->remaining_work > 0)
    {
        ;
    }
}

void ThreadPool::push(JobInterface* new_job)
{
    pthread_mutex_lock(&(this->jobqueue_lock));
    pthread_mutex_lock(&(this->remaining_work_lock));
    this->jobqueue.push(new_job);
    this->queue_len++;
    this->remaining_work++;
    pthread_mutex_unlock(&(this->jobqueue_lock));
    pthread_mutex_unlock(&(this->remaining_work_lock));
    this->block_thread.notify_all();
}

void ThreadPool::run(ThreadPool *obj)
{
    JobInterface *myjob = NULL;
    std::unique_lock<std::mutex> execution_blocker(obj->block_thread_lock);
    while(obj)
    {
        obj->block_thread.wait(execution_blocker);
        pthread_mutex_lock(&(obj->jobqueue_lock));
        if(obj->queue_len > 0)
        {
            myjob = obj->jobqueue.front();
            obj->jobqueue.pop();
            obj->queue_len -= 1;
            pthread_mutex_unlock(&(obj->jobqueue_lock));
            myjob->work();
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
