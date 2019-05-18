#include "threadpool.h"
#include "layers/layers.h"

using namespace std;

CalculatingNabla::CalculatingNabla(int id, class Convolutional *convlay):
                           id(id), convlay(convlay), input(input), nabla(nabla){};


void CalculatingNabla::work()
{
    convlay->layers_delta[id][0] = hadamart_product(convlay->layers_delta_helper[id][0], convlay->output_derivative[id][0]);
    for(int j = 0; j < convlay->fmap[id]->get_mapdepth(); j++)
    {
         convolution(input[j][0], convlay->layers_delta[id][0], nabla[id]->weights[j][0]);
    }
}

CalculatingDeltaHelperNonConv::CalculatingDeltaHelperNonConv(int id, Convolutional *convlay):
                           id(id), next_layers_neuroncount(0), convlay(convlay), next_layers_fmaps(NULL), padded_delta(NULL),
                           kernel(convlay->output_row, convlay->output_col), helper(convlay->output_row, convlay->output_col){};


void CalculatingDeltaHelperNonConv::work()
{
    convlay->layers_delta_helper[id][0].zero();
    helper.zero();
    for(int j = 0; j < next_layers_neuroncount; j++)
    {
        convlay->get_2D_weights(j, id, kernel, next_layers_fmaps);
        convolution(padded_delta[j][0], kernel, helper);
        convlay->layers_delta_helper[id][0] += helper;
    }
}


CalculatingDeltaHelperConv::CalculatingDeltaHelperConv(int id, Convolutional *convlay):
                           id(id), next_layers_fmapcount(0), convlay(convlay), next_layers_fmaps(NULL), padded_delta(NULL),
                           kernel(convlay->output_row, convlay->output_col), helper(convlay->output_row, convlay->output_col){};


void CalculatingDeltaHelperConv::work()
{
    convlay->layers_delta_helper[id][0].zero();
    for(int j = 0; j < next_layers_fmapcount; j++)
    {
        convolution(padded_delta[j][0], kernel, helper);
        convlay->layers_delta_helper[id][0] += helper;
    }
}


/*GetPaddedDeltaConv::GetPaddedDeltaConv(int id, int top, int right, int bottom, int left, Convolutional *convlay, Matrice **padded_delta, Matrice **delta):
                           id(id), top(top), right(right), bottom(bottom), left(left), convlay(convlay), padded_delta(padded_delta), delta(delta){};


void GetPaddedDeltaConv::work()
{
    padded_delta[id][0] = delta[0][0];
    padded_delta[id][0] = padded_delta[id][0].zero_padd(top, right, bottom, left);
}

GetPaddedDeltaNonConv::GetPaddedDeltaNonConv(int id, int top, int right, int bottom, int left, Convolutional *convlay, Matrice **padded_delta, Matrice **delta):
                           id(id), top(top), right(right), bottom(bottom), left(left), convlay(convlay), padded_delta(padded_delta), delta(delta){};


void GetPaddedDeltaNonConv::work()
{
    padded_delta[id][0].data[0][0] = delta[0][0].data[id][0];
    padded_delta[id][0] = padded_delta[id][0].zero_padd(top, right, bottom, left);
}*/

GetOutputJob::GetOutputJob(int id, Convolutional *convlay):
                           id(id), convlay(convlay), input(NULL), convolved(convlay->output_row, convlay->output_col), helper(convlay->output_row, convlay->output_col){};


void GetOutputJob::work()
{
    helper.zero();
    for(int channel_index = 0; channel_index < convlay->fmap[id]->get_mapdepth(); channel_index++)
    {
        convolution(input[channel_index][0], convlay->fmap[id]->weights[channel_index][0], convolved, convlay->stride);
        helper += convolved;
    }
    helper+=convlay->fmap[id]->biases[0][0].data[0][0];
    convlay->neuron.neuron(helper, convlay->outputs[id][0]);
}


GetOutputDerivativeJob::GetOutputDerivativeJob(int id, Convolutional *convlay):
                           id(id), convlay(convlay), input(NULL), convolved(convlay->output_row, convlay->output_col), helper(convlay->output_row, convlay->output_col){};

void GetOutputDerivativeJob::work()
{
    helper.zero();
    for(int channel_index = 0; channel_index < convlay->fmap[id]->get_mapdepth(); channel_index++)
    {
        convolution(input[channel_index][0], convlay->fmap[id]->weights[channel_index][0], convolved, convlay->stride);
        helper += convolved;
    }
    helper+=convlay->fmap[id]->biases[0][0].data[0][0];
    convlay->neuron.neuron_derivative(helper, convlay->outputs[id][0]);
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
    this->block_thread.notify_one();
}

void ThreadPool::push_many(JobInterface** jobs, int num, Matrice** input, Feature_map** nabla)
{
    pthread_mutex_lock(&(this->jobqueue_lock));
    pthread_mutex_lock(&(this->remaining_work_lock));
    for(int i = 0; i < num; i++)
    {
        dynamic_cast<CalculatingNabla*>(jobs[i])->input = input;
        dynamic_cast<CalculatingNabla*>(jobs[i])->nabla = nabla;
        this->jobqueue.push(jobs[i]);
    }
    this->queue_len += num;
    this->remaining_work += num;
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
            obj->block_thread.wait(execution_blocker);
        }
        else
        {
            pthread_mutex_unlock(&(obj->jobqueue_lock));
            break;
        }
    }
}
