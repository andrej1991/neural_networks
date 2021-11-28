#include "game.h"
#include "../additional.h"
#include <unistd.h>

#define input_col 50
#define input_row 50

void print_action(Matrix &action)
{
    int dir = getmax(action.data, 4);
    switch(dir)
    {
        case LEFT:
            cout << "LEFT  ";
            break;
        case UP:
            cout << "UP  ";
            break;
        case RIGHT:
            cout << "RIGHT  ";
            break;
        case DOWN:
            cout << "DOWN  ";
            break;
    }
    cout << action.data[0][0] << "  " << action.data[1][0] << "  " << action.data[2][0] << "  " << action.data[3][0] << "  " << endl;
}

bool food_is_infront(Game &g)
{
    Colider snake_colider, food_colider;
    snake_colider = g.snake->get_colider()[0];
    food_colider = g.food->get_colider();
    if(snake_colider.xpos == food_colider.xpos)
    {
        if(snake_colider.ypos < food_colider.ypos && g.snake->get_direction() == UP)
        {
            cout << "front1\n";
            return true;
        }
        if(snake_colider.ypos > food_colider.ypos && g.snake->get_direction() == DOWN)
        {
            cout << "front2\n";
            return true;
        }
    }
    if(snake_colider.ypos == food_colider.ypos)
    {
        if(snake_colider.xpos < food_colider.xpos && g.snake->get_direction() == RIGHT)
        {
            cout << "front3\n";
            return true;
        }
        if(snake_colider.xpos > food_colider.xpos && g.snake->get_direction() == LEFT)
        {
            cout << "front4\n";
            return true;
        }
    }
    return false;
}

bool food_is_behind(Game &g)
{
    Colider snake_colider, food_colider;
    snake_colider = g.snake->get_colider()[0];
    food_colider = g.food->get_colider();
    if(snake_colider.xpos == food_colider.xpos)
    {
        if(snake_colider.ypos < food_colider.ypos && g.snake->get_direction() == DOWN)
        {
            cout << "behind1\n";
            return true;
        }
        if(snake_colider.ypos > food_colider.ypos && g.snake->get_direction() == UP)
        {
            cout << "behind2\n";
            return true;
        }
    }
    if(snake_colider.ypos == food_colider.ypos)
    {
        if(snake_colider.xpos < food_colider.xpos && g.snake->get_direction() == LEFT)
        {
            cout << "behind3\n";
            return true;
        }
        if(snake_colider.xpos > food_colider.xpos && g.snake->get_direction() == RIGHT)
        {
            cout << "behind4\n";
            return true;
        }
    }
    return false;
}

bool food_is_intheleft(Game &g)
{
    int direction = g.snake->get_direction();
    Colider snake_colider, food_colider;
    snake_colider = g.snake->get_colider()[0];
    food_colider = g.food->get_colider();
    if(direction == UP || direction == DOWN)
    {
        if(snake_colider.xpos > food_colider.xpos)
        {
            cout << "left\n";
            return true;
        }
    }
    return false;
}

bool food_is_intheright(Game &g)
{
    int direction = g.snake->get_direction();
    Colider snake_colider, food_colider;
    snake_colider = g.snake->get_colider()[0];
    food_colider = g.food->get_colider();
    if(direction == UP || direction == DOWN)
    {
        if(snake_colider.xpos < food_colider.xpos)
        {
            cout << "right\n";
            return true;
        }
    }
    return false;
}

bool food_is_above(Game &g)
{
    int direction = g.snake->get_direction();
    Colider snake_colider, food_colider;
    snake_colider = g.snake->get_colider()[0];
    food_colider = g.food->get_colider();
    if(direction == LEFT || direction == RIGHT)
    {
        /**the top left corner is the 0,0*/
        if(snake_colider.ypos > food_colider.ypos)
        {
            cout << "abowe\n";
            return true;
        }
    }
    return false;
}

bool food_is_below(Game &g)
{
    int direction = g.snake->get_direction();
    Colider snake_colider, food_colider;
    snake_colider = g.snake->get_colider()[0];
    food_colider = g.food->get_colider();
    if(direction == LEFT || direction == RIGHT)
    {
        /**the top left corner is the 0,0*/
        if(snake_colider.ypos < food_colider.ypos)
        {
            cout << "below\n";
            return true;
        }
    }
    return false;
}

Matrix get_correct_way(Game &g)
{
    Matrix ret(4, 1);
    if(food_is_infront(g))
    {
        ret.data[g.snake->get_direction()][0] = 1;
        return ret;
    }
    if(food_is_behind(g))
    {
        return ret;
    }
    if(food_is_intheleft(g))
    {
        ret.data[LEFT][0] = 1;
        return ret;
    }
    if(food_is_intheright(g))
    {
        ret.data[RIGHT][0] = 1;
        return ret;
    }
    if(food_is_above(g))
    {
        ret.data[UP][0] = 1;
        return ret;
    }
    if(food_is_below(g))
    {
        ret.data[DOWN][0] = 1;
        return ret;
    }
    //return ret;
}

void reinforcement_snake(Network &net, StochasticGradientDescent &learn, double learning_rate, double regularization_rate)
{
    SDL_Event event;
    SDL_Surface *window_surface = SDL_CreateRGBSurface(0, input_col, input_row, 32, 0x00ff0000, 0x0000ff00, 0x000000ff, 0xff000000);
    bool gameover = false;
    Game g(input_row, input_col);
    g.drop_food();
    MNIST_data *d, *train_incaseof_fail;
    Matrix action;
    std::vector<MNIST_data*> training;
    std::vector<MNIST_data*> training_incaseof_fail;
    ifstream inp;
    int direction;
    double temp;
    while(!g.quit)
    {
        usleep(200000);
        while(SDL_PollEvent(&event) != 0)
        {
            if(event.type == SDL_QUIT)
            {
                g.quit = true;
            }
        }
        d = new MNIST_data(input_row, input_col, 4, 3);
        train_incaseof_fail = new MNIST_data(input_row, input_col, 4, 3);
        if(!gameover)
        {

            SDL_LockSurface(window_surface);
            SDL_RenderReadPixels(g.bckgrnd_renderer, NULL, window_surface->format->format, window_surface->pixels, window_surface->pitch);
            SDL_SaveBMP(window_surface, "../window_surface.bmp");
            SDL_UnlockSurface(window_surface);
            inp.open("../window_surface.bmp");
            d[0].load_bmp(inp);
            train_incaseof_fail[0].load_bmp(inp);
            inp.close();
            action = net.get_output(d[0].input);
            //print_action(action);
            d[0].required_output = action;
            train_incaseof_fail[0].required_output = get_correct_way(g);
            training_incaseof_fail.push_back(train_incaseof_fail);
            if(g.snake->handle_event(action))
            {
                training.push_back(d);
                //delete d;
                gameover = g.snake->detect_colission(g.walls, g.wallcount);
                gameover |= g.snake->detect_self_colission();
                if(g.snake->detect_colission(g.food))
                {
                    cout << "food found\n";
                    g.score++;
                    g.drop_food();
                    for(int i=0; i<training.size(); i++)
                    {
                        direction = getmax(training[i]->required_output.data, 4);
                        for(int j=0; j<training[i]->required_output.get_row(); j++)
                        {
                            training[i]->required_output.data[j][0] = 0;
                        }
                        training[i]->required_output.data[direction][0] = 1;
                    }
                    /*(MNIST_data **training_data, int epochs, int minibatch_len, double learning_rate, double momentum, bool monitor_learning_cost = false,
                       double regularization_rate = 0, MNIST_data **test_data = NULL, int minibatch_count = 500, int test_data_len = 10000,  int trainingdata_len = 50000);*/
                    learn.nesterov_accelerated_gradient(&training[0], 1, training.size(), learning_rate, 0.9, false, regularization_rate, NULL, 1, 0, training.size());
                    do
                    {
                        training.pop_back();
                        training_incaseof_fail.pop_back();
                    }while(training.size() > 0);
                }
            }
        }
        else
        {
            cout << "game over\n";
            g.score = 0;
            for(int i=0; i<training_incaseof_fail.size(); i++)
            {
                /*for(int j=0; j<training_incaseof_fail[i]->required_output.get_row(); j++)
                {
                    training_incaseof_fail[i]->required_output.data[j][0] = 1 - training_incaseof_fail[i]->required_output.data[j][0];
                }*/
                print_action(training_incaseof_fail[i]->required_output);
            }
            learn.nesterov_accelerated_gradient(&training_incaseof_fail[0], 1, training_incaseof_fail.size(), learning_rate, 0.9, false, regularization_rate, NULL, 1, 0, training_incaseof_fail.size());
            do
            {
                training_incaseof_fail.pop_back();
                training.pop_back();
            }while(training.size() > 0);
            //training.resize(0);
            gameover = false;
        }
        g.draw(gameover);
    }
    SDL_FreeSurface(window_surface);
}
