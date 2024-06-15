#include "game.h"
#include "../additional.h"
#include <unistd.h>
#include <math.h>
#include <algorithm>


void print_action(Matrix &action)
{
    int dir = argmax(action.data, 4);
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
        /**the top left corner is the 0,0*/
        if(snake_colider.ypos < food_colider.ypos && g.snake->get_direction().direction == DOWN)
        {
            //cout << "front1\n";
            return true;
        }
        /**the top left corner is the 0,0*/
        if(snake_colider.ypos > food_colider.ypos && g.snake->get_direction().direction == UP)
        {
            //cout << "front2\n";
            return true;
        }
    }
    if(snake_colider.ypos == food_colider.ypos)
    {
        if(snake_colider.xpos < food_colider.xpos && g.snake->get_direction().direction == RIGHT)
        {
            //cout << "front3\n";
            return true;
        }
        if(snake_colider.xpos > food_colider.xpos && g.snake->get_direction().direction == LEFT)
        {
            //cout << "front4\n";
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
        /**the top left corner is the 0,0*/
        if(snake_colider.ypos < food_colider.ypos && g.snake->get_direction().direction == UP)
        {
            //cout << "behind1\n";
            return true;
        }
        /**the top left corner is the 0,0*/
        if(snake_colider.ypos > food_colider.ypos && g.snake->get_direction().direction == DOWN)
        {
            //cout << "behind2\n";
            return true;
        }
    }
    if(snake_colider.ypos == food_colider.ypos)
    {
        if(snake_colider.xpos < food_colider.xpos && g.snake->get_direction().direction == LEFT)
        {
            //cout << "behind3\n";
            return true;
        }
        if(snake_colider.xpos > food_colider.xpos && g.snake->get_direction().direction == RIGHT)
        {
            //cout << "behind4\n";
            return true;
        }
    }
    return false;
}

bool food_is_intheleft(Game &g)
{
    int direction = g.snake->get_direction().direction;
    Colider snake_colider, food_colider;
    snake_colider = g.snake->get_colider()[0];
    food_colider = g.food->get_colider();
    if(direction == UP || direction == DOWN)
    {
        if(snake_colider.xpos >= food_colider.xpos)
        {
            //cout << "left\n";
            return true;
        }
    }
    return false;
}

bool food_is_intheright(Game &g)
{
    int direction = g.snake->get_direction().direction;
    Colider snake_colider, food_colider;
    snake_colider = g.snake->get_colider()[0];
    food_colider = g.food->get_colider();
    if(direction == UP || direction == DOWN)
    {
        if(snake_colider.xpos <= food_colider.xpos)
        {
            //cout << "right\n";
            return true;
        }
    }
    return false;
}

bool food_is_above(Game &g)
{
    int direction = g.snake->get_direction().direction;
    Colider snake_colider, food_colider;
    snake_colider = g.snake->get_colider()[0];
    food_colider = g.food->get_colider();
    if(direction == LEFT || direction == RIGHT)
    {
        /**the top left corner is the 0,0*/
        if(snake_colider.ypos >= food_colider.ypos)
        {
            //cout << "abowe\n";
            return true;
        }
    }
    return false;
}

bool food_is_below(Game &g)
{
    int direction = g.snake->get_direction().direction;
    Colider snake_colider, food_colider;
    snake_colider = g.snake->get_colider()[0];
    food_colider = g.food->get_colider();
    if(direction == LEFT || direction == RIGHT)
    {
        /**the top left corner is the 0,0*/
        if(snake_colider.ypos <= food_colider.ypos)
        {
            //cout << "below\n";
            return true;
        }
    }
    return false;
}

void select_unblocked_path(Game &g, Matrix &ret, int *prefered)
{
    Direction d(g.snake->get_direction().direction);
    Colider c = g.snake->get_colider()[0];
    for(int i = 0; i < 3; i++)
    {
        /*if(i > 0)
        {
            cout << "colission on the previous choice:  " << i-1 << endl;
        }*/
        g.snake->shadow_move(c.xpos, c.ypos, prefered[i]);
        if(!(g.snake->detect_colission(g.walls, g.get_wallcount(), c.xpos, c.ypos) || g.snake->detect_self_colission(c.xpos, c.ypos)))
        {
            ret.data[prefered[i]][0] = 1;
            return;
        }
    }
    cout << "totally fucked up\n";
}

Matrix get_correct_way(Game &g)
{
    Matrix ret(4, 1);
    Direction d(g.snake->get_direction().direction);
    Colider c = g.snake->get_colider()[0];
    if(food_is_infront(g))
    {
        int req[3] = {g.snake->get_direction().direction, (g.snake->get_direction()+1), (g.snake->get_direction()-1)};
        select_unblocked_path(g, ret, req);
        return ret;

    }
    if(food_is_behind(g))
    {
        int req[3] = {(g.snake->get_direction()+1), (g.snake->get_direction()-1), g.snake->get_direction().direction};
        select_unblocked_path(g, ret, req);
        return ret;
    }
    if(food_is_intheleft(g))
    {
        int req[3] = {LEFT, g.snake->get_direction().direction, RIGHT};
        select_unblocked_path(g, ret, req);
        return ret;
    }
    if(food_is_intheright(g))
    {
        int req[3] = {RIGHT, g.snake->get_direction().direction, LEFT};
        select_unblocked_path(g, ret, req);
        return ret;
    }
    if(food_is_above(g))
    {
        int req[3] = {UP, g.snake->get_direction().direction, DOWN};
        select_unblocked_path(g, ret, req);
        return ret;
    }
    if(food_is_below(g))
    {
        int req[3] = {DOWN, g.snake->get_direction().direction, UP};
        select_unblocked_path(g, ret, req);
        return ret;
    }
}

bool snake_is_totally_wrong(Matrix *action, Matrix *required_action)
{
    bool ret = false;
    for(int i = 0; i < action->get_row(); i++)
    {
        if(required_action->data[i][0] != 1)
        {
            if(action->data[i][0] >= (1 - 1E-5))
            {
                ret = true;
            }
        }
    }
    return ret;
}

void reinforcement_snake(Network &net, StochasticGradientDescent &learn, double learning_rate, double regularization_rate, int input_row, int input_col, double momentum, double denominator, int input_channels)
{
    bool enableg = false;
    bool gameover = false;
    bool debugprint = false;
    SDL_Event event;
    int debugx, debugy;
    //int games = 0;
    double avarange_score, szoras = 0;
    int median_score;
    SDL_Surface *window_surface = SDL_CreateRGBSurface(0, input_col, input_row, 32, 0x00ff0000, 0x0000ff00, 0x000000ff, 0xff000000);
    Game g(input_row, input_col);
    g.drop_food();
    g.snake->enable_growth = enableg;
    //g.play();
    //return;
    Data_Loader *d, *train_incaseof_fail;
    Matrix action;
    std::vector<Data_Loader*> training;
    //std::vector<MNIST_data*> training_incaseof_fail;
    std::vector<int> score_list;
    int direction, highscore = 0, last5steps = 5;
    double temp;
    while(!g.quit)
    {
        if(debugprint)
        {
            usleep(200000);
        }
        while(SDL_PollEvent(&event) != 0)
        {
            if(event.type == SDL_QUIT)
            {
                g.quit = true;
                double sum = 0;
                for(int i: score_list)
                {
                    sum += i;
                }
                avarange_score = sum/score_list.size();
                for(int i: score_list)
                {
                    szoras += (i-avarange_score)*(i-avarange_score);
                }
                szoras /= score_list.size();
                szoras = sqrt(szoras);
                for(int i: score_list)
                {
                    cout << i << " ";
                }
                cout << endl;
                cout << "avarange: " << avarange_score << endl;
                sort(score_list.begin(), score_list.end());
                cout << "median score " << score_list[score_list.size()/2] << endl;
                cout << "szórás: " << szoras << endl;
                cout << g.score << endl;
            }
            if((event.type == SDL_MOUSEBUTTONUP) && (event.button.button == SDL_BUTTON_LEFT))
            {

                debugprint = !debugprint;
            }
            if((event.type == SDL_MOUSEBUTTONUP) && (event.button.button == SDL_BUTTON_RIGHT))
            {
                cout << "killed by mouse\n";
                gameover = true;
            }
        }
        d = new Data_Loader(input_row, input_col, 4, input_channels);
        train_incaseof_fail = new Data_Loader(input_row, input_col, 4, input_channels);
        if(!gameover)
        {
            SDL_RenderReadPixels(g.bckgrnd_renderer, NULL, window_surface->format->format, window_surface->pixels, window_surface->pitch);
            SDL_LockSurface(window_surface);
            d[0].load_sdl_pixels(window_surface);
            SDL_UnlockSurface(window_surface);
            for(int i = 0; i < input_channels; i++)
            {
                train_incaseof_fail[0].input[i][0] = d[0].input[i][0];
            }
            action = net.get_output(d[0].input);
            d[0].required_output = action;
            train_incaseof_fail[0].required_output = get_correct_way(g);
            //training_incaseof_fail.push_back(train_incaseof_fail);
            g.snake->handle_event(action);
            debugx = g.snake->get_colider()[0].xpos;
            debugy = g.snake->get_colider()[0].ypos;
            training.push_back(d);
            if(debugprint)
            {
                cout << "=======\n";
                print_action(action);
                print_action(train_incaseof_fail[0].required_output);
                cout << "x: " << debugx << "   y: " << debugy << endl;
                cout << "foodx: " << g.food->c.xpos << "   foody: " << g.food->c.ypos << endl;
            }
            gameover = g.snake->detect_colission(g.walls, g.wallcount);
            gameover |= g.snake->detect_self_colission();
            if(g.snake->detect_colission(g.food))
            {
                g.score++;
                g.drop_food();
                for(int i=0; i<training.size(); i++)
                {
                    direction = argmax(training[i]->required_output.data, 4);
                    for(int j=0; j<training[i]->required_output.get_row(); j++)
                    {
                        training[i]->required_output.data[j][0] = 0;
                    }
                    training[i]->required_output.data[direction][0] = 1;
                }
                /*(MNIST_data **training_data, int epochs, int minibatch_len, double learning_rate, double momentum, bool monitor_learning_cost = false,
                   double regularization_rate = 0, MNIST_data **test_data = NULL, int minibatch_count = 500, int test_data_len = 10000,  int trainingdata_len = 50000);*/
                if(training.size() > 5)
                    last5steps = 5;
                else last5steps = training.size();
                for(int i = last5steps - 1; i > 0; i --)
                {
                    learn.rmsprop(&training[training.size() - i], 8 - i, 1, learning_rate, momentum, false, regularization_rate, denominator, NULL, 1, 0, 1);
                    //learn.momentum_gradient_descent(&training[training.size() - i], 8 - i, 1, learning_rate, momentum, false, regularization_rate, NULL, 1, 0, 1);
                }

                do
                {
                    delete training[training.size()-1];
                    training.pop_back();
                    //delete training_incaseof_fail[training_incaseof_fail.size()-1];
                    //training_incaseof_fail.pop_back();
                }while(training.size() > 0);
            }
            else
            {
                if(!gameover)
                {
                    int k = 1;
                    if(snake_is_totally_wrong(&action, &(train_incaseof_fail->required_output)))
                    {
                        if(training.size() > input_col)
                            k = 100;
                        else
                            k = 10;
                    }
                    if(debugprint)
                    {
                        cout << "the value of K: " << k << endl;
                    }
                    learn.rmsprop(&train_incaseof_fail, k, 1, learning_rate, momentum, false, regularization_rate, denominator, NULL, 1, 0, 1);
                    //learn.momentum_gradient_descent(&train_incaseof_fail, k, 1, learning_rate, momentum, false, regularization_rate, NULL, 1, 0, 1);
                    if(training.size() > (input_col+input_col + input_row+input_row)*2 && k > 1)
                    {
                        cout << "intentionally killed\n";
                        gameover = true;
                    }
                }
            }
        }
        else
        {
            if(g.score >= highscore)
            {
                highscore = g.score;
                cout << "highest score: " << highscore << endl;
            }
            score_list.push_back(g.score);
            g.score = 0;
            learn.rmsprop(&train_incaseof_fail, 5, 1, learning_rate, momentum, false, regularization_rate, denominator, NULL, 1, 0, 1);
            //learn.momentum_gradient_descent(&train_incaseof_fail, 5, 1, learning_rate, momentum, false, regularization_rate, NULL, 1, 0, 1);
            do
            {
                //delete training_incaseof_fail[training_incaseof_fail.size()-1];
                //training_incaseof_fail.pop_back();
                delete training[training.size()-1];
                training.pop_back();
            }while(training.size() > 0);
            delete g.snake;
            g.snake = g.create_snake();
            g.snake->enable_growth = enableg;
            gameover = false;
        }
        g.draw(gameover);
    }
    cout << highscore << endl;
    SDL_FreeSurface(window_surface);
}
