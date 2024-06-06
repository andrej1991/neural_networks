#include "game.h"
#include "../additional.h"
#include <unistd.h>


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
        /**the top left corner is the 0,0*/
        if(snake_colider.ypos < food_colider.ypos && g.snake->get_direction().direction == DOWN)
        {
            cout << "front1\n";
            return true;
        }
        /**the top left corner is the 0,0*/
        if(snake_colider.ypos > food_colider.ypos && g.snake->get_direction().direction == UP)
        {
            cout << "front2\n";
            return true;
        }
    }
    if(snake_colider.ypos == food_colider.ypos)
    {
        if(snake_colider.xpos < food_colider.xpos && g.snake->get_direction().direction == RIGHT)
        {
            cout << "front3\n";
            return true;
        }
        if(snake_colider.xpos > food_colider.xpos && g.snake->get_direction().direction == LEFT)
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
        /**the top left corner is the 0,0*/
        if(snake_colider.ypos < food_colider.ypos && g.snake->get_direction().direction == UP)
        {
            cout << "behind1\n";
            return true;
        }
        /**the top left corner is the 0,0*/
        if(snake_colider.ypos > food_colider.ypos && g.snake->get_direction().direction == DOWN)
        {
            cout << "behind2\n";
            return true;
        }
    }
    if(snake_colider.ypos == food_colider.ypos)
    {
        if(snake_colider.xpos < food_colider.xpos && g.snake->get_direction().direction == LEFT)
        {
            cout << "behind3\n";
            return true;
        }
        if(snake_colider.xpos > food_colider.xpos && g.snake->get_direction().direction == RIGHT)
        {
            cout << "behind4\n";
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
            cout << "left\n";
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
            cout << "right\n";
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
            cout << "abowe\n";
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
        ret.data[g.snake->get_direction().direction][0] = 1;
        return ret;
    }
    if(food_is_behind(g))
    {
        Direction d(g.snake->get_direction().direction);
        d = d+1;
        Colider c = g.snake->get_colider()[0];
        g.snake->shadow_move(c.xpos, c.ypos, d.direction);
        if(g.snake->detect_colission(g.walls, g.get_wallcount(), c.xpos, c.ypos) || g.snake->detect_self_colission())
        {
            ret.data[g.snake->get_direction()-1][0] = 1;
        }
        else
        {
            ret.data[d.direction][0] = 1;
        }
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

void reinforcement_snake(Network &net, StochasticGradientDescent &learn, double learning_rate, double regularization_rate, int input_row, int input_col, double momentum, double denominator)
{
    bool enableg = true;
    SDL_Event event;
    int debugx, debugy;
    SDL_Surface *window_surface = SDL_CreateRGBSurface(0, input_col, input_row, 32, 0x00ff0000, 0x0000ff00, 0x000000ff, 0xff000000);
    bool gameover = false;
    Game g(input_row, input_col);
    g.drop_food();
    g.snake->enable_growth = enableg;
    //g.play();
    //return;
    MNIST_data *d, *train_incaseof_fail;
    Matrix action;
    std::vector<MNIST_data*> training;
    std::vector<MNIST_data*> training_incaseof_fail;
    ifstream inp;
    int direction, highscore = 0, last5steps = 5;
    double temp;
    while(!g.quit)
    {
        //usleep(200000);
        while(SDL_PollEvent(&event) != 0)
        {
            if(event.type == SDL_QUIT)
            {
                g.quit = true;
                cout << g.score << endl;
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
            d[0].required_output = action;
            train_incaseof_fail[0].required_output = get_correct_way(g);
            training_incaseof_fail.push_back(train_incaseof_fail);
            g.snake->handle_event(action);
            debugx = g.snake->get_colider()[0].xpos;
            debugy = g.snake->get_colider()[0].ypos;
            training.push_back(d);

            /*print_action(action);
            print_action(training_incaseof_fail[training.size()-1]->required_output);
            cout << "x: " << debugx << "   y: " << debugy << endl;
            cout << "foodx: " << g.food->c.xpos << "   foody: " << g.food->c.ypos << endl;
            cout << "=======\n";*/

            gameover = g.snake->detect_colission(g.walls, g.wallcount);
            gameover |= g.snake->detect_self_colission();
            if(g.snake->detect_colission(g.food))
            {
                //cout << "food found\n";
                g.score++;
                g.drop_food();
                gameover = false;
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
                if(training.size() > 5)
                    last5steps = 5;
                else last5steps = training.size();
                for(int i = 0; i < 3; i++)
                {
                    for(int i = last5steps; i > 0 ; i--)
                        learn.rmsprop(&training[training.size()-i], 1, 1, learning_rate, momentum, false, regularization_rate, denominator, NULL, 1, 0, 1);
                }
                do
                {
                    training.pop_back();
                    training_incaseof_fail.pop_back();
                }while(training.size() > 0);
            }
            else
            {
                if(!gameover)
                    learn.rmsprop(&train_incaseof_fail, 1, 1, learning_rate, momentum, false, regularization_rate, denominator, NULL, 1, 0, 1);
            }
        }
        else
        {
            /*print_action(action);
            print_action(training_incaseof_fail[training.size()-1]->required_output);
            cout << "x: " << debugx << "   y: " << debugy << endl;
            cout << "=======\n";*/
            if(g.score >= highscore)
            {
                highscore = g.score;
                cout << "highest score: " << highscore << endl;
            }
            g.score = 0;
            for(int i = 0; i < 3; i++)
                learn.rmsprop(&training_incaseof_fail[training.size()-1], 1, 1, learning_rate, momentum, false, regularization_rate, denominator, NULL, 1, 0, 1);
            do
            {
                training_incaseof_fail.pop_back();
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
