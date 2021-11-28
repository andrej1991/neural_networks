#include "game.h"
#include "../network.h"
#include "../SGD.h"

#define input_col 50
#define input_row 50

void reinforcement_snake(double learning_rate, double regularization_rate)
{
    SDL_Event event;
    SDL_Surface *window_surface = SDL_CreateRGBSurface(0, input_col, input_row, 32, 0x00ff0000, 0x0000ff00, 0x000000ff, 0xff000000);
    bool gameover = false;
    Game g(input_row, input_col);
    g.drop_food();
    MNIST_data *d;
    Matrix action;
    std::vector<MNIST_data*> training;
    ifstream inp;
    int direction;
    double temp;
    while(!g.quit)
    {
        while(SDL_PollEvent(&event) != 0)
        {
            if(event.type == SDL_QUIT)
            {
                g.quit = true;
            }
        }
        d = new MNIST_data(input_row, input_col, 4, 3);
        if(!gameover)
        {
            SDL_LockSurface(window_surface);
            SDL_RenderReadPixels(g.bckgrnd_renderer, NULL, window_surface->format->format, window_surface->pixels, window_surface->pitch);
            SDL_SaveBMP(window_surface, "../window_surface.bmp");
            SDL_UnlockSurface(window_surface);
            inp.open("../window_surface.bmp");
            d[0].load_bmp(inp);
            inp.close();
            action = this->get_output(d[0].input);
            print_action(action);
            d[0].required_output = action;
            if(g.snake->handle_event(action))
            {
                training.push_back(d);
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
                        //temp = 1.1 * training[i]->required_output.data[direction][0];
                        for(int j=0; j<training[i]->required_output.get_row(); j++)
                        {
                            training[i]->required_output.data[j][0] = 0;
                        }
                        training[i]->required_output.data[direction][0] = 1;
                    }
                    this->update_weights_and_biasses(training, training.size(), 100*training.size(), learning_rate, regularization_rate);
                    do
                    {
                        training.pop_back();
                    }while(training.size() > 0);
                    //training.resize(0);
                }
            }
            else
            {
                for(int i=0; i<d[0].required_output.get_row(); i++)
                {
                    d[0].required_output.data[i][0] *= -1 ;
                }
                training.push_back(d);
                this->update_weights_and_biasses(training, training.size(), 100*training.size(), learning_rate, regularization_rate);
                do
                {
                    training.pop_back();
                }while(training.size() > 0);
                //training.resize(0);
            }
        }
        else
        {
            cout << "game over\n";
            g.score = 0;
            for(int i=0; i<training.size(); i++)
                {
                    for(int j=0; j<training[i]->required_output.get_row(); j++)
                    {
                        training[i]->required_output.data[j][0] = 1 - training[i]->required_output.data[j][0];
                    }
                }
            this->update_weights_and_biasses(training, training.size(), 100*training.size(), learning_rate, regularization_rate);
            do
            {
                training.pop_back();
            }while(training.size() > 0);
            //training.resize(0);
            gameover = false;
        }
        g.draw(gameover);
    }
    SDL_FreeSurface(window_surface);
}

void print_action(Matrice &action)
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
