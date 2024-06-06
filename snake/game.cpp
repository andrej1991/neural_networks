#include <time.h>
#include "game.h"

#include <iostream>
using namespace std;

#define SNAKE_W 5
#define SNAKE_H 5


Game::Game(int h, int w): window_h(h), window_w(w), window(NULL), bckgrnd_renderer(NULL)
{
    quit = false;
    background = {0, 0, window_w, window_h};
    wallcount = 4;
    score = 0;
    if( SDL_Init( SDL_INIT_VIDEO ) < 0 )
	{
		printf( "SDL could not initialize! SDL Error: %s\n", SDL_GetError() );
	}
	else
	{
        window = SDL_CreateWindow( "SDL Tutorial", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, window_w, window_h, SDL_WINDOW_SHOWN );
        if( window == NULL )
        {
            printf( "Window could not be created! SDL_Error: %s\n", SDL_GetError() );
        }
        else
        {
            bckgrnd_renderer = SDL_CreateRenderer( window, -1, SDL_RENDERER_ACCELERATED );
            if( bckgrnd_renderer == NULL )
			{
				printf( "Renderer could not be created! SDL Error: %s\n", SDL_GetError() );
			}
            SDL_SetRenderDrawColor(bckgrnd_renderer, 0x00, 0x00, 0xF5, 0xFF);
        }
    }
    walls = new Wall* [wallcount];
    walls[0] = new Wall(bckgrnd_renderer, 0, 0, window_h, SNAKE_W-1);
    walls[1] = new Wall(bckgrnd_renderer, SNAKE_W - 1, 0, SNAKE_H - 1, window_w - 2*(SNAKE_W-1));
    walls[2] = new Wall(bckgrnd_renderer, window_w - SNAKE_W + 1, 0, window_h, SNAKE_W - 1);
    walls[3] = new Wall(bckgrnd_renderer, SNAKE_W - 1, window_h - SNAKE_H + 1, SNAKE_W - 1, window_w - 2*(SNAKE_W-1));
    snake = create_snake();
    food = new Food(bckgrnd_renderer, SNAKE_W, SNAKE_H);
}

Snake* Game::create_snake()
{
    Snake *s = new Snake(bckgrnd_renderer, 2*SNAKE_W, window_h/2 - (window_h/2)%SNAKE_H, SNAKE_W, SNAKE_H);
    return s;
}

Game::~Game()
{
    SDL_DestroyRenderer( bckgrnd_renderer );
	SDL_DestroyWindow( window );
	SDL_Quit();
}

void Game::draw(bool gameover)
{
    SDL_SetRenderDrawColor(bckgrnd_renderer, 0xFF, 0xFF, 0xFF, 0xFF);
    SDL_RenderClear(bckgrnd_renderer);
    SDL_SetRenderDrawColor(bckgrnd_renderer, 0x00, 0x00, 0xF5, 0xFF);
    SDL_RenderFillRect(bckgrnd_renderer, &background);
    for(int i = 0; i < wallcount; i++)
    {
        walls[i]->draw("");
    }
    int h = window_h/snake->coliders[0].height/2*snake->coliders[0].height;
    if(gameover)
        snake->put(snake->coliders[0].width, h);
    snake->draw("");
    food->draw();
    SDL_RenderPresent(bckgrnd_renderer);
}

void Game::drop_food()
{
    bool colission = false;
    int food_x, food_y;
    srand(time(NULL));
    do
    {
        colission = false;
        food_x = rand() % window_w;
        food_x -= food_x%snake->coliders[0].width;
        food_y = rand() % window_h;
        food_y -= food_y%snake->coliders[0].height;
        for(int i=0; i<wallcount; i++)
        {
            if(colission_detection(food_x, food_y, food->c.width, food->c.height, walls[i]->c))
            {
                colission = true;
            }
        }
        for(int i=0; i<snake->length; i++)
        {
            if(colission_detection(food_x, food_y, food->c.width, food->c.height, snake->coliders[i]))
            {
                colission = true;
            }
        }
    }while(colission);
    food->set_coordinates(food_x, food_y);
}

void Game::play()
{
    SDL_Event event;
    bool gameover = false;
    this->drop_food();
    while(!quit)
    {
        while(SDL_PollEvent(&event) != 0)
        {
            if(event.type == SDL_QUIT)
            {
                quit = true;
            }
            if(!gameover)
            {
                snake->handle_event(event);
                gameover = snake->detect_colission(walls, wallcount);
                gameover |= snake->detect_self_colission();
                if(snake->detect_colission(food))
                {
                    score++;
                    drop_food();
                }
            }
        }
        this->draw(gameover);
        if(gameover)
        {
            delete snake;
            snake = create_snake();
        }
        gameover = false;
    }
    return;
}
