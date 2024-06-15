#ifndef GAME_H_INCLUDED
#define GAME_H_INCLUDED

#include <SDL2/SDL.h>
#include "../matrix/matrix.h"

enum DIRECTION {LEFT, UP, RIGHT, DOWN};


struct Colider
{
    int xpos, ypos, width, height;
};


class Wall
{
    SDL_Texture *obj_texture;
    SDL_Renderer* wall_renderer;
    SDL_Rect obj;
    Colider c;
    public:
    Wall(SDL_Renderer* wall_renderer, int x, int y, int h, int w);
    ~Wall();
    void draw(char *img);
    inline Colider& get_colider() {return c;};
    friend class Game;
};

class Food
{
    Colider c;
    SDL_Rect rect;
    SDL_Renderer* renderer;
    public:
    Food(SDL_Renderer *r, int w, int h);
    ~Food();
    void draw();
    inline Colider& get_colider() {return c;};
    void set_coordinates(int x, int y);
    friend class Game;
};

class Snake
{
    SDL_Renderer* renderer;
    SDL_Rect rect;
    Colider *coliders;
    int length, xpos, ypos, direction, speed_x, speed_y;
    public:
    bool detect_colission(Wall **w, int wallcount);
    bool detect_colission(Food *f);
    bool detect_self_colission();
    Snake(SDL_Renderer* r, int x, int y, int w, int h);
    ~Snake();
    bool handle_event(Matrix &action);
    void move();
    void draw(char *img);
    void put(int x_pos, int y_pos);
    friend class Game;
};

class Game
{
    SDL_Window* window;
    SDL_Event event;
    SDL_Texture *bckgrnd_texture;
    SDL_Rect background;
    Wall **walls;
    Snake *snake;
    Food *food;
    int window_w, window_h, wallcount, score;
    public:
    SDL_Renderer *bckgrnd_renderer;
    bool quit;
    Game(int h, int w);
    ~Game();
    //void play();
    void draw(bool gameover);
    void drop_food();
    friend class Network;
};


inline bool colission_detection(int xpos, int ypos, int w, int h, Colider c)
{
    bool x_colission = false;
    bool y_colission = false;
    if((((xpos+w) >= c.xpos) && ((xpos+w) <= (c.xpos+c.width))) || ((xpos >= c.xpos) && (xpos <= (c.xpos+c.width))))
    {
        x_colission = true;
    }
    if((((c.xpos + c.width) >= xpos) && ((c.xpos+c.width) <= (xpos+w))) || ((c.xpos >= xpos) && (c.xpos <= (xpos + w ))))
    {
        x_colission = true;
    }
    if((((ypos+h) >= c.ypos) && ((ypos+h) <= (c.ypos+c.height))) || ((ypos >= c.ypos) && (ypos <= (c.ypos+c.height))))
    {
        y_colission = true;
    }
    if((((c.ypos + c.height) >= ypos) && ((c.ypos+c.height) <= (ypos+h))) || ((c.ypos >= ypos) && (c.ypos <= (ypos + h))))
    {
        y_colission = true;
    }
    return (x_colission && y_colission);
}


#endif // GAME_H_INCLUDED
