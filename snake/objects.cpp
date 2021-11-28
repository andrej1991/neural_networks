#include "game.h"
#include "../additional.h"


Wall::Wall(SDL_Renderer* wall_renderer, int x, int y, int h, int w): wall_renderer(wall_renderer)
{
    obj = {x, y, w, h};
    c = {x, y, w, h};
    obj_texture = NULL;
}

Wall::~Wall()
{
    ;
}

void Wall::draw(char *img)
{
    SDL_SetRenderDrawColor(wall_renderer, 0x0, 0x05, 0x0, 0xFF);
    SDL_RenderFillRect(wall_renderer, &obj);
}


Snake::Snake(SDL_Renderer* r, int x, int y, int w, int h): renderer(r), ypos(y)
{
    length = 3;
    rect = {x + w*(length-1), y, w, h};
    xpos = x + w*(length-1);
    speed_x = w;
    speed_y = h;
    direction = RIGHT;
    coliders = new Colider[length];
    put(x, y);
}

void Snake::put(int x_pos, int y_pos)
{
    for(int i=0; i<length; i++)
    {
        coliders[i] = {x_pos + rect.w*(length-i-1), y_pos, rect.w, rect.h};
    }
    rect.x = x_pos + rect.w*(length-1);
    rect.y = y_pos;
    xpos = x_pos + rect.w*(length-1);
    ypos = y_pos;
    direction = RIGHT;
}

Snake::~Snake()
{
    ;
}

bool Snake::detect_colission(Wall **w, int wallcount)
{
    Colider c;
    for(int i = 0; i < wallcount; i++)
    {
        c = w[i]->get_colider();
        if(colission_detection(xpos, ypos, rect.w, rect.h, c))
        {
            return true;
        }
    }
    return false;
}

bool Snake::detect_colission(Food *f)
{
    return colission_detection(xpos, ypos, rect.w, rect.h, f->get_colider());
}

bool Snake::detect_self_colission()
{
    return false;
}

void Snake::shadow_move(int &x, int &y)
{
    switch(direction)
    {
        case LEFT:
            xpos -= speed_x;
            break;
        case UP:
            ypos -= speed_y;
            break;
        case RIGHT:
            xpos += speed_x;
            break;
        case DOWN:
            ypos += speed_y;
            break;
    }
}

void Snake::move()
{
    shadow_move(xpos, ypos);
    for(int i=length-1; i>0; i--)
    {
        coliders[i].xpos = coliders[i-1].xpos;
        coliders[i].ypos = coliders[i-1].ypos;
    }
    coliders[0].xpos = xpos;
    coliders[0].ypos = ypos;
}



bool Snake::handle_event(Matrix &action)
{
    int dir = getmax(action.data, 4);
    switch(dir)
    {
        case LEFT:
            if(direction != RIGHT)
            {
                direction = LEFT;
            }
            break;
        case UP:
            if(direction != DOWN)
            {
                direction = UP;
            }
            break;
        case RIGHT:
            if(direction != LEFT)
            {
                direction = RIGHT;
            }
            break;
        case DOWN:
            if(direction != UP)
            {
                direction = DOWN;
            }
            break;
    }
    move();
    return true;
}

void Snake::draw(char *img)
{
    rect.x = coliders[0].xpos;
    rect.y = coliders[0].ypos;
    SDL_SetRenderDrawColor(renderer, 139, 69, 19, 0xFF);
    SDL_RenderFillRect(renderer, &rect);
    for(int i=1; i<length; i++)
    {
        rect.x = coliders[i].xpos;
        rect.y = coliders[i].ypos;
        SDL_SetRenderDrawColor(renderer, 0, 200, 19, 0xFF);
        SDL_RenderFillRect(renderer, &rect);
    }
}

Food::Food(SDL_Renderer *r, int w, int h): renderer(r)
{
    c = {0, 0, w, h};
    rect = {0, 0, w, h};
}

Food::~Food()
{
    ;
}

void Food::draw()
{
    SDL_SetRenderDrawColor(renderer, 220, 69, 0, 0xFF);
    SDL_RenderFillRect(renderer, &rect);
}

void Food::set_coordinates(int x, int y)
{
    c.xpos = x;
    c.ypos = y;
    rect.x = x;
    rect.y = y;
}
