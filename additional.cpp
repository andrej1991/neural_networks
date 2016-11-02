#include "additional.h"

int getmax(double **d)
{
    double Max = d[0][0];
    int index = 0;
    for(int i = 0; i < 10; i++)
        {
            if(Max < d[i][0])
                {
                    Max = d[i][0];
                    index = i;
                }
        }
    return index;
}

int partition( int *a, int l, int r) {
    int pivot, i, j, t;
    pivot = a[l];
    i = l; j = r+1;

    while( 1)
        {
            do ++i;
            while( a[i] <= pivot && i <= r )
                ;
            do --j;
            while( a[j] > pivot );
            if( i >= j )
                break;
            t = a[i]; a[i] = a[j]; a[j] = t;
        }
    t = a[l]; a[l] = a[j]; a[j] = t;
    return j;
}

void quickSort( int *a, int l, int r)
{
    int j;

    if( l < r )
    {
    // divide and conquer
        j = partition( a, l, r);
        quickSort( a, l, j-1);
        quickSort( a, j+1, r);
    }

}
