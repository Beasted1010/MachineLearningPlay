

#include <stdio.h>
#include <stdlib.h>



typedef enum Variable {
    X,
    Y,
    Z
} Variable;

typedef struct Point {
    int x;
    int y;
} Point;

typedef struct Function2D {
    int xScalar;
    int xExponent;
    int yScalar;
    int yExponent;
    int out;
} Function2D;

typedef struct Gradient2D {
    Function2D outUnitI;
    Function2D outUnitJ;
} Gradient2D;

Point* CreatePoint(int x, int y)
{
    Point* point = malloc(sizeof(Point));

    point->x = x;
    point->y = y;

    return point;
}

void DestroyPoint(Point* point)
{
    free(point);
}

Function2D* CreateFunction2D(int xScalar, int xExponent, int yScalar, int yExponent)
{
    Function2D* function = malloc(sizeof(Function2D));

    function->xScalar = xScalar;
    function->xExponent = xExponent;
    function->yScalar = yScalar;
    function->yExponent = yExponent;

    function->out = 0;

    return function;
}

void DestroyFunction2D(Function2D* function)
{
    free(function);
}

// Function to apply the power rule to the scalar and exponent of a variable.
// TODO: Make this support more complex things? Such as trigonometry functions, exponentials, logarithms, etc?
void ApplyPowerRule(int* variableScalar, int* variableExponent)
{
    *variableScalar *= *variableExponent;
    *variableExponent -= 1;
}

int Exponentiate(int base, int exponent)
{
    int result = 1;
    for(int i = 0; i < exponent; i++)
    {
        result *= base;
    }

    return result;
}

// TODO: Supporting only polynomials for now
// TODO: And these polynomials must consist of individual variables being summed together (with potential scalar)
void CalculatePartialDerivative(Function2D* function, Variable variable)
{
    switch(variable)
    {
        case X:
        {
            ApplyPowerRule(&function->xScalar, &function->xExponent);
        } break;

        case Y:
        {
            ApplyPowerRule(&function->yScalar, &function->yExponent);
        } break;

        default:
        {
            printf("ERROR: The variable used is unsupported!\n");
        } break;
    }
}


Gradient2D* CreateGradient(Function2D* function, Point* point)
{
    Gradient2D* result = malloc(sizeof(Gradient2D));

    result->outUnitI.out = 0;
    result->outUnitJ.out = 0;
    
    if(function->xScalar != 0)
    {
        CalculatePartialDerivative(function, X);
        result->outUnitI.out += function->xScalar * (Exponentiate(point->x, function->xExponent));
    }
    if(function->yScalar != 0)
    {
        CalculatePartialDerivative(function, Y);
        result->outUnitJ.out += function->yScalar * (Exponentiate(point->y, function->yExponent));
    }

    return result;
}

void PrintFunction(Function2D* function)
{
    if(function->xScalar != 0)
    {
        printf("%i", function->xScalar);
        if(function->xExponent != 0)
        {
            printf("x^%i ", function->xExponent);
        }
    }
    printf("\t");
    if(function->yScalar != 0)
    {
        printf("%i", function->yScalar);
        if(function->yExponent != 0)
        {
            printf("y^%i ", function->yExponent);
        }
    }
    printf("\n");
}


void PrintEvaluatedGradient(Gradient2D* gradient)
{
    printf("< %i, %i >\n", gradient->outUnitI.out, gradient->outUnitJ.out);
}


int main(int argc, char** argv)
{
    Function2D* function = CreateFunction2D(1, 2, 3, 1);
    Point* point = CreatePoint(3, 2);

    PrintFunction(function);

    Gradient2D* gradient = CreateGradient(function, point);
    PrintFunction(function);

    printf("\n"); PrintEvaluatedGradient(gradient);


    DestroyPoint(point);
    DestroyFunction2D(function);
}


