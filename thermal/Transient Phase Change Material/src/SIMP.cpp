#include "../include/Nonlinear_Time.h"

double SIMP::simp_value(double x)
{
    return pow(x,penalty);
}

double SIMP::simp_grad_value(double x)
{
    return penalty * pow(x,penalty-1);
}

double OrderSIMP::simp_value(double x)
{
    double A, B = 0;
    int index = 0;
    if (x >= rho[0] && x < rho[1])
    {
        index = 0;
    }
    else if (x >= rho[1] && x < rho[2])
    {
        index = 1;
    }
    else
    {
        index = 2;
    }
    A = (kappa[index] - kappa[index + 1]) /
        (pow(rho[index], penalty) - 
         pow(rho[index+1], penalty));

    B = kappa[index] - A * pow(rho[index], penalty);

    return A * pow(x, penalty) + B;
}

double OrderSIMP::simp_grad_value(double x)
{
    double A = 0;
    int index = 0;
    if (x >= rho[0] && x < rho[1])
    {
        index = 0;
    }
    else if (x >= rho[1] && x < rho[2])
    {
        index = 1;
    }
    else
    {
        index = 2;
    }
    A = (kappa[index] - kappa[index + 1]) /
        (pow(rho[index], penalty) - 
         pow(rho[index+1], penalty));

    return A * pow(x, penalty - 1) * penalty;
}