#include "../include/Nonlinear_Time.h"

double PCM::cp_t_x_value(double t, double x)
{
    return ((l*(1-x))/(2*delta_t)-cp_s/2)*tanh(t-t_start)-
           ((l*(1-x))/(2*delta_t)-cp_l/2)*tanh(t-t_start-delta_t)+
           (cp_l+cp_s)/2;
}

double PCM::dcp_dt_x_value(double t, double x)
{
    return ((l*(1-x))/(2*delta_t)-cp_s/2)*(1-pow(tanh(t-t_start),2))-
           ((l*(1-x))/(2*delta_t)-cp_l/2)*(1-pow(tanh(t-t_start-delta_t),2));
}

double PCM::dcp_t_dx_value(double t, double x)
{
    return (-l/(2*delta_t))*tanh(t-t_start)-
           (-l/(2*delta_t))*tanh(t-t_start-delta_t);
}