#ifndef PARAMETER_RELAX_H
#define PARAMETER_RELAX_H

#include <cmath>

namespace TopOpt
{
    class ParameterRelax
    {
    public:
        ParameterRelax(unsigned int step_init, unsigned int step_size,
                       double para_start, double para_end, double para_coeff);

        double parameter_update(unsigned int loop_counter);

    private:
        unsigned int step_init;
        unsigned int step_size;
        double para_start;
        double para_end;
        double para_coeff;
        double value;
    };
}

#endif