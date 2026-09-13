#include "../include/parameter_relax.h"

namespace TopOpt
{
    ParameterRelax::ParameterRelax(unsigned int step_init, unsigned int step_size,
                                   double para_start, double para_end, double para_coeff)
    {
        this->step_init = step_init;
        this->step_size = step_size;
        this->para_start = para_start;
        this->para_end = para_end;
        this->para_coeff = para_coeff;
    }

    double ParameterRelax::parameter_update(unsigned int loop_counter)
    {
        value = para_start;

        for (unsigned int i = 0; i <= loop_counter; i++)
        {
            if (((i - step_init) % step_size == 0) && (i >= step_init))
            {
                value = value * para_coeff;
            }
        }

        if (para_start < para_end)
            value = std::min(value, para_end);
        else if (para_start > para_end)
            value = std::max(value, para_end);
        else
            value = para_start;

        return value;
    }
}