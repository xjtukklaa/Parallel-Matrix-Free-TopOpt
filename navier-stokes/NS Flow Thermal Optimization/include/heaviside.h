#ifndef HEAVISIDE_H
#define HEAVISIDE_H

#include <cmath>

namespace TopOpt
{
    class Heaviside
    {
    public:
        Heaviside(const double beta, const double yita);

        double value(const double xphys);

        double partial(const double xphys);

    private:
        double beta;
        double yita;
    };
}

#endif