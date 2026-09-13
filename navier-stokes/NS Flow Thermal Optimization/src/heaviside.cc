#include "../include/heaviside.h"

namespace TopOpt
{
    Heaviside::Heaviside(const double beta, const double yita)
    {
        this->beta = beta;
        this->yita = yita;
    }

    double Heaviside::value(const double xphys)
    {
        if (beta == 1.0)
            return xphys;
        else
            return ((std::tanh(beta * yita) + std::tanh(beta * (xphys - yita))) /
                    (std::tanh(beta * yita) + std::tanh(beta * (1.0 - yita))));
    }

    double Heaviside::partial(const double xphys)
    {
        if (beta == 1.0)
            return 1.0;
        else
            return ((beta * std::pow(1.0 / std::cosh(beta * (xphys - yita)), 2.0)) /
                    (std::tanh(beta * yita) + std::tanh(beta * (1.0 - yita))));
    }
}