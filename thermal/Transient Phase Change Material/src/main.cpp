#include "../include/Nonlinear_Time.h"

int main(int argc, char **argv)
{
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    Nonlinear_Time nonlinear_time;
    nonlinear_time.run();
    return 0;
}