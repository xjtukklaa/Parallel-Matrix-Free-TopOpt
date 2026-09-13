#include "../include/Heat_Matrix_Free.h"
int main(int argc, char *argv[])
{
  using namespace Heat_Matrix_Free;
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);
  Heat_Solver solver;
  solver.run();
  return 0;
}