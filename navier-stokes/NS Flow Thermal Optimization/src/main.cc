#include "../include/top_opt_heat_flow.h"


int main(int argc, char *argv[])
{
  try
  {
    using namespace dealii;
    using namespace TopOpt;
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    ParameterHandler prm;
    PRM::ParameterModifier prm_mdf(prm);
    prm_mdf.read_parameters("../input/parameters.prm");
    prm_mdf.write_parameters();

    TopOptHeatFlow top_opt_heat_flow(/* degree = */ PRM::degree,
                                     /* top_degree =  */ PRM::top_degree);
    top_opt_heat_flow.run(PRM::loop, PRM::refinement);
  }
  catch (std::exception &exc)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }
  catch (...)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }
  return 0;
}
