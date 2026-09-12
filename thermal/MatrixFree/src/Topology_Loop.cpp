#include "../include/Heat_Matrix_Free.h"

using namespace Heat_Matrix_Free;

/**
 * @brief Constructor of the Heat_Solver class.
 *
 * This constructor initializes the Heat_Solver object, including:
 * - Setting up the MPI communicator.
 * - Constructing and initializing the triangulation with the MPI communicator, level difference limit at vertices, and multigrid hierarchy.
 * - Initializing the finite element objects with the specified degrees.
 * - Initializing the DoF handlers for the main, simplified, and filter triangulations.
 * - Setting up the parallel output stream for the root MPI process.
 * - Initializing the computing timer with the MPI communicator, output stream, and timing options.
 */
Heat_Solver::Heat_Solver()
    : mpi_communicator(MPI_COMM_WORLD),
      triangulation(mpi_communicator,
                    Triangulation<dimension>::limit_level_difference_at_vertices,
                    parallel::distributed::Triangulation<dimension>::construct_multigrid_hierarchy),
      mapping(FE_Q<dimension>(degree_finite_element)),
      mapping_filter(FE_Q<dimension>(filter_degree_finite_element)),
      fe(degree_finite_element),
      fe_simp(0),
      fe_filter(filter_degree_finite_element),
      dof_handler(triangulation),
      dof_handler_simp(triangulation),
      dof_handler_filter(triangulation),
      pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0),
      computing_timer(mpi_communicator,
                      pcout,
                      TimerOutput::never,
                      TimerOutput::wall_times)
{
    for(unsigned int i = 0; i < dimension; i++)
    { 
        Material_Thermal_Conductivity_Matrix[i][i] = make_vectorized_array<double>(10.);
        LeveLMaterial_Thermal_Conductivity_Matrix[i][i] = make_vectorized_array<float>(10.);
        SumMaterial_Thermal_Conductivity_Matrix[i][i] = 10.;
    }   
}

/**
 * @brief Generate the mesh for the heat solver and set up periodic boundary conditions.
 *
 * This function creates a hypercube mesh within the range [0, 1] in each dimension.
 * Then, it loops over the active faces of the mesh and assigns boundary IDs based on the face center positions. For 3D meshes, additional boundary IDs are assigned to faces in the z-dimension.
 *
 * After setting the boundary IDs, the function collects periodic face pairs for each dimension and adds them to the triangulation. Finally, the mesh is globally refined six times.
 *
 * @note This function uses TimerOutput::Scope to measure the time required for mesh generation.
 */
void Heat_Solver::make_grid()
{
    TimerOutput::Scope t(computing_timer, "make_grid");
    GridGenerator::hyper_cube(triangulation, -1, 1);
    triangulation.refine_global(7);
    for (auto &face : triangulation.active_face_iterators())
    {
        if (std::fabs(face->center()(0) - 0.) <=  0.1 &&
            std::fabs(face->center()(1) - 0.) <=  0.1 &&
            std::fabs(face->center()(2) + 1.) <=  1e-10)
        {
            face->set_boundary_id(1);
        }
    }
}

/**
 * @brief Initialize the simplified vector structures used in the heat solver.
 *
 * This function sets up the degrees of freedom (DoFs) for the simplified finite element (FE), and initializes the discrete vectors for the simplified density (Simp_Rho_Discrete),
 * cell volume (Cell_Volume_Discrete), and filtered simplified density (Simp_Rho_Discrete_Filter).
 * 
 * It uses TimerOutput::Scope to measure the time required for the initialization process.
 * 
 * @note This function assumes that the finite element (fe_simp) and MPI communicator (mpi_communicator) have been set up.
 */
void Heat_Solver::setup_simp_system()
{
    TimerOutput::Scope t(computing_timer, "setup_simp_system");
    dof_handler_simp.distribute_dofs(fe_simp);

    Simp_Rho_Discrete.reinit(dof_handler_simp.locally_owned_dofs(), mpi_communicator);
    Simp_Rho_Discrete_Filter.reinit(dof_handler_simp.locally_owned_dofs(), mpi_communicator);  
    Simp_Rho_Discrete_Max.reinit(dof_handler_simp.locally_owned_dofs(), mpi_communicator);
    Simp_Rho_Discrete_Min.reinit(dof_handler_simp.locally_owned_dofs(), mpi_communicator);

    Cell_Volume_Discrete.reinit(dof_handler_simp.locally_owned_dofs(), mpi_communicator);
    Object_Discrete_Derivative.reinit(dof_handler_simp.locally_owned_dofs(), mpi_communicator);
    
    Constraint_Discrete_Derivative.resize(1);
    Constraint_Discrete_Derivative[0].reinit(dof_handler_simp.locally_owned_dofs(), mpi_communicator);
}

/**
 * @brief Compute the volume of each cell in the mesh and store it in the Cell_Volume_Discrete vector.
 *
 * This function loops over all active cells in the dof_handler_simp object. For each locally owned cell,
 * it retrieves the degree-of-freedom (DOF) indices and adds the cell measure (volume) to the corresponding entry in the Cell_Volume_Discrete vector.
 * After processing all cells, the vector is compressed to combine contributions from different processes (if running in parallel), and ghost values are updated.
 * Finally, the function computes the filter radius (Rmin), which is the square root of the infinity norm of the Cell_Volume_Discrete vector.
 *
 * @note This function uses a TimerOutput::Scope object to measure the time required for execution.
 */
void Heat_Solver::get_cell_volume()
{
    TimerOutput::Scope t(computing_timer, "get_cell_volume");
    Cell_Volume_Discrete = 0;
    //
    std::vector<types::global_dof_index> local_rho_dof_indices(fe_simp.n_dofs_per_cell());
    for (auto &cell_iter : dof_handler_simp.active_cell_iterators())
    {
        if (cell_iter->is_locally_owned())
        {
            cell_iter->get_dof_indices(local_rho_dof_indices);
            Cell_Volume_Discrete[local_rho_dof_indices[0]] += cell_iter->measure();
        }
    }
    Cell_Volume_Discrete.compress(VectorOperation::add);
    Cell_Volume_Discrete.update_ghost_values();
}

/**
 * @brief Output the results of the heat solver to VTU files.
 *
 * This function handles the output of various data vectors related to the heat solver.
 * It attaches the degree-of-freedom (DoF) handler to a DataOut object and adds multiple data vectors
 * for continuous and discrete variables as well as filter-related data. The results are then written to
 * VTU files with appropriate compression settings.
 *
 * The following data vectors are output:
 * - Simp_Rho_Continuous: Continuous simplified density.
 * - unit_test_rhs: Right-hand side vector of the unit test (one per dimension).
 * - unit_test_temperature: Temperature vector of the unit test (one per dimension).
 * - Simp_Rho_Discrete: Discrete simplified density.
 * - Cell_Volume_Discrete: Discrete cell volume.
 * - filter_rhs: Right-hand side vector of the filter.
 * - filter_solution: Solution vector of the filter.
 *
 * The function also sets the compression level of the output files for optimal speed.
 *
 * @note This function uses TimerOutput::Scope to measure the time required for the output process.
 */
void Heat_Solver::output_results()
{
    TimerOutput::Scope t(computing_timer, "output_results");
    // Simp Output
    DataOut<dimension> data_out;
    data_out.attach_dof_handler(dof_handler_simp);
    data_out.add_data_vector(dof_handler_simp, Simp_Rho_Discrete, "Simp_Rho_Discrete");
    data_out.add_data_vector(dof_handler_simp, Simp_Rho_Discrete_Filter, "Simp_Rho_Discrete_Filter");
    data_out.add_data_vector(dof_handler_simp, Cell_Volume_Discrete, "Cell_Volume_Discrete");
    data_out.add_data_vector(dof_handler_simp, Object_Discrete_Derivative, "Object_Discrete_Derivative");
    data_out.add_data_vector(dof_handler_simp, Constraint_Discrete_Derivative[0], "Constraint_Discrete_Derivative");
    data_out.build_patches();
    DataOutBase::VtkFlags flags;
    flags.compression_level = DataOutBase::CompressionLevel::best_speed;
    data_out.set_flags(flags);
    data_out.write_vtu_with_pvtu_record("./", "simp", MMA_Optimizer.Loop_Iter, mpi_communicator);
}

double Heat_Solver::mma_optimizer(Vector<double> input)
{
    TimerOutput::Scope t(computing_timer, "mma_optimizer");
    MMA_Optimizer.Deal_II_MMA_Dual_Problem_Solve(Simp_Rho_Discrete,
                                                 Object_Discrete_Derivative,
                                                 input,
                                                 Constraint_Discrete_Derivative,
                                                 Simp_Rho_Discrete_Max,
                                                 Simp_Rho_Discrete_Min);
    return MMA_Optimizer.Deal_II_MMA_Get_Change(Simp_Rho_Discrete);
}

void Heat_Solver::run()
{
    const unsigned int n_vect_doubles = VectorizedArray<double>::size();
    const unsigned int n_vect_bits = 8 * sizeof(double) * n_vect_doubles;

    pcout << "Vectorization over " << n_vect_doubles
          << " doubles = " << n_vect_bits << " bits ("
          << Utilities::System::get_current_vectorization_level() << ')'
          << std::endl;
    // 
    Vector<double> Object_Function(1);
    Vector<double> Constraint_Function(1);
    double change = 1e10;
    double volfrac = 0.3;
    // Initialize mesh
    pcout<<"+---------------------------------------------------------------------------+"<<std::endl;
    pcout<<"|                             INIT OPTIMIZATION                             |"<<std::endl;
    pcout<<"+---------------------------------------------------------------------------+"<<std::endl;
    pcout<<std::endl;
    make_grid();
    
    setup_simp_system();
    get_cell_volume();
    // Compute filter radius; the filter radius must be larger than the cell length, otherwise the solution will be incorrect
    Rmin = 2. * pow(Cell_Volume_Discrete.linfty_norm(), 1./(double)dimension) / (2. * sqrt(3.));
    pcout<<"Rmin : "<<Rmin<<std::endl;
    pcout<<std::endl;
    
    // Initialize MMA optimizer
    MMA_Optimizer.Deal_II_MMA_Init(1,0,1e3,Simp_Rho_Discrete);
    MMA_Optimizer.Loop_Iter = 0;    

    setup_laplace_system();
    setup_multigrid_laplace_system();

    setup_filter_system();
    setup_multigrid_filter_system();

    assemble_filter_rhs(Cell_Volume_Discrete);
    solve_filter_chebyshev();
    generate_average_vector(Constraint_Discrete_Derivative[0]);
    Constraint_Discrete_Derivative[0].operator/=(volfrac * Cell_Volume_Discrete.l1_norm());

    pcout<<std::endl;
    pcout<<"+---------------------------------------------------------------------------+"<<std::endl;
    pcout<<"|                             DEGREE OF FREEDOM                             |"<<std::endl;
    pcout<<"+---------------------------------------------------------------------------+"<<std::endl;
    pcout<<std::endl;

    pcout<<"+---------------------------------------------------------------------------+"<<std::endl;
    pcout<<"                             "<< "Simp    : "<< dof_handler_simp.n_dofs() << std::endl;
    pcout<<"                             "<< "Laplace : "<< dof_handler.n_dofs() << std::endl;
    pcout<<"                             "<< "Filter  : "<< dof_handler_filter.n_dofs() << std::endl;
    pcout<<"+---------------------------------------------------------------------------+"<<std::endl; 
    pcout<<std::endl;

    pcout<<std::endl;
    pcout<<"+---------------------------------------------------------------------------+"<<std::endl;
    pcout<<"|                            START OPTIMIZATION                             |"<<std::endl;
    pcout<<"+---------------------------------------------------------------------------+"<<std::endl;
    // MMA optimization loop
    while (MMA_Optimizer.Loop_Iter < 1000 && change > 1e-3 * dof_handler_simp.n_dofs())  
    { 
        MMA_Optimizer.Loop_Iter++; 

        if (MMA_Optimizer.Loop_Iter == 1)
        {
            Simp_Rho_Discrete = 1.;
            Simp_Rho_Discrete_Max = 1.;
            Simp_Rho_Discrete_Min = 1.e-3;    
        }

        pcout<<std::endl;
        pcout<<"+---------------------------------------------------------------------------+"<<std::endl;
        pcout<<"|                             SOLVER INFOMATION                             |"<<std::endl;
        pcout<<"+---------------------------------------------------------------------------+"<<std::endl;
        pcout<<std::endl;

        pcout<<"+---------------------------------------------------------------------------+"<<std::endl;
        assemble_filter_rhs(Simp_Rho_Discrete);
        solve_filter_chebyshev();
        Simp_Rho_Continuous_Filter = filter_solution;
        generate_average_vector(Simp_Rho_Discrete_Filter);

        interpolate_simp_vector();
        assemble_laplace_rhs();
        solve_chebyshev();
        generate_object_discrete_derivative();

        // Objective sensitivity filtering
        assemble_filter_rhs(Object_Discrete_Derivative);
        solve_filter_chebyshev();
        generate_average_vector(Object_Discrete_Derivative);

        pcout<<"+---------------------------------------------------------------------------+"<<std::endl;
        pcout<<std::endl;

        // Compute volume constraint
        Constraint_Function[0] = Simp_Rho_Discrete_Filter * Cell_Volume_Discrete;
        Constraint_Function[0] /= volfrac * Cell_Volume_Discrete.l1_norm();
        Constraint_Function[0] -= 1.;   
        // Compute objective function
        Object_Function[0] = laplace_rhs * laplace_temperature;
        // Update design variables
        change = mma_optimizer(Constraint_Function);

        pcout<<std::endl;
        pcout<<"+---------------------------------------------------------------------------+"<<std::endl;
        pcout<<"|                              OUTPUT  RESULTS                              |"<<std::endl;
        pcout<<"+---------------------------------------------------------------------------+"<<std::endl;

        pcout<<std::endl;
        pcout<<"                             "<< "Iter    : "<<MMA_Optimizer.Loop_Iter<<std::endl
             <<"                             "<< "Object  : "<<Object_Function[0]<<std::endl
             <<"                             "<< "Change  : "<<change<<std::endl
             <<"                             "<< "Volume  : "<<(Constraint_Function[0] + 1) * volfrac<<std::endl;
        pcout<<"+---------------------------------------------------------------------------+"<<std::endl;
        pcout<<std::endl;    
        if (MMA_Optimizer.Loop_Iter %10 == 0) { output_results(); };   
    }                
    time();  
}

void Heat_Solver::time()
{
    computing_timer.print_summary();
    computing_timer.reset();
}