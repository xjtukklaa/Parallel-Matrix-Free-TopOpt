#include "../include/Heat_Matrix_Free.h"

using namespace Heat_Matrix_Free;

/**
 * @brief Set up the filter system of the heat solver.
 * 
 * This function initializes and configures the filter matrix and related data structures required for the filter system in the heat solver. It performs the following steps:
 * 
 * 1. Clear the existing filter matrix.
 * 2. Distribute the filter finite element (FE) and multigrid (MG) degrees of freedom (DoFs).
 * 3. Clear and set up constraints for hanging nodes.
 * 4. Configure additional data for the MatrixFree object, including parallel scheme and mapping update flags.
 * 5. Initialize the MatrixFree object using the mapping, DoF handler, constraints, quadrature, and additional data.
 * 6. Initialize the filter matrix using the MatrixFree object and set the filter radius.
 * 7. Initialize the degree-of-freedom vectors for the filter right-hand side (RHS) and filter solution.
 */
void Heat_Solver::setup_filter_system()
{
    TimerOutput::Scope t(computing_timer, "setup_filter_system");
    filter_matrix.clear();
    dof_handler_filter.distribute_dofs(fe_filter);
    dof_handler_filter.distribute_mg_dofs();
    // 
    constraints_filter.clear();
    constraints_filter.reinit(dof_handler_filter.locally_owned_dofs(),
                         DoFTools::extract_locally_relevant_dofs(dof_handler_filter));
    DoFTools::make_hanging_node_constraints(dof_handler_filter,constraints_filter);
    constraints_filter.close();
    // 
    typename MatrixFree<dimension, double>::AdditionalData additional_data;
    additional_data.tasks_parallel_scheme = MatrixFree<dimension, double>::AdditionalData::none;
    additional_data.mapping_update_flags = (update_gradients | update_JxW_values | update_quadrature_points | update_values);
    std::shared_ptr<MatrixFree<dimension, double>> filter_mf_storage(new MatrixFree<dimension, double>());
    filter_mf_storage->reinit(mapping_filter,
                              dof_handler_filter,
                              constraints_filter,
                              QGauss<1>(fe_filter.degree + 1),
                              additional_data);
    filter_matrix.initialize(filter_mf_storage);
    // 
    filter_matrix.Rmin_Pow2 = (double)(Rmin * Rmin);
    filter_matrix.compute_diagonal();
    // 
    filter_matrix.initialize_dof_vector(filter_rhs);
    filter_matrix.initialize_dof_vector(filter_solution);
    filter_matrix.initialize_dof_vector(Simp_Rho_Continuous_Filter);
}

/**
 * @brief Set up the multigrid filter system of the heat solver.
 *
 * This function initializes and configures the multigrid matrices and constrained degrees of freedom for the filter system in the heat solver.
 * It clears any existing elements and constraints, resizes the multigrid matrices, and initializes the constrained degrees of freedom.
 *
 * The function loops over each level of the triangulation, setting up the constraints and additional data required for the MatrixFree object.
 * It then initializes the multigrid matrices for each level with the appropriate settings.
 *
 * @note This function uses TimerOutput::Scope to measure the time taken by the setup process.
 */
void Heat_Solver::setup_multigrid_filter_system()
{
    TimerOutput::Scope t(computing_timer, "setup_multigrid_filter_system");
    // 
    mg_matrices_filter.clear_elements();
    mg_constrained_dofs_filter.clear();    
    //
    const unsigned int nlevels = triangulation.n_levels();
    mg_matrices_filter.resize(0, nlevels - 1);
    mg_constrained_dofs_filter.initialize(dof_handler_filter);
    // 
    for (unsigned int level = 0; level < nlevels; level++)
    {    
        AffineConstraints<float> level_constraints(dof_handler_filter.locally_owned_mg_dofs(level),
                                 DoFTools::extract_locally_relevant_level_dofs(dof_handler_filter, level));;
        level_constraints.close();
        typename MatrixFree<dimension, float>::AdditionalData additional_data;
        additional_data.tasks_parallel_scheme = MatrixFree<dimension, float>::AdditionalData::none;
        additional_data.mapping_update_flags = (update_gradients | update_JxW_values | update_quadrature_points | update_values);
        additional_data.mg_level = level;
        std::shared_ptr<MatrixFree<dimension, float>> mg_mf_storage_level = std::make_shared<MatrixFree<dimension, float>>();
        mg_mf_storage_level->reinit(mapping_filter,
                                    dof_handler_filter,
                                    level_constraints,
                                    QGauss<1>(fe_filter.degree + 1),
                                    additional_data);
        mg_matrices_filter[level].initialize(mg_mf_storage_level,
                                            mg_constrained_dofs_filter,
                                            level);
        mg_matrices_filter[level].Rmin_Pow2 = static_cast<float>(Rmin * Rmin);
        mg_matrices_filter[level].compute_diagonal();
    }
}

/**
 * @brief Assemble the right-hand side (RHS) vector for the filter operation in the heat solver.
 *
 * This function constructs the RHS vector for the filter operation by looping over all cell batches, evaluating the finite element (FE) values, and integrating them to form the global RHS vector.
 *
 * @param input Input vector containing the values used to assemble the filter RHS.
 *
 * The function performs the following steps:
 * 1. Initialize the filter RHS vector to zero.
 * 2. Update the ghost values in the input vector.
 * 3. Initialize the FE evaluation object for the filter matrix.
 * 4. Loop over all cell batches:
 *    a. Reinitialize the FE evaluation object for the current cell.
 *    b. Read the degree-of-freedom (DOF) values from the input vector.
 *    c. Submit the DOF values to the FE evaluation object for each quadrature point.
 *    d. Integrate the submitted values.
 *    e. Distribute the local contributions to the global filter RHS vector.
 * 5. Compress the filter RHS vector to complete the assembly.
 */
void Heat_Solver::assemble_filter_rhs(LinearAlgebra::distributed::Vector<double> input)
{
    TimerOutput::Scope t(computing_timer, "assemble_filter_rhs");
    filter_rhs = 0;    
    QGauss<dimension> quadrature_formula_filter(fe_filter.degree + 1);
    FEValues<dimension> fe_values_filter(fe_filter,
                                         quadrature_formula_filter,
                                         update_values | 
                                         update_quadrature_points | update_JxW_values);

    unsigned int dofs_per_cell = fe_filter.n_dofs_per_cell();
    Vector<double> cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices_filter(dofs_per_cell);
    std::vector<types::global_dof_index> local_rho_dof_indices_filter(1);
    std::vector<double> rho_values_filter(1);
    //
    auto cell_begin_filter = dof_handler_filter.begin_active();
    auto cell_end_filter = dof_handler_filter.end();
    auto base_begin_filter = dof_handler_simp.begin_active();
    //
    for (; cell_begin_filter != cell_end_filter; ++cell_begin_filter, ++base_begin_filter)
    {
        if (cell_begin_filter->is_locally_owned() && cell_begin_filter->level() == base_begin_filter->level())
        {
            cell_rhs = 0;
            fe_values_filter.reinit(cell_begin_filter);
            base_begin_filter->get_dof_indices(local_rho_dof_indices_filter);
            rho_values_filter[0] = input[local_rho_dof_indices_filter[0]];
            for (unsigned int q_point : fe_values_filter.quadrature_point_indices())
            {
                for (unsigned int i : fe_values_filter.dof_indices())
                {
                    cell_rhs(i) += (fe_values_filter.shape_value(i, q_point) *
                                    rho_values_filter[0] *
                                    fe_values_filter.JxW(q_point));
                }
            }
            cell_begin_filter->get_dof_indices(local_dof_indices_filter);
            constraints_filter.distribute_local_to_global(cell_rhs,
                                                          local_dof_indices_filter,
                                                          filter_rhs);
        }
    }
    filter_rhs.compress(VectorOperation::add);
    filter_rhs.update_ghost_values();
}

/**
 * @brief Solve the filter problem using the multigrid method.
 *
 * This function sets up and solves the linear system of the filter problem using the multigrid method.
 * It initializes the multigrid transfer, smoother, coarse grid solver, and interface matrices,
 * then uses these components to solve the system with a BiCGStab solver.
 *
 * The function performs the following steps:
 * 1. Initialize the multigrid transfer operator.
 * 2. Set up the smoother for each level of the multigrid hierarchy.
 * 3. Initialize the coarse grid solver.
 * 4. Build the multigrid matrix and interface matrices.
 * 5. Set up the multigrid solver using the matrix, coarse grid solver, transfer operator, and smoother.
 * 6. Solve the linear system using a BiCGStab solver with the multigrid preconditioner.
 * 7. Distribute the solution and update ghost values.
 * 8. Output the number of solver iterations.
 */
void Heat_Solver::solve_filter_chebyshev()
{
    TimerOutput::Scope t(computing_timer, "solve_filter_chebyshev");
    MGTransferMatrixFree<dimension, float> mg_transfer_filter(mg_constrained_dofs_filter);
    mg_transfer_filter.build(dof_handler_filter);
    using SmootherType = PreconditionChebyshev<LeveLFilterMatrixType, LeveLVectorType>;
    mg::SmootherRelaxation<SmootherType, LeveLVectorType> mg_smoother;
    MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
    smoother_data.resize(0, triangulation.n_global_levels() - 1);
    for (unsigned int level = 0; level < triangulation.n_global_levels(); ++level)
    {
        if (level > 0)
        {
            smoother_data[level].smoothing_range = 15.;
            smoother_data[level].degree = 5;
            smoother_data[level].eig_cg_n_iterations = 10;
        }
        else
        {
            smoother_data[0].smoothing_range = 1e-3;
            smoother_data[0].degree = numbers::invalid_unsigned_int;
            smoother_data[0].eig_cg_n_iterations = mg_matrices_filter[0].m();
        }
        mg_matrices_filter[level].compute_diagonal();
        smoother_data[level].preconditioner = mg_matrices_filter[level].get_matrix_diagonal_inverse();
    }
    mg_smoother.initialize(mg_matrices_filter, smoother_data);

    MGCoarseGridApplySmoother<LeveLVectorType> mg_coarse;
    mg_coarse.initialize(mg_smoother);

    mg::Matrix<LeveLVectorType> mg_matrix(mg_matrices_filter);

    MGLevelObject<MatrixFreeOperators::MGInterfaceOperator<LeveLFilterMatrixType>> mg_interface_matrices;
    mg_interface_matrices.resize(0, triangulation.n_global_levels() - 1);
    for (unsigned int level = 0; level < triangulation.n_global_levels(); ++level)
    {
        mg_interface_matrices[level].initialize(mg_matrices_filter[level]);
    }

    mg::Matrix<LeveLVectorType> mg_interface(mg_interface_matrices);

    Multigrid<LeveLVectorType> mg(mg_matrix, mg_coarse, mg_transfer_filter, mg_smoother, mg_smoother);
    mg.set_edge_matrices(mg_interface, mg_interface);

    PreconditionMG<dimension, LeveLVectorType, MGTransferMatrixFree<dimension, float>> preconditioner(dof_handler_filter, mg, mg_transfer_filter);
    // 
    filter_solution = 0;
    SolverControl solver_control_filter(dof_handler_filter.n_dofs(),1e-12,false,false);
    SolverFGMRES<VectorType> solver_filter(solver_control_filter);
    solver_filter.solve(filter_matrix,
                        filter_solution,
                        filter_rhs,
                        preconditioner);
    constraints_filter.distribute(filter_solution);
    filter_solution.update_ghost_values();
    pcout<<"                             "<< "Filter  Solver Last Steps : "<< solver_control_filter.last_step() << std::endl;
}

/**
 * @brief Generate an average vector based on the filter solution.
 *
 * This function computes the average value of the filter solution for each cell and stores it in the output vector.
 * It loops over all active cells in dof_handler_filter and dof_handler_simp,
 * computes the average value of the filter solution for each cell, and assigns it to the corresponding entry in the output vector.
 *
 * @param output Reference to the vector used to store the computed average values.
 */
void Heat_Solver::generate_average_vector(LinearAlgebra::distributed::Vector<double> &output)
{
    TimerOutput::Scope t(computing_timer, "generate_average_vector");
    //
    auto cell_begin_aver = dof_handler_filter.begin_active();
    auto cell_end_aver = dof_handler_filter.end();
    auto rho_begin_aver = dof_handler_simp.begin_active();
    //
    double average_i;
    std::vector<types::global_dof_index> local_dof_indices_aver(fe_filter.n_dofs_per_cell());
    std::vector<types::global_dof_index> local_rho_dof_indices_aver(fe_simp.n_dofs_per_cell());
    for (; cell_begin_aver != cell_end_aver; ++cell_begin_aver, ++rho_begin_aver)
    {
        if (cell_begin_aver->is_locally_owned() && cell_begin_aver->level() == rho_begin_aver->level())
        {
            average_i = 0;
            cell_begin_aver->get_dof_indices(local_dof_indices_aver);
            rho_begin_aver->get_dof_indices(local_rho_dof_indices_aver);
            for (unsigned int i = 0; i < fe_filter.n_dofs_per_cell(); i++)
            {
                average_i += filter_solution[local_dof_indices_aver[i]] / (double)fe_filter.n_dofs_per_cell();
            }
            output[local_rho_dof_indices_aver[0]] = average_i;
        }
    }
    output.compress(VectorOperation::insert);
}