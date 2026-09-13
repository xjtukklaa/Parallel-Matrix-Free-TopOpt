#include "../include/Heat_Matrix_Free.h"

using namespace Heat_Matrix_Free;

void Heat_Solver::setup_laplace_system()
{
    TimerOutput::Scope t(computing_timer, "setup_laplace_system");
    //
    laplace_matrix.clear();
    //
    dof_handler.distribute_dofs(fe);
    dof_handler.distribute_mg_dofs();
    //
    constraints.clear();
    constraints.reinit(dof_handler.locally_owned_dofs(),
                   DoFTools::extract_locally_relevant_dofs(dof_handler));
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    VectorTools::interpolate_boundary_values(mapping, 
                                             dof_handler, 
                                             1, 
                                             Functions::ConstantFunction<dimension,double>(0), 
                                             constraints);
    constraints.close();
    //
    typename MatrixFree<dimension, double>::AdditionalData additional_data;
    additional_data.tasks_parallel_scheme = MatrixFree<dimension, double>::AdditionalData::none;
    additional_data.mapping_update_flags = (update_gradients | update_JxW_values | update_quadrature_points | update_values);
    std::shared_ptr<MatrixFree<dimension, double>> system_mf_storage(new MatrixFree<dimension, double>());
    system_mf_storage->reinit(mapping,
                              dof_handler,
                              constraints,
                              QGauss<1>(fe.degree + 1),
                              additional_data);
    laplace_matrix.initialize(system_mf_storage);

    laplace_matrix.initialize_dof_vector(laplace_rhs);
    laplace_matrix.initialize_dof_vector(laplace_temperature);
    laplace_matrix.initialize_dof_vector(Simp_Rho_Continuous);
}

void Heat_Solver::setup_multigrid_laplace_system()
{
    TimerOutput::Scope t(computing_timer, "setup_multigrid_laplace_system");
    // Clear multigrid
    mg_matrices.clear_elements();
    mg_constrained_dofs.clear();
    //
    const unsigned int nlevels = triangulation.n_levels();
    mg_matrices.resize(0, nlevels - 1);
    // Periodic boundary conditions on multigrid levels
    mg_constrained_dofs.initialize(dof_handler);
    mg_constrained_dofs.make_zero_boundary_constraints(dof_handler, {1});
    //
    for (unsigned int level = 0; level < nlevels; level++)
    {    
        AffineConstraints<float> level_constraints(dof_handler.locally_owned_mg_dofs(level),
                                 DoFTools::extract_locally_relevant_level_dofs(dof_handler, level));

        for (const types::global_dof_index dof_index : mg_constrained_dofs.get_boundary_indices(level))
        {
            level_constraints.constrain_dof_to_zero(dof_index);
        } 
        level_constraints.close();

        typename MatrixFree<dimension, float>::AdditionalData additional_data;
        additional_data.tasks_parallel_scheme = MatrixFree<dimension, float>::AdditionalData::none;
        additional_data.mapping_update_flags = (update_gradients | update_JxW_values | update_quadrature_points | update_values);
        additional_data.mg_level = level;
        std::shared_ptr<MatrixFree<dimension, float>> mg_mf_storage_level(std::make_shared<MatrixFree<dimension, float>>());
        mg_mf_storage_level->reinit(mapping,
                                    dof_handler,
                                    level_constraints,
                                    QGauss<1>(fe.degree + 1),
                                    additional_data);

        mg_matrices[level].initialize(mg_mf_storage_level,
                                      mg_constrained_dofs,
                                      level);
    }
}

void Heat_Solver::transfer_simp_vector()
{
    TimerOutput::Scope t(computing_timer, "transfer_simp_vector");
    Simp_Rho_Continuous = 0;
    //
    std::vector<types::global_dof_index> local_dof_indices(fe.n_dofs_per_cell());
    std::vector<types::global_dof_index> local_rho_dof_indices(fe_simp.n_dofs_per_cell());
    //
    auto cell_iter_begin = dof_handler.begin_active();
    auto cell_iter_end = dof_handler.end();
    auto simp_iter = dof_handler_simp.begin_active();
    //
    for (; cell_iter_begin != cell_iter_end; cell_iter_begin++, simp_iter++)
    {
        if (cell_iter_begin->is_locally_owned())
        {
            cell_iter_begin->get_dof_indices(local_dof_indices);
            simp_iter->get_dof_indices(local_rho_dof_indices);
            Simp_Rho_Continuous[local_dof_indices[0]] += Simp_Rho_Discrete_Filter[local_rho_dof_indices[0]];
        }
    }
    Simp_Rho_Continuous.compress(VectorOperation::add);
    Simp_Rho_Continuous.update_ghost_values();
}

void Heat_Solver::interpolate_simp_vector()
{
    TimerOutput::Scope t(computing_timer, "interpolate_simp_vector");
    //
    const unsigned int nlevels = triangulation.n_levels();
    Simp_Rho_Continuous_Level.clear();
    Simp_Rho_Continuous_Level.resize(0, nlevels - 1);
    for (unsigned int level = 0; level < nlevels; level++)
    {
        mg_matrices[level].initialize_dof_vector(Simp_Rho_Continuous_Level[level]);
    }
    //
    transfer_simp_vector();
    laplace_matrix.evaluate_coefficient(Simp_Rho_Continuous, Material_Thermal_Conductivity_Matrix);
    laplace_matrix.compute_diagonal();
    //
    MGTransferMatrixFree<dimension, float> mg_vector_transfer;
    mg_vector_transfer.build(dof_handler);
    mg_vector_transfer.interpolate_to_mg(dof_handler,
                                         Simp_Rho_Continuous_Level,
                                         Simp_Rho_Continuous);
    for (unsigned int i = 0; i < nlevels; i++)
    {
        Simp_Rho_Continuous_Level[i].update_ghost_values();
        mg_matrices[i].evaluate_coefficient(Simp_Rho_Continuous_Level[i], LeveLMaterial_Thermal_Conductivity_Matrix);
        mg_matrices[i].compute_diagonal();
    }
}

/**
 * @brief Assemble the right-hand side (RHS) of the Laplace equation.
 *
 * This function initializes the RHS vector to zero, then loops over all cells and quadrature points to compute the contribution of the gradients to the RHS.
 * The computed values are then integrated and distributed to the global RHS vector.
 *
 * This function uses matrix-free evaluation to perform the assembly, which is efficient for large-scale problems.
 *
 * @note This function assumes that `laplace_matrix` and `unit_test_rhs` are properly initialized, and that `dimension` and `degree_finite_element` are defined.
 */
void Heat_Solver::assemble_laplace_rhs()
{
    TimerOutput::Scope t(computing_timer, "assemble_laplace_rhs");
    laplace_rhs = 0;
    QGauss<dimension> quadrature_formula(fe.degree + 1);
    FEValues<dimension> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

    unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    
    Vector<double> cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    //
    auto cell_begin = dof_handler.begin_active();
    auto cell_end = dof_handler.end();
    auto rho_begin = dof_handler_simp.begin_active();
    //
    for (; cell_begin != cell_end; ++cell_begin, ++rho_begin)
    {
        if (cell_begin->is_locally_owned() && cell_begin->level() == rho_begin->level())
        {
            cell_rhs = 0;
            fe_values.reinit(cell_begin);
            cell_begin->get_dof_indices(local_dof_indices);

            for (const unsigned int q_index : fe_values.quadrature_point_indices())
            {
                for (const unsigned int i : fe_values.dof_indices())
                {
                    cell_rhs[i] += (fe_values.shape_value(i, q_index) *
                                    10 *
                                    fe_values.JxW(q_index));
                }
            }
            constraints.distribute_local_to_global(cell_rhs,
                                                   local_dof_indices,
                                                   laplace_rhs);                
        }
    }
    laplace_rhs.compress(VectorOperation::add);
    laplace_rhs.update_ghost_values();
}

void Heat_Solver::solve_chebyshev()
{
    TimerOutput::Scope t(computing_timer, "solve_chebyshev");
    MGTransferMatrixFree<dimension, float> mg_transfer(mg_constrained_dofs);
    mg_transfer.build(dof_handler);

    using SmootherType = PreconditionChebyshev<LeveLLaplaceMatrixType, LeveLVectorType>;
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
            smoother_data[0].eig_cg_n_iterations = mg_matrices[0].m();
        }
        mg_matrices[level].compute_diagonal();
        smoother_data[level].preconditioner = mg_matrices[level].get_matrix_diagonal_inverse();
    }
    mg_smoother.initialize(mg_matrices, smoother_data);

    MGCoarseGridApplySmoother<LeveLVectorType> mg_coarse;
    mg_coarse.initialize(mg_smoother);

    mg::Matrix<LeveLVectorType> mg_matrix(mg_matrices);

    MGLevelObject<MatrixFreeOperators::MGInterfaceOperator<LeveLLaplaceMatrixType>> mg_interface_matrices;
    mg_interface_matrices.resize(0, triangulation.n_global_levels() - 1);
    for (unsigned int level = 0; level < triangulation.n_global_levels(); ++level)
    {
        mg_interface_matrices[level].initialize(mg_matrices[level]);
    }

    mg::Matrix<LeveLVectorType> mg_interface(mg_interface_matrices);

    Multigrid<LeveLVectorType> mg(mg_matrix, mg_coarse, mg_transfer, mg_smoother, mg_smoother);
    mg.set_edge_matrices(mg_interface, mg_interface);

    PreconditionMG<dimension, LeveLVectorType, MGTransferMatrixFree<dimension, float>> preconditioner(dof_handler, mg, mg_transfer);

    SolverControl solve_control(dof_handler.n_dofs(), 1e-12 * laplace_rhs.l2_norm());
    laplace_temperature = 0;
    SolverBicgstab<VectorType> solver(solve_control);
    solver.solve(laplace_matrix,
                 laplace_temperature,
                 laplace_rhs,
                 preconditioner);
    constraints.distribute(laplace_temperature);
    laplace_temperature.update_ghost_values();
    pcout<<"                             "<< "Laplace Solver Last Steps : "<< solve_control.last_step() << std::endl;
}

void Heat_Solver::generate_object_discrete_derivative()
{
    TimerOutput::Scope t(computing_timer, "generate_object_discrete_derivative");
    Object_Discrete_Derivative = 0;
    QGauss<dimension> quadrature_formula(fe.degree + 1);
    FEValues<dimension> fe_values(fe,
                                  quadrature_formula,
                                  update_values | update_gradients |
                                  update_quadrature_points | update_JxW_values);

    std::vector<types::global_dof_index> local_simp_dof_indices(fe_simp.n_dofs_per_cell());
    std::vector<double> rho_values(1);

    std::vector<Tensor<1, dimension, double>> solution_gradients(fe_values.n_quadrature_points);
    // 
    Tensor<2, dimension, double> Thermal_Conductivity_Derivative_Matrix;
    //
    auto cell_begin = dof_handler.begin_active();
    auto cell_end = dof_handler.end();
    auto base_begin = dof_handler_simp.begin_active();
    //
    for (; cell_begin != cell_end; ++cell_begin, ++base_begin)
    {
        if (cell_begin->is_locally_owned() && cell_begin->level() == base_begin->level())
        {
            fe_values.reinit(cell_begin);
            // Simp
            base_begin->get_dof_indices(local_simp_dof_indices);
            rho_values[0] = Simp_Rho_Discrete_Filter[local_simp_dof_indices[0]];
            Thermal_Conductivity_Derivative_Matrix = penal * (1. - epsimin) * SumMaterial_Thermal_Conductivity_Matrix 
                                                     * pow(rho_values[0], penal - 1);

            fe_values.get_function_gradients(laplace_temperature, solution_gradients);

            for (unsigned int q_point : fe_values.quadrature_point_indices())
            {
                Object_Discrete_Derivative[local_simp_dof_indices[0]] += 
                                         -solution_gradients[q_point] * 
                                          Thermal_Conductivity_Derivative_Matrix * 
                                          solution_gradients[q_point] *
                                          fe_values.JxW(q_point);

            }
        }
    }
    Object_Discrete_Derivative.compress(VectorOperation::add);
}