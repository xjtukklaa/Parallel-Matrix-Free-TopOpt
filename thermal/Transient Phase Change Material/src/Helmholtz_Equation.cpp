#include "../include/Nonlinear_Time.h"

void Nonlinear_Time::setup_filter_system()
{
    TimerOutput::Scope t(computing_timer, "setup_filter_system");
    dof_handler_filter.distribute_dofs(fe_filter);

    locally_owned_dofs_filter = dof_handler_filter.locally_owned_dofs();
    locally_relevant_dofs_filter = 
    DoFTools::extract_locally_relevant_dofs(dof_handler_filter);

    helmholtz_constraints.clear();
    helmholtz_constraints.reinit(locally_owned_dofs_filter,locally_relevant_dofs_filter);
    DoFTools::make_hanging_node_constraints(dof_handler_filter,helmholtz_constraints);
    helmholtz_constraints.close();

    DynamicSparsityPattern dsp_Filter(dof_handler_filter.n_dofs(),
                                      dof_handler_filter.n_dofs(),
                                      locally_relevant_dofs_filter);   
    DoFTools::make_sparsity_pattern(dof_handler,
                                    dsp_Filter,
                                    helmholtz_constraints,
                                    false);                      
    SparsityTools::distribute_sparsity_pattern(dsp_Filter,
                                               locally_owned_dofs_filter,
                                               mpi_communicator,
                                               locally_relevant_dofs_filter);
    helmholtz_matrix.reinit(locally_owned_dofs_filter,
                            locally_owned_dofs_filter,
                            dsp_Filter,
                            mpi_communicator);
    
    helmholtz_rhs.reinit(locally_owned_dofs_filter,mpi_communicator);
    helmholtz_locally_solution.reinit(locally_owned_dofs_filter,mpi_communicator);
    helmholtz_solution.reinit(locally_owned_dofs_filter,
                              locally_relevant_dofs_filter,
                              mpi_communicator);
}

void Nonlinear_Time::assemble_filter_matrix()
{
    TimerOutput::Scope t(computing_timer, "assemble_filter_matrix");

    helmholtz_matrix = 0;

    QGauss<dim> quadrature_formula_filter(fe_filter.degree + 1);
    FEValues<dim> filter_fe_values(fe_filter, quadrature_formula_filter,
                                   update_values | update_gradients |
                                   update_quadrature_points | update_JxW_values);
    unsigned int dofs_per_cell = fe_filter.n_dofs_per_cell();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices_filter(dofs_per_cell);

    for (auto & cell_begin_filter : dof_handler_filter.active_cell_iterators())
    {
        if (cell_begin_filter->is_locally_owned())
        {
            cell_matrix = 0;
            filter_fe_values.reinit(cell_begin_filter);
            for (unsigned int filter_points : filter_fe_values.quadrature_point_indices())
            {
                for (unsigned int filter_i : filter_fe_values.dof_indices())
                {
                    for (unsigned int filter_j : filter_fe_values.dof_indices())
                    {
                        cell_matrix(filter_i, filter_j) +=
                            (Rmin * Rmin *
                                 filter_fe_values.shape_grad(filter_i, filter_points) *
                                 filter_fe_values.shape_grad(filter_j, filter_points) 
                                +
                                 filter_fe_values.shape_value(filter_i, filter_points) *
                                 filter_fe_values.shape_value(filter_j, filter_points)
                            ) *
                            filter_fe_values.JxW(filter_points);
                    }
                }
            }
            cell_begin_filter->get_dof_indices(local_dof_indices_filter);
            helmholtz_constraints.distribute_local_to_global(cell_matrix,
                                                             local_dof_indices_filter,
                                                             helmholtz_matrix);
        }
    }
    helmholtz_matrix.compress(VectorOperation::add);
}

void Nonlinear_Time::assemble_filter_rhs(PETScWrappers::MPI::Vector intput)
{
    TimerOutput::Scope t(computing_timer, "assemble_filter_rhs");

    helmholtz_rhs = 0;

    QGauss<dim> quadrature_formula_filter(fe_filter.degree + 1);
    FEValues<dim> fe_values_filter(fe_filter, quadrature_formula_filter,
                                   update_values | update_gradients |
                                   update_quadrature_points | update_JxW_values);
    unsigned int dofs_per_cell = fe_filter.n_dofs_per_cell();

    QGauss<dim> quadrature_formula_simp(fe_simp.degree + 1);
    FEValues<dim> fe_values_simp(fe_simp, quadrature_formula_simp,
                                 update_values | update_quadrature_points);

    Vector<double> cell_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices_filter(dofs_per_cell);

    std::vector<double>  intput_values(1);

    auto cell = dof_handler_filter.begin_active();
    auto cell_simp = dof_handler_simp.begin_active();
    auto cell_end = dof_handler_filter.end();

    for (;cell != cell_end; cell++, cell_simp++)
    {        
        if (cell->is_locally_owned())
        {
            cell_rhs = 0;

            fe_values_filter.reinit(cell);
            
            fe_values_simp.reinit(cell_simp);
            fe_values_simp.get_function_values(intput, intput_values);

            for (unsigned int q_point : fe_values_filter.quadrature_point_indices())
            {
                for (unsigned int i : fe_values_filter.dof_indices())
                {
                    cell_rhs(i) += (fe_values_filter.shape_value(i, q_point) *
                                    intput_values[0] *
                                    fe_values_filter.JxW(q_point));
                }
            }
            cell->get_dof_indices(local_dof_indices_filter);
            helmholtz_constraints.distribute_local_to_global(cell_rhs,
                                                             local_dof_indices_filter,
                                                             helmholtz_rhs);
        }
    }
    helmholtz_rhs.compress(VectorOperation::add);
}

void Nonlinear_Time::solve_filter_system()
{
    TimerOutput::Scope t(computing_timer, "solve_filter_system");
    helmholtz_locally_solution = 0;
    SolverControl solver_control(1000, 1e-12 * helmholtz_rhs.l2_norm());
    PETScWrappers::SparseDirectMUMPS solver_filter(solver_control);
    solver_filter.solve(helmholtz_matrix,
                        helmholtz_locally_solution,
                        helmholtz_rhs);
    helmholtz_constraints.distribute(helmholtz_locally_solution);
    helmholtz_solution = helmholtz_locally_solution;
}

void Nonlinear_Time::generate_average_solution(PETScWrappers::MPI::Vector &output)
{
    TimerOutput::Scope t(computing_timer, "generate_average_solution");

    auto cell_begin_aver = dof_handler_filter.begin_active();
    auto cell_end_aver = dof_handler_filter.end();
    auto rho_begin_aver = dof_handler_simp.begin_active();
    //
    double average_i;
    std::vector<types::global_dof_index> local_dof_indices_aver(fe_filter.n_dofs_per_cell());
    std::vector<types::global_dof_index> local_rho_dof_indices_aver(1);

    for (; cell_begin_aver != cell_end_aver; ++cell_begin_aver, ++rho_begin_aver)
    {
        if (cell_begin_aver->is_locally_owned())
        {
            average_i = 0;
            cell_begin_aver->get_dof_indices(local_dof_indices_aver);
            rho_begin_aver->get_dof_indices(local_rho_dof_indices_aver);
            for (unsigned int i = 0; i < fe_filter.n_dofs_per_cell(); i++)
            {
                average_i += helmholtz_solution[local_dof_indices_aver[i]] / 
                             (double)fe_filter.n_dofs_per_cell();
            }
            output[local_rho_dof_indices_aver[0]] = average_i;
        }
    }
    output.compress(VectorOperation::insert);
}

void Nonlinear_Time::helmholtz_filter(PETScWrappers::MPI::Vector intput,
                                      PETScWrappers::MPI::Vector &output)
{
    TimerOutput::Scope t(computing_timer, "helmholtz_filter");
    assemble_filter_rhs(intput);
    solve_filter_system();
    generate_average_solution(output);
}

void Nonlinear_Time::get_cell_volume()
{
    TimerOutput::Scope t(computing_timer, "get_cell_volume");
    // 计算单元体积
    std::vector<types::global_dof_index> local_rho_dof_indices_constraint(1);
    for (auto & cell : dof_handler_simp.active_cell_iterators())
    {
        if (cell->is_locally_owned())
        {
            cell->get_dof_indices(local_rho_dof_indices_constraint);
            cell_volume[local_rho_dof_indices_constraint[0]] = cell->measure();  
        }
    }
    cell_volume.compress(VectorOperation::insert);
}