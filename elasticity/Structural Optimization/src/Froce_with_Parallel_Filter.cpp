#include "../include/Force_with_Parallel.h"

void ForceProblem::setup_filter_system()
{
  TimerOutput::Scope t(computing_timer, "setup_filter_system");
  // 过滤器
  for (auto &cell : dof_handler_filter.active_cell_iterators())
  {
    if (cell->material_id() == 1)
      cell->set_active_fe_index(0);
    else if (cell->material_id() == 2)
      cell->set_active_fe_index(1);
    else
      Assert(false, ExcNotImplemented());
  }  
  dof_handler_filter.distribute_dofs(fe_filter);
  locally_owned_dofs_filter = dof_handler_filter.locally_owned_dofs();
  locally_relevant_dofs_filter =
      DoFTools::extract_locally_relevant_dofs(dof_handler_filter);

  constraints_filter.clear();
  constraints_filter.reinit(locally_relevant_dofs_filter);
  DoFTools::make_hanging_node_constraints(dof_handler_filter, constraints_filter);
  constraints_filter.close();

  DynamicSparsityPattern dsp_filter(locally_relevant_dofs_filter);
  DoFTools::make_sparsity_pattern(dof_handler_filter,
                                  dsp_filter,
                                  constraints_filter,
                                  false);
  SparsityTools::distribute_sparsity_pattern(dsp_filter,
                                             locally_owned_dofs_filter,
                                             mpi_communicator,
                                             locally_relevant_dofs_filter);
  system_matrix_filter.reinit(locally_owned_dofs_filter,
                              locally_owned_dofs_filter,
                              dsp_filter,
                              mpi_communicator);
  locally_relevant_solution_filter.reinit(locally_owned_dofs_filter, locally_relevant_dofs_filter,
                                          mpi_communicator);
  system_rhs_filter.reinit(locally_owned_dofs_filter, mpi_communicator);
  completely_distributed_solution_f.reinit(locally_owned_dofs_filter, mpi_communicator);
}

void ForceProblem::assemble_filter_system()
{
  TimerOutput::Scope t(computing_timer, "assemble_filter_system");
  system_matrix_filter = 0;
  hp::FEValues<dim> hp_filter_fe_values(mapping_fe_filter,
                                        fe_filter,
                                        quadrature_formula_filter,
                                        update_values | update_gradients |
                                        update_quadrature_points | update_JxW_values);
  //
  auto cell_begin_filter = dof_handler_filter.begin_active();
  auto cell_end_filter = dof_handler_filter.end();
  //
  double Cell_Rmin = 0;
  // 
  unsigned int dofs_per_cell = cell_begin_filter->get_fe().n_dofs_per_cell();
  FullMatrix<double> cell_matrix;
  std::vector<types::global_dof_index> local_dof_indices_filter;  
  for (; cell_begin_filter != cell_end_filter; ++cell_begin_filter)
  {
    if (cell_begin_filter->is_locally_owned() && cell_begin_filter->material_id() == 1)
    {
      dofs_per_cell = cell_begin_filter->get_fe().n_dofs_per_cell();
      cell_matrix.reinit(dofs_per_cell,dofs_per_cell);
      local_dof_indices_filter.resize(dofs_per_cell);
      // 
      hp_filter_fe_values.reinit(cell_begin_filter);
      const auto & filter_fe_values = hp_filter_fe_values.get_present_fe_values();
      // 
      Cell_Rmin = 1.5 * std::pow(cell_begin_filter->measure()/(2.*sqrt(3)),1./(double)dim);
      for (unsigned int filter_points : filter_fe_values.quadrature_point_indices())
      {
        for (unsigned int filter_i : filter_fe_values.dof_indices())
        {
          for (unsigned int filter_j : filter_fe_values.dof_indices())
          {
            cell_matrix(filter_i, filter_j) +=
                (Cell_Rmin * Cell_Rmin *
                     filter_fe_values.shape_grad(filter_i, filter_points) *
                     filter_fe_values.shape_grad(filter_j, filter_points) +
                     filter_fe_values.shape_value(filter_i, filter_points) *
                     filter_fe_values.shape_value(filter_j, filter_points)) *
                     filter_fe_values.JxW(filter_points);
          }
        }
      }
      cell_begin_filter->get_dof_indices(local_dof_indices_filter);
      constraints_filter.distribute_local_to_global(cell_matrix,
                                                    local_dof_indices_filter,
                                                    system_matrix_filter);
    }
  }
  system_matrix_filter.compress(VectorOperation::add);
}

void ForceProblem::assemble_filter_rhs(LA::MPI::Vector Vec)
{
  TimerOutput::Scope t(computing_timer, "assemble_filter_rhs");
  // 记得清零
  system_rhs_filter = 0;
  hp::FEValues<dim> hp_fe_filter_values(mapping_fe_filter,
                                        fe_filter,
                                        quadrature_formula_filter,
                                        update_values | update_gradients |
                                        update_quadrature_points | update_JxW_values);
  hp::FEValues<dim> hp_fe_filter_values_rho(fe_rho,
                                            quadrature_formula_rho,
                                            update_values | update_quadrature_points);
  //
  auto cell_begin_filter = dof_handler_filter.begin_active();
  auto cell_end_filter = dof_handler_filter.end();
  auto base_begin_filter = dof_handler_rho.begin_active();
  // 
  unsigned int dofs_per_cell = cell_begin_filter->get_fe().n_dofs_per_cell();
  Vector<double> cell_rhs;
  std::vector<types::global_dof_index> local_dof_indices_filter;
  std::vector<double> rho_values_filter(1);
  //
  for (; cell_begin_filter != cell_end_filter; ++cell_begin_filter, ++base_begin_filter)
  {
    if (cell_begin_filter->is_locally_owned() && cell_begin_filter->material_id() == 1)
    {
      dofs_per_cell = cell_begin_filter->get_fe().n_dofs_per_cell();
      cell_rhs.reinit(dofs_per_cell);
      local_dof_indices_filter.resize(dofs_per_cell);
      // 
      hp_fe_filter_values.reinit(cell_begin_filter);
      hp_fe_filter_values_rho.reinit(base_begin_filter);
      // 
      const auto & fe_values_filter = hp_fe_filter_values.get_present_fe_values();
      const auto & fe_filter_rho = hp_fe_filter_values_rho.get_present_fe_values();
      fe_filter_rho.get_function_values(Vec, rho_values_filter);
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
                                                    system_rhs_filter);
    }
  }
  system_rhs_filter.compress(VectorOperation::add);
}

void ForceProblem::solve_filter()
{
  TimerOutput::Scope t(computing_timer, "solve_filter");
  completely_distributed_solution_f = 0;
  SolverControl solver_control_filter(dof_handler_filter.n_dofs(), 1e-9 * system_rhs_filter.l2_norm());
  PETScWrappers::SparseDirectMUMPS solver_dirt(solver_control_filter);
  solver_dirt.solve(system_matrix_filter,
                    completely_distributed_solution_f,
                    system_rhs_filter);
  constraints_filter.distribute(completely_distributed_solution_f);
  locally_relevant_solution_filter = completely_distributed_solution_f;
}

void ForceProblem::generate_average_vector(LA::MPI::Vector &out_average)
{
  TimerOutput::Scope t(computing_timer, "generate_average_vector");
  //
  auto cell_begin_aver = dof_handler_filter.begin_active();
  auto cell_end_aver = dof_handler_filter.end();
  auto rho_begin_aver = dof_handler_rho.begin_active();
  //
  double average_i;
  unsigned int dofs_per_cell = cell_begin_aver->get_fe().n_dofs_per_cell();
  std::vector<types::global_dof_index> local_dof_indices_aver(dofs_per_cell);
  std::vector<types::global_dof_index> local_rho_dof_indices_aver(1);
  for (; cell_begin_aver != cell_end_aver; ++cell_begin_aver, ++rho_begin_aver)
  {
    if (cell_begin_aver->is_locally_owned() && cell_begin_aver->material_id() == 1)
    {
      average_i = 0;
      dofs_per_cell = cell_begin_aver->get_fe().n_dofs_per_cell();
      local_dof_indices_aver.resize(dofs_per_cell);
      // 
      cell_begin_aver->get_dof_indices(local_dof_indices_aver);
      rho_begin_aver->get_dof_indices(local_rho_dof_indices_aver);
      for (unsigned int i = 0; i < dofs_per_cell; i++)
      {
        average_i += locally_relevant_solution_filter[local_dof_indices_aver[i]] / (double)dofs_per_cell;
      }
      out_average[local_rho_dof_indices_aver[0]] = average_i;
    }
  }
  out_average.compress(VectorOperation::insert);
}