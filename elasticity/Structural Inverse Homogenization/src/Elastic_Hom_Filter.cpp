#include "../include/Elastic_In_Hom.h"

void ElasticHomogenization::Init_Filter_System()
{
  // 过滤器
  dof_handler_filter.distribute_dofs(fe_filter);
  locally_owned_dofs_filter = dof_handler_filter.locally_owned_dofs();
  locally_relevant_dofs_filter =
      DoFTools::extract_locally_relevant_dofs(dof_handler_filter);

  constraints_filter.clear();
  constraints_filter.reinit(locally_owned_dofs_filter,locally_relevant_dofs_filter);
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
  completely_distributed_solution_filter.reinit(locally_owned_dofs_filter, mpi_communicator);
}

void ElasticHomogenization::Assemble_Filter_System()
{
  system_matrix_filter = 0;
  QGauss<dim> quadrature_formula_filter(fe_filter.degree + 1);
  FEValues<dim> filter_fe_values(fe_filter,
                                 quadrature_formula_filter,
                                 update_values | update_gradients |
                                     update_quadrature_points | update_JxW_values);
  unsigned int dofs_per_cell = fe_filter.n_dofs_per_cell();
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices_filter(dofs_per_cell);
  //
  auto cell_begin_filter = dof_handler_filter.begin_active();
  auto cell_end_filter = dof_handler_filter.end();
  double cell_Rmin = 0;
  //
  for (; cell_begin_filter != cell_end_filter; ++cell_begin_filter)
  {
    if (cell_begin_filter->is_locally_owned())
    {
      cell_matrix = 0;
      filter_fe_values.reinit(cell_begin_filter);
      cell_Rmin = 3 * std::pow(cell_begin_filter->measure(),1./(double)dim)/(2*std::sqrt(3.));
      for (unsigned int filter_points : filter_fe_values.quadrature_point_indices())
      {
        for (unsigned int filter_i : filter_fe_values.dof_indices())
        {
          for (unsigned int filter_j : filter_fe_values.dof_indices())
          {
            cell_matrix(filter_i, filter_j) +=
                (cell_Rmin * cell_Rmin *
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

void ElasticHomogenization::Assemble_Filter_Rhs(LA::MPI::Vector Vec)
{
  system_rhs_filter = 0;
  QGauss<dim> quadrature_formula_filter(fe_filter.degree + 1);
  QGauss<dim> quadrature_formula_simp(fe_simp.degree + 1);
  FEValues<dim> filter_fe_values(fe_filter,
                                 quadrature_formula_filter,
                                 update_values | update_gradients |
                                     update_quadrature_points | update_JxW_values);
  FEValues<dim> fe_values_simp(fe_simp,
                               quadrature_formula_simp,
                               update_values | update_quadrature_points);
  unsigned int dofs_per_cell = fe_filter.n_dofs_per_cell();
  Vector<double> cell_rhs(dofs_per_cell);
  std::vector<double> rho_values_filter(1);
  std::vector<types::global_dof_index> local_dof_indices_filter(dofs_per_cell);
  //
  auto cell_begin_filter = dof_handler_filter.begin_active();
  auto cell_end_filter = dof_handler_filter.end();
  auto simp_filter = dof_handler_simp.begin_active();
  //
  for (; cell_begin_filter != cell_end_filter; ++cell_begin_filter, ++simp_filter)
  {
    if (cell_begin_filter->is_locally_owned())
    {
      cell_rhs = 0;
      fe_values_simp.reinit(simp_filter);
      filter_fe_values.reinit(cell_begin_filter);
      fe_values_simp.get_function_values(Vec, rho_values_filter);
      for (unsigned int q_point : filter_fe_values.quadrature_point_indices())
      {
        for (unsigned int i : filter_fe_values.dof_indices())
        {
          cell_rhs(i) += (filter_fe_values.shape_value(i, q_point) *
                          rho_values_filter[0] *
                          filter_fe_values.JxW(q_point));
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

void ElasticHomogenization::Solve_Filter()
{
  completely_distributed_solution_filter = 0;
  SolverControl solver_control_filter(dof_handler_filter.n_dofs(), 1e-9 * system_rhs_filter.l2_norm());
  PETScWrappers::SparseDirectMUMPS solver_filter(solver_control_filter);
  solver_filter.set_symmetric_mode(true);
  solver_filter.solve(system_matrix_filter,
                      completely_distributed_solution_filter,
                      system_rhs_filter);
  constraints_filter.distribute(completely_distributed_solution_filter);
  locally_relevant_solution_filter = completely_distributed_solution_filter;
}

void ElasticHomogenization::Generate_Average_Vector(LA::MPI::Vector &out_average)
{
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
    if (cell_begin_aver->is_locally_owned())
    {
      average_i = 0;
      cell_begin_aver->get_dof_indices(local_dof_indices_aver);
      rho_begin_aver->get_dof_indices(local_rho_dof_indices_aver);
      for (unsigned int i = 0; i < fe_filter.n_dofs_per_cell(); i++)
      {
        average_i += locally_relevant_solution_filter[local_dof_indices_aver[i]] / (double)fe_filter.n_dofs_per_cell();
      }
      out_average[local_rho_dof_indices_aver[0]] = average_i;
    }
  }
  out_average.compress(VectorOperation::insert);
}