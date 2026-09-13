#include "../include/Force_with_Parallel.h"

void ForceProblem::make_grid()
{
  TimerOutput::Scope t(computing_timer, "make_grid");
  GridGenerator::hyper_rectangle(triangulation,{0,0}, {2,1});
  for (auto &face : triangulation.active_face_iterators())
  {
    if (face->center()(0) == 0.)
    {
      face->set_boundary_id(1);
    }
  }
  triangulation.refine_global(init_refine_times);
  for (auto &face : triangulation.active_face_iterators())
  {
    if (std::fabs(face->center()(1) - 0.) < 1e-10)
    {
      if (std::fabs(face->center()(0) - (2.)) < 0.1)
      {
        face->set_boundary_id(2);
      }
    }
  }
  for (auto &cell :triangulation.active_cell_iterators())
  {
    if (cell->is_locally_owned())
    {
      cell->set_material_id(1);
    }
  }
}

void ForceProblem::setup_system()
{
  TimerOutput::Scope t(computing_timer, "setup_system");
  // 力学有限元
	for (auto &cell : dof_handler.active_cell_iterators())
	{
		if (cell->is_locally_owned())
		{
			if (cell->material_id() == 1)
      {
				cell->set_active_fe_index(0);        
      }
			else if (cell->material_id() == 2)
      {
				cell->set_active_fe_index(1);    
      }
			else
				Assert(false, ExcNotImplemented());			
		}
	}
  // 
  dof_handler.distribute_dofs(fe);
  
  locally_owned_dofs = dof_handler.locally_owned_dofs();
  locally_relevant_dofs =
      DoFTools::extract_locally_relevant_dofs(dof_handler);

  locally_relevant_solution.reinit(locally_owned_dofs,
                                   locally_relevant_dofs,
                                   mpi_communicator);

  system_rhs.reinit(locally_owned_dofs, mpi_communicator);

  constraints.clear();
  constraints.reinit(locally_owned_dofs,locally_relevant_dofs);
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  VectorTools::interpolate_boundary_values(dof_handler,
                                           1,
                                           Functions::ZeroFunction<dim>(dim),
                                           constraints);
  constraints.close();

  DynamicSparsityPattern dsp(locally_relevant_dofs);
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
  SparsityTools::distribute_sparsity_pattern(dsp,
                                             dof_handler.locally_owned_dofs(),
                                             mpi_communicator,
                                             locally_relevant_dofs);
  system_matrix.reinit(locally_owned_dofs,
                       locally_owned_dofs,
                       dsp,
                       mpi_communicator);
  completely_distributed_solution.reinit(locally_owned_dofs, mpi_communicator);
}

void ForceProblem::assemble_system()
{
  TimerOutput::Scope t(computing_timer, "assemble_system");
  // 位移场刚度矩阵进行重新初始化
  system_matrix = 0;
  hp::FEValues<dim> hp_fe_values(mapping_fe,
                                 fe,
                                 quadrature_formula,
                                 update_values | update_gradients |
                                 update_quadrature_points | update_JxW_values);
  hp::FEValues<dim> hp_fe_values_rho(fe_rho,
                                     quadrature_formula_rho,
                                     update_values | update_quadrature_points);
  //
  auto cell_begin = dof_handler.begin_active();
  auto cell_end = dof_handler.end();
  auto rho_begin = dof_handler_rho.begin_active();
  // 
  double E_with_penal;
  double lambda;
  double mu;
  //
  unsigned int dofs_per_cell = cell_begin->get_fe().n_dofs_per_cell();
  FullMatrix<double> cell_matrix;
  std::vector<types::global_dof_index> local_dof_indices;
  std::vector<double> rho_values(1);  
  for (; cell_begin != cell_end; ++cell_begin, ++rho_begin)
  {
    if (cell_begin->is_locally_owned() && cell_begin->material_id() == 1)
    {
      dofs_per_cell = cell_begin->get_fe().n_dofs_per_cell();
      cell_matrix.reinit(dofs_per_cell,dofs_per_cell);
      local_dof_indices.resize(dofs_per_cell);
      // 
      hp_fe_values.reinit(cell_begin);
      hp_fe_values_rho.reinit(rho_begin);
      // 
      const auto & fe_values = hp_fe_values.get_present_fe_values();
      const auto & fe_values_rho = hp_fe_values_rho.get_present_fe_values();
      // 
      fe_values_rho.get_function_values(Simp_Rho_Filted, rho_values);
      E_with_penal = E_all * epsimin + E_all * (1. - epsimin) * std::pow(rho_values[0], penal);
      lambda = E_with_penal * possion / ((1 + possion) * (1 - 2 * possion));
      mu = E_with_penal / (2 * (1 + possion));
      for (const unsigned int i : fe_values.dof_indices())
      {
        const unsigned int component_i = cell_begin->get_fe().system_to_component_index(i).first;
        for (const unsigned int j : fe_values.dof_indices())
        {
          const unsigned int component_j = cell_begin->get_fe().system_to_component_index(j).first;
          for (const unsigned int q_point : fe_values.quadrature_point_indices())
          {
            cell_matrix(i, j) +=
                ((fe_values.shape_grad(i, q_point)[component_i] *
                  fe_values.shape_grad(j, q_point)[component_j] *
                  lambda) +
                 (fe_values.shape_grad(i, q_point)[component_j] *
                  fe_values.shape_grad(j, q_point)[component_i] *
                  mu) +
                 ((component_i == component_j) ? (fe_values.shape_grad(i, q_point) *
                                                  fe_values.shape_grad(j, q_point) *
                                                  mu)
                                               : 0)) *
                fe_values.JxW(q_point);
          }
        }
      }
      cell_begin->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(cell_matrix,
                                             local_dof_indices,
                                             system_matrix);
    }
  }
  system_matrix.compress(VectorOperation::add);
}

void ForceProblem::assemble_system_rhs()
{
  TimerOutput::Scope t(computing_timer, "assemble_system_rhs");
  system_rhs = 0;
  hp::FEFaceValues<dim> hp_fe_face_values(fe,
                                          quadrature_formula_face,
                                          update_values | update_gradients |
                                          update_quadrature_points | update_JxW_values);
  hp::FEValues<dim> hp_fe_values(mapping_fe,
                                 fe,
                                 quadrature_formula,
                                 update_values | update_gradients |
                                 update_quadrature_points | update_JxW_values);
  // 力学载荷
  Vector<double> force(dim);
  force[1] = 1e6;
  //
  auto cell_begin = dof_handler.begin_active();
  auto cell_end = dof_handler.end();
  //
  unsigned int dofs_per_cell = cell_begin->get_fe().n_dofs_per_cell();
  Vector<double> cell_rhs;
  std::vector<types::global_dof_index> local_dof_indices;
  // 
  for (; cell_begin != cell_end; ++cell_begin)
  {
    if (cell_begin->is_locally_owned() && cell_begin->material_id() == 1)
    {
      dofs_per_cell = cell_begin->get_fe().n_dofs_per_cell();
      cell_rhs.reinit(dofs_per_cell);
      local_dof_indices.resize(dofs_per_cell);
      // 
      hp_fe_values.reinit(cell_begin);
      const auto & fe_values = hp_fe_values.get_present_fe_values();
      // 
      for (auto &cell_face : cell_begin->face_iterators())
      {
        if (cell_face->at_boundary() && cell_face->boundary_id() == 2)
        {
          hp_fe_face_values.reinit(cell_begin, cell_face,0,0,0);
          const auto & fe_face_values = hp_fe_face_values.get_present_fe_values();
          for (const unsigned int i : fe_values.dof_indices())
          {
            const unsigned int component_i = cell_begin->get_fe().system_to_component_index(i).first;
            for (const unsigned int q_point : fe_face_values.quadrature_point_indices())
            {
              cell_rhs(i) += (fe_face_values.shape_value(i, q_point) *
                              force[component_i] *
                              fe_face_values.JxW(q_point));
            }
          }
        }
      }
      cell_begin->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(cell_rhs,
                                             local_dof_indices,
                                             system_rhs);
    }
  }
  system_rhs.compress(VectorOperation::add);
}

void ForceProblem::solve()
{
  TimerOutput::Scope t(computing_timer, "solve");
  completely_distributed_solution = 0;
  SolverControl solver_control(dof_handler.n_dofs(), 1e-12 * system_rhs.l2_norm());
  PETScWrappers::SparseDirectMUMPS solver(solver_control);
  solver.solve(system_matrix,
               completely_distributed_solution,
               system_rhs);
  constraints.distribute(completely_distributed_solution);
  locally_relevant_solution = completely_distributed_solution;
}

void ForceProblem::get_object_diff_values(LA::MPI::Vector &Object_Diff_Values)
{
  TimerOutput::Scope t(computing_timer, "get_object_diff_values");
  hp::FEValues<dim> hp_fe_values(mapping_fe,
                                 fe,
                                 quadrature_formula,
                                 update_values | update_gradients |
                                 update_quadrature_points | update_JxW_values);
  hp::FEValues<dim> hp_fe_values_rho(fe_rho,
                                     quadrature_formula_rho,
                                     update_values | update_quadrature_points);
  //
  auto cell_begin = dof_handler.begin_active();
  auto cell_end = dof_handler.end();
  auto rho_begin = dof_handler_rho.begin_active();
  //
  double c_i;
  unsigned int dofs_per_cell = cell_begin->get_fe().n_dofs_per_cell();
  FullMatrix<double> cell_matrix;
  Vector<double> cell_solution;
  std::vector<types::global_dof_index> local_dof_indices;
  std::vector<types::global_dof_index> local_dof_indices_rho(1);
  std::vector<double> rho_values(1);
  // 
  for (; cell_begin != cell_end; ++cell_begin, ++rho_begin)
  {
    if (cell_begin->is_locally_owned() && cell_begin->material_id() == 1)
    {
      c_i = 0;
      dofs_per_cell = cell_begin->get_fe().n_dofs_per_cell();
      cell_solution.reinit(dofs_per_cell);
      cell_matrix.reinit(dofs_per_cell,dofs_per_cell);
      local_dof_indices.resize(dofs_per_cell);
      // 
      hp_fe_values.reinit(cell_begin);
      hp_fe_values_rho.reinit(rho_begin);
      // 
      const auto & fe_values = hp_fe_values.get_present_fe_values();
      const auto & fe_values_rho = hp_fe_values_rho.get_present_fe_values();
      // 
      fe_values_rho.get_function_values(Simp_Rho_Filted, rho_values);
      double dlambda_dE = possion / ((1 + possion) * (1 - 2 * possion)) *
                          penal * E_all * (1. - epsimin) * std::pow(rho_values[0], penal - 1);
      double dmu_dE = 1 / (2 * (1 + possion)) *
                      penal * E_all * (1. - epsimin) * std::pow(rho_values[0], penal - 1);
      for (const unsigned int i : fe_values.dof_indices())
      {
        const unsigned int component_i =  cell_begin->get_fe().system_to_component_index(i).first;
        for (const unsigned int j : fe_values.dof_indices())
        {
          const unsigned int component_j =  cell_begin->get_fe().system_to_component_index(j).first;
          for (const unsigned int q_point :
               fe_values.quadrature_point_indices())
          {
            cell_matrix(i, j) +=
                ((fe_values.shape_grad(i, q_point)[component_i] *
                  fe_values.shape_grad(j, q_point)[component_j] *
                  dlambda_dE) +
                 (fe_values.shape_grad(i, q_point)[component_j] *
                  fe_values.shape_grad(j, q_point)[component_i] *
                  dmu_dE) +
                 ((component_i == component_j) ? (fe_values.shape_grad(i, q_point) *
                                                  fe_values.shape_grad(j, q_point) *
                                                  dmu_dE)
                                               : 0)) *
                fe_values.JxW(q_point);
          }
        }
      }
      cell_begin->get_dof_indices(local_dof_indices);
      rho_begin->get_dof_indices(local_dof_indices_rho);
      for (const unsigned int i : fe_values.dof_indices())
      {
        cell_solution[i] = locally_relevant_solution[local_dof_indices[i]];
      }
      for (const unsigned int i : fe_values.dof_indices())
      {
        for (const unsigned int j : fe_values.dof_indices())
        {
          c_i += cell_solution[i] * cell_matrix(i, j) * cell_solution[j];
        }
      }
      Object_Diff_Values[local_dof_indices_rho[0]] = -c_i;
    }
  }
  Object_Diff_Values.compress(VectorOperation::insert);
}

void ForceProblem::output_results(unsigned int loop)
{
  TimerOutput::Scope t(computing_timer, "output_results");
  
  // DataOut<dim> data_out;
  // data_out.set_cell_selection(
  //   [](const typename parallel::distributed::Triangulation<dim>::cell_iterator &cell) {
  //       return (cell->is_active() && cell->material_id() == 1);
  //   });
  // data_out.attach_dof_handler(dof_handler);
  // std::vector<std::string> solution_names;
  // switch (dim)
  // {
  // case 1:
  //   solution_names.emplace_back("displacement");
  //   break;
  // case 2:
  //   solution_names.emplace_back("x_displacement");
  //   solution_names.emplace_back("y_displacement");
  //   break;
  // case 3:
  //   solution_names.emplace_back("x_displacement");
  //   solution_names.emplace_back("y_displacement");
  //   solution_names.emplace_back("z_displacement");
  //   break;
  // default:
  //   Assert(false, ExcNotImplemented());
  // }
  // data_out.add_data_vector(locally_relevant_solution, solution_names);
  // data_out.build_patches(mapping_fe);
  // data_out.write_vtu_in_parallel("Solution-Displacement-" + Utilities::to_string(loop,1) + ".vtu", mpi_communicator);
  
  DataOut<dim> data_out_rho;
  data_out_rho.attach_dof_handler(dof_handler_rho);  
  data_out_rho.add_data_vector(dof_handler_rho, Simp_Rho, "Simp_Rho");
  data_out_rho.add_data_vector(dof_handler_rho, Simp_Rho_Filted, "Simp_Rho_Filted");
  data_out_rho.add_data_vector(dof_handler_rho, Object_Diff_Values, "Object_Diff_Values");
  data_out_rho.add_data_vector(dof_handler_rho, Constraint_Diff_Values[0], "Constraint_Diff_Values");
  data_out_rho.add_data_vector(dof_handler_rho, MMA_Solver.Moving_Low_Boundary, "Moving_Low_Boundary");
  data_out_rho.add_data_vector(dof_handler_rho, MMA_Solver.Moving_Upp_Boundary, "Moving_Upp_Boundary");
  data_out_rho.add_data_vector(dof_handler_rho, MMA_Solver.Old_Design_Variables[0], "Old_Design_Variables0");
  data_out_rho.add_data_vector(dof_handler_rho, MMA_Solver.Old_Design_Variables[1], "Old_Design_Variables1");
  data_out_rho.build_patches();
  data_out_rho.write_vtu_in_parallel("Simp-Out-"+ Utilities::to_string(loop,1) + ".vtu", mpi_communicator);
}

void ForceProblem::get_cell_volume()
{
  TimerOutput::Scope t(computing_timer, "get_cell_volume");
  std::vector<types::global_dof_index> local_dof_indices_rho(1);
  for (auto & cell_iter : dof_handler_rho.active_cell_iterators())
  {
    if (cell_iter->is_locally_owned())
    {
      cell_iter->get_dof_indices(local_dof_indices_rho);
      Cell_Volume[local_dof_indices_rho[0]] = cell_iter->measure();
    }
  }
  Cell_Volume.compress(VectorOperation::insert);
}

void ForceProblem::set_cell_material_id()
{
  TimerOutput::Scope t(computing_timer, "set_cell_material_id");
  std::vector<types::global_dof_index> local_dof_indices_rho(1);
  for (auto & cell_iter : dof_handler_rho.active_cell_iterators())
  {
    if (cell_iter->is_locally_owned())
    {
      cell_iter->get_dof_indices(local_dof_indices_rho);
      if (Simp_Rho[local_dof_indices_rho[0]] < 1e-5)
      {
        cell_iter->set_material_id(2);
      }else{
        cell_iter->set_material_id(1);
      }
    }
  }
}

void ForceProblem::set_cell_nothing_values(LA::MPI::Vector &Diff_Values)
{
  TimerOutput::Scope t(computing_timer, "set_cell_nothing_values");
  std::vector<types::global_dof_index> local_dof_indices_rho(1);
  for (auto & cell_iter : dof_handler_rho.active_cell_iterators())
  {
    if (cell_iter->is_locally_owned() && cell_iter->material_id() == 2)
    {
      cell_iter->get_dof_indices(local_dof_indices_rho);
      Diff_Values[local_dof_indices_rho[0]] = 0;
    }
  }
  Diff_Values.compress(VectorOperation::insert);
}

void ForceProblem::Refine_Grid()
{
	TimerOutput::Scope t(computing_timer, "Refine_Grid");

	Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
  const QGauss<dim - 1> face_quadrature_1(degree + 1);
  const QGauss<dim - 1> face_quadrature_2(degree + 1);

  hp::QCollection<dim - 1> face_q_collection;
  face_q_collection.push_back(face_quadrature_1);
  face_q_collection.push_back(face_quadrature_2);

  const FEValuesExtractors::Vector force(0);

	KellyErrorEstimator<dim>::estimate(dof_handler,
										                 face_q_collection,
										                 std::map<types::boundary_id, const Function<dim> *>(),
										                 locally_relevant_solution,
										                 estimated_error_per_cell,
                                     fe.component_mask(force));
	parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(triangulation, 
	                                                                       estimated_error_per_cell, 
                                                                         0.2, 
                                                                         0.1);
  if (triangulation.n_levels() > 9)
  {
      for (auto &cell : triangulation.active_cell_iterators_on_level(9))
      {    
          cell->clear_refine_flag();             
      }
  }
  // 细化网格后需要重新插值的向量
  const std::vector<const LA::MPI::Vector *> 
                grid_refine_vectors_in = {&Simp_Rho,
                                          &MMA_Solver.Moving_Low_Boundary,
                                          &MMA_Solver.Moving_Upp_Boundary,
                                          &MMA_Solver.Old_Design_Variables[0],
                                          &MMA_Solver.Old_Design_Variables[1],
                                          &Simp_Max,
                                          &Simp_Min};

  parallel::distributed::SolutionTransfer<dim,LA::MPI::Vector> solution_transfer(dof_handler_rho);

  triangulation.prepare_coarsening_and_refinement();    
  solution_transfer.prepare_for_coarsening_and_refinement(grid_refine_vectors_in);
  triangulation.execute_coarsening_and_refinement();

  // 
  init_simp();
  MMA_Solver.Deal_II_MMA_Reinit(Simp_Rho);

  pcout << "\n";
  pcout << "Number Of Refine Grid Cells : " << triangulation.n_global_active_cells() << std::endl;
  pcout << "Number Of Refine Grid Dofs : " << dof_handler.n_dofs() << std::endl;
  pcout << "\n";

  std::vector<LA::MPI::Vector *> grid_refine_vectors_out;
  // 7个需要在细化网格后插值的向量
  grid_refine_vectors_out = {&Simp_Rho,
                            &MMA_Solver.Moving_Low_Boundary,
                            &MMA_Solver.Moving_Upp_Boundary,
                            &MMA_Solver.Old_Design_Variables[0],
                            &MMA_Solver.Old_Design_Variables[1],                                          
                            &Simp_Max,
                            &Simp_Min};
  // 
  solution_transfer.interpolate(grid_refine_vectors_out);    
  // 细化完成后,网格材料信息完全消失
  set_cell_material_id();
  // 
  setup_system();
  setup_filter_system();	
  // 
  for (unsigned int i = 0; i < 7; i++)
  {
    set_cell_nothing_values(*grid_refine_vectors_out[i]);
  } 
}