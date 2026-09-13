#include "../include/Elastic_In_Hom.h"

void ElasticHomogenization::Make_Grid()
{
  GridGenerator::hyper_cube(triangulation,-1,1);
  for (auto & face : triangulation.active_face_iterators())
  {
    if (face->center()(0) == -1)
    {
      face->set_boundary_id(1);
    }
    if (face->center()(0) ==  1)
    {
      face->set_boundary_id(2);
    }
    if (face->center()(1) == -1)
    {
      face->set_boundary_id(3);
    }
    if (face->center()(1) ==  1)
    {
      face->set_boundary_id(4);
    }
  }
  Tensor<1, dim> offset;
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> periodicity_vector;

  GridTools::collect_periodic_faces(triangulation,
                                    1,
                                    2,
                                    0,
                                    periodicity_vector,
                                    offset);
  GridTools::collect_periodic_faces(triangulation,
                                    3,
                                    4,
                                    1,
                                    periodicity_vector,
                                    offset);
  triangulation.add_periodicity(periodicity_vector);
  triangulation.refine_global(init_refine_time);
}

void ElasticHomogenization::Init_Rho_System()
{
  dof_handler_simp.distribute_dofs(fe_simp);
  locally_owned_dofs_simp = dof_handler_simp.locally_owned_dofs();
  Simp_rho.reinit(locally_owned_dofs_simp, mpi_communicator);
  Simp_rho_Filter.reinit(locally_owned_dofs_simp, mpi_communicator);
  Simp_rho_Max.reinit(locally_owned_dofs_simp, mpi_communicator);
  Simp_rho_Min.reinit(locally_owned_dofs_simp, mpi_communicator);
  Cell_Volume.reinit(locally_owned_dofs_simp, mpi_communicator);
  Object_Function_Diff_Values.reinit(locally_owned_dofs_simp, mpi_communicator);
  Constriant_Function_Diff_Values.resize(1);
  Constriant_Function_Diff_Values[0].reinit(locally_owned_dofs_simp, mpi_communicator);
}
 
void ElasticHomogenization::Init_Elastic_System()
{
  dof_handler.distribute_dofs(fe);
  locally_owned_dofs = dof_handler.locally_owned_dofs();
  locally_relevant_dofs =
      DoFTools::extract_locally_relevant_dofs(dof_handler);

  constraints.clear();
  constraints.reinit(locally_relevant_dofs);
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  Tensor<1, dim> offset;
  std::vector<GridTools::PeriodicFacePair<typename DoFHandler<dim>::cell_iterator>> periodicity_vector;
  GridTools::collect_periodic_faces(dof_handler,
                                    1,
                                    2,
                                    0,
                                    periodicity_vector,
                                    offset);
  GridTools::collect_periodic_faces(dof_handler,
                                    3,
                                    4,
                                    1,
                                    periodicity_vector,
                                    offset);
  DoFTools::make_periodicity_constraints<dim, dim>(periodicity_vector,constraints);
  constraints.constrain_dof_to_zero(0);    
  constraints.constrain_dof_to_zero(1);    
  constraints.close();

  DynamicSparsityPattern dsp(locally_relevant_dofs);
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
  SparsityTools::distribute_sparsity_pattern(dsp,
                                             dof_handler.locally_owned_dofs(),
                                             mpi_communicator,
                                             locally_relevant_dofs);
  //
  elastic_matrix.reinit(locally_owned_dofs,
                        locally_owned_dofs,
                        dsp,
                        mpi_communicator);
  //
  unit_test_elastic.resize(dimepsilon);
  unit_test_rhs.resize(dimepsilon);
  //
  for (unsigned int i = 0; i < dimepsilon; i++)
  {
    unit_test_elastic[i].reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
    unit_test_rhs[i].reinit(locally_owned_dofs, mpi_communicator);
  }
  completely_distributed_solution.reinit(locally_owned_dofs, mpi_communicator);
}

void ElasticHomogenization::Assemble_Elastic_System()
{
  elastic_matrix = 0;
  for (unsigned int i = 0; i < dimepsilon; i++)
  {
    unit_test_rhs[i] = 0;
  }
  QGauss<dim> quadrature_formula(fe.degree + 1);
  QGauss<dim> quadrature_formula_rho(fe_simp.degree + 1);
  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);
  FEValues<dim> fe_values_rho(fe_simp,
                              quadrature_formula_rho,
                              update_values | update_quadrature_points);
  unsigned int dofs_per_cell = fe.n_dofs_per_cell();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  std::vector<double> rho_values(1);

  std::vector<Tensor<1, 3, double>> epsilon(dimepsilon);
  for (unsigned int i = 0; i < dimepsilon; i++)
  {
    epsilon[i][i] = 1.;
  }
  //
  auto cell_begin = dof_handler.begin_active();
  auto cell_end = dof_handler.end();
  auto rho_begin = dof_handler_simp.begin_active();
  //
  for (; cell_begin != cell_end; ++cell_begin, ++rho_begin)
  {
    if (cell_begin->is_locally_owned())
    {
      cell_matrix = 0;
      fe_values.reinit(cell_begin);
      fe_values_rho.reinit(rho_begin);
      fe_values_rho.get_function_values(Simp_rho_Filter, rho_values);
      const double EPenal = Emin + E * std::pow(rho_values[0],penal);
      const double mu = EPenal/(2*(1+possion));
      const double lambda = EPenal * possion/((1-2*possion)*(1+possion));

      const Tensor<2, 3> stress_strain_tensor = get_stress_strain_tensor(lambda,mu);

      for (const unsigned int q_index : fe_values.quadrature_point_indices())
      {
        const Tensor<2, 3> TransJacobian = Trans_Func.Generate_Invers_Trans(fe_values.quadrature_point(q_index));
        const Tensor<2, 3> TransJacobianT = Trans_Func.Generate_Invers_Trans_T(fe_values.quadrature_point(q_index));

        for (const unsigned int i : fe_values.dof_indices())
        {
          for (const unsigned int j : fe_values.dof_indices())
          {
            const Tensor<1, 3>
                      eps_phi_i = get_strain(fe_values, i, q_index),
                      eps_phi_j = get_strain(fe_values, j, q_index);

            cell_matrix(i, j) += (eps_phi_i *
                                  TransJacobianT *
                                  stress_strain_tensor *
                                  TransJacobian *
                                  eps_phi_j *
                                  fe_values.JxW(q_index));
          }
        }
      }
      cell_begin->get_dof_indices(local_dof_indices);

      for (unsigned int Index = 0; Index < dimepsilon; Index++)
      {
        cell_rhs = 0;
        for (const unsigned int q_index : fe_values.quadrature_point_indices())
        {
          const Tensor<2, 3> TransJacobianT = Trans_Func.Generate_Invers_Trans_T(fe_values.quadrature_point(q_index));
          for (const unsigned int i : fe_values.dof_indices())
          {
            const Tensor<1, 3>
                        eps_phi_i = get_strain(fe_values, i, q_index);
            cell_rhs(i) += (eps_phi_i *
                            TransJacobianT *
                            stress_strain_tensor *
                            epsilon[Index] *
                            fe_values.JxW(q_index));
          }
        }
        constraints.distribute_local_to_global(cell_rhs,
                                              local_dof_indices,
                                              unit_test_rhs[Index]);        
      }
      constraints.distribute_local_to_global(cell_matrix,
                                             local_dof_indices,
                                             elastic_matrix);
    }
  }
  elastic_matrix.compress(VectorOperation::add);
  for (unsigned int Index = 0; Index < dimepsilon; Index++)
  {
    unit_test_rhs[Index].compress(VectorOperation::add);
  }  
}

void ElasticHomogenization::Solve_Elastic_System()
{
  for (unsigned int Index = 0; Index < dimepsilon; Index++)
  {
    completely_distributed_solution = 0;
    SolverControl solve_control(dof_handler.n_dofs(), 1e-9 * unit_test_rhs[Index].l2_norm());
    PETScWrappers::SparseDirectMUMPS solver(solve_control,mpi_communicator);
    solver.set_symmetric_mode(true);
    solver.solve(elastic_matrix,completely_distributed_solution, unit_test_rhs[Index]);
    constraints.distribute(completely_distributed_solution);
    unit_test_elastic[Index] = completely_distributed_solution;
  }
}

void ElasticHomogenization::Generate_Homogenization()
{
  QGauss<dim> quadrature_formula(fe.degree + 1);
  QGauss<dim> quadrature_formula_rho(fe_simp.degree + 1);
  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);
  FEValues<dim> fe_values_rho(fe_simp,
                              quadrature_formula_rho,
                              update_values | update_quadrature_points);
  unsigned int dofs_per_cell = fe.n_dofs_per_cell();

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  std::vector<double> rho_values(1);
  // 
  Opt_Elastic_Conductivity_Matrix = 0;
  std::vector<Tensor<2,3,double>> Elastic_Tensor(fe_values.n_quadrature_points);
  std::vector<Tensor<2,3,double>> Elastic_Tensor_T(fe_values.n_quadrature_points);
  std::vector<std::vector<Tensor<1,dim>>> Tensor_Point(fe_values.n_quadrature_points);
  for (unsigned int index = 0; index < fe_values.n_quadrature_points; index++)
  {
    Tensor_Point[index].resize(dim);
  } 
  //
  auto cell_begin = dof_handler.begin_active();
  auto cell_end = dof_handler.end();
  auto rho_begin = dof_handler_simp.begin_active();
  //
  for (; cell_begin != cell_end; ++cell_begin, ++rho_begin)
  {
    if (cell_begin->is_locally_owned())
    {
      fe_values.reinit(cell_begin);
      fe_values_rho.reinit(rho_begin);
      fe_values_rho.get_function_values(Simp_rho_Filter, rho_values);

      const double EPenal = Emin + E * std::pow(rho_values[0],penal);
      const double mu = EPenal/(2*(1+possion));
      const double lambda = EPenal * possion/((1-2*possion)*(1+possion));

      const Tensor<2, 3> stress_strain_tensor = get_stress_strain_tensor(lambda,mu);
      //
      for (unsigned int index = 0; index < dimepsilon; index++)
      {
          fe_values.get_function_gradients(unit_test_elastic[index],Tensor_Point);
          for (unsigned int q_point : fe_values.quadrature_point_indices())
          {
              const Tensor<1,3> eps_phi_i = get_strain(Tensor_Point[q_point]);
              Elastic_Tensor[q_point][index] = eps_phi_i;
          }
      }
      for (unsigned int q_point : fe_values.quadrature_point_indices())
      {
        for (unsigned int i = 0; i < 3; i ++)
        {
          for (unsigned int j = 0; j < 3; j++)
          {
            Elastic_Tensor_T[q_point][i][j] = Elastic_Tensor[q_point][j][i];
          }
        }
      }
      // 
      for (unsigned int q_point : fe_values.quadrature_point_indices())
      {
          const Tensor<2, 3> TransJacobian = Trans_Func.Generate_Invers_Trans(fe_values.quadrature_point(q_point));
          const Tensor<2, 3> TransJacobianT = Trans_Func.Generate_Invers_Trans_T(fe_values.quadrature_point(q_point));

          Opt_Elastic_Conductivity_Matrix +=(stress_strain_tensor - 
                                              Elastic_Tensor_T[q_point] * TransJacobianT * stress_strain_tensor -
                                              stress_strain_tensor * TransJacobian * Elastic_Tensor[q_point] +
                                              Elastic_Tensor_T[q_point] * TransJacobianT * 
                                              stress_strain_tensor *
                                              TransJacobian * Elastic_Tensor[q_point]) * 
                                              fe_values.JxW(q_point);
      }
    }
  }
  Opt_Elastic_Conductivity_Matrix = Utilities::MPI::sum(Opt_Elastic_Conductivity_Matrix,mpi_communicator);
  Opt_Elastic_Conductivity_Matrix /= Cell_Volume.l1_norm();
}

void ElasticHomogenization::Generate_Object_Diff()
{
  Object_Function_Diff_Values = 0;
  
  QGauss<dim> quadrature_formula(fe.degree + 1);
  QGauss<dim> quadrature_formula_rho(fe_simp.degree + 1);
  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);
  FEValues<dim> fe_values_rho(fe_simp,
                              quadrature_formula_rho,
                              update_values | update_quadrature_points);
  unsigned int dofs_per_cell = fe.n_dofs_per_cell();

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  std::vector<types::global_dof_index> local_simp_dof_indices(fe_simp.n_dofs_per_cell());
  std::vector<double> rho_values(1);
  // 
  std::vector<Tensor<2,3,double>> Elastic_Tensor(fe_values.n_quadrature_points);
  std::vector<Tensor<2,3,double>> Elastic_Tensor_T(fe_values.n_quadrature_points);
  std::vector<std::vector<Tensor<1,dim>>> Tensor_Point(fe_values.n_quadrature_points);

  Tensor<2,3,double> Object_Discrete_Derivative_Cell;

  for (unsigned int index = 0; index < fe_values.n_quadrature_points; index++)
  {
    Tensor_Point[index].resize(dim);
  } 
  //
  auto cell_begin = dof_handler.begin_active();
  auto cell_end = dof_handler.end();
  auto rho_begin = dof_handler_simp.begin_active();
  //
  for (; cell_begin != cell_end; ++cell_begin, ++rho_begin)
  {
    if (cell_begin->is_locally_owned())
    {
      fe_values.reinit(cell_begin);
      fe_values_rho.reinit(rho_begin);
      fe_values_rho.get_function_values(Simp_rho_Filter, rho_values);
      const double EPenal = E * std::pow(rho_values[0],penal) * penal;
      const double mu = EPenal/(2*(1+possion));
      const double lambda = EPenal * possion/((1-2*possion)*(1+possion));

      const Tensor<2, 3> stress_strain_tensor = get_stress_strain_tensor(lambda,mu);
      //
      for (unsigned int index = 0; index < dimepsilon; index++)
      {
          fe_values.get_function_gradients(unit_test_elastic[index],Tensor_Point);
          for (unsigned int q_point : fe_values.quadrature_point_indices())
          {
              const Tensor<1,3> eps_phi_i = get_strain(Tensor_Point[q_point]);
              Elastic_Tensor[q_point][index] = eps_phi_i;
          }
      }
      for (unsigned int q_point : fe_values.quadrature_point_indices())
      {
        for (unsigned int i = 0; i < 3; i ++)
        {
          for (unsigned int j = 0; j < 3; j++)
          {
            Elastic_Tensor_T[q_point][i][j] = Elastic_Tensor[q_point][j][i];
          }
        }
      }
      // 
      Object_Discrete_Derivative_Cell = 0;
      for (unsigned int q_point : fe_values.quadrature_point_indices())
      {
        const Tensor<2, 3> TransJacobian = Trans_Func.Generate_Invers_Trans(fe_values.quadrature_point(q_point));
        const Tensor<2, 3> TransJacobianT = Trans_Func.Generate_Invers_Trans_T(fe_values.quadrature_point(q_point));

        Object_Discrete_Derivative_Cell += (stress_strain_tensor - 
                                              Elastic_Tensor_T[q_point] * TransJacobianT * stress_strain_tensor -
                                              stress_strain_tensor * TransJacobian * Elastic_Tensor[q_point] +
                                              Elastic_Tensor_T[q_point] * TransJacobianT * 
                                              stress_strain_tensor *
                                              TransJacobian * Elastic_Tensor[q_point]) * 
                                              fe_values.JxW(q_point);
      }
      // 
      rho_begin->get_dof_indices(local_simp_dof_indices);
      Object_Function_Diff_Values[local_simp_dof_indices[0]] += 
                              Object_Discrete_Derivative_Cell[0][1];                        
      // 
    }
  }
  Object_Function_Diff_Values.compress(VectorOperation::add);
  Object_Function_Diff_Values.operator/=(Cell_Volume.l1_norm());
}

void ElasticHomogenization::Generate_Cell_Volume()
{
  auto rho_begin = dof_handler_simp.begin_active();
  auto rho_end = dof_handler_simp.end();
  //
  std::vector<types::global_dof_index> local_rho_dof_indices(1);
  for (; rho_begin != rho_end; ++rho_begin)
  {
    if (rho_begin->is_locally_owned())
    {
      rho_begin->get_dof_indices(local_rho_dof_indices);
      Cell_Volume[local_rho_dof_indices[0]] = rho_begin->measure();
    }
  }
  Cell_Volume.compress(VectorOperation::insert);
}

void ElasticHomogenization::Output_Results()
{
  GridTools::transform(Trans_Func,triangulation);
  DataOut<dim> data_out_rho;
  data_out_rho.attach_dof_handler(dof_handler_simp);
  data_out_rho.add_data_vector(dof_handler_simp,Simp_rho,"Simp_rho");
  data_out_rho.add_data_vector(dof_handler_simp,Simp_rho_Filter,"Simp_rho_Filter");
  data_out_rho.add_data_vector(dof_handler_simp,Object_Function_Diff_Values,"Object_Function_Diff_Values");
  data_out_rho.add_data_vector(dof_handler_simp,Constriant_Function_Diff_Values[0],"Constriant_Function_Diff_Values");
  data_out_rho.build_patches();
  data_out_rho.write_vtu_in_parallel("./Optimization/SimpOut-"+ Utilities::to_string(MMA_Solver.Loop_Iter) +".vtu",mpi_communicator);
  // 
  // DataOut<dim> data_out;
  // data_out.attach_dof_handler(dof_handler);
  // std::vector<std::vector<std::string>> solution_names(3);

  // solution_names[0].emplace_back("xx_x_displacement");
  // solution_names[0].emplace_back("xx_y_displacement");
  
  // solution_names[1].emplace_back("yy_x_displacement");
  // solution_names[1].emplace_back("yy_y_displacement");

  // solution_names[2].emplace_back("xy_x_displacement");
  // solution_names[2].emplace_back("xy_y_displacement");
  
  // data_out.add_data_vector(unit_test_elastic[0], solution_names[0]);
  // data_out.add_data_vector(unit_test_elastic[1], solution_names[1]);
  // data_out.add_data_vector(unit_test_elastic[2], solution_names[2]);
  // data_out.build_patches();
  // data_out.write_vtu_in_parallel("./Phy/Dis-"+ Utilities::to_string(MMA_Solver.Loop_Iter) +".vtu",mpi_communicator);
}