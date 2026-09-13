#include "../include/Nonlinear_Time.h"

void Nonlinear_Time::setup_adjoint_system()
{
    TimerOutput::Scope t(computing_timer, "setup_adjoint_system");
    dof_handler_simp.distribute_dofs(fe_simp);
    pcout<<"dof simp : "<<dof_handler_simp.n_dofs()<<std::endl;

    locally_owned_dofs_simp = dof_handler_simp.locally_owned_dofs();
    locally_relevant_dofs_simp =
    DoFTools::extract_locally_relevant_dofs(dof_handler_simp);
    
    DynamicSparsityPattern dsp_DF_DP(dof_handler.n_dofs(),
                                     dof_handler_simp.n_dofs(),
                                     locally_relevant_dofs);   
    DoFTools::make_sparsity_pattern(dof_handler,
                                    dof_handler_simp,
                                    dsp_DF_DP);                      
    SparsityTools::distribute_sparsity_pattern(dsp_DF_DP,
                                               locally_owned_dofs,
                                               mpi_communicator,
                                               locally_relevant_dofs);
    jacobian_matrix_dFdP.reinit(locally_owned_dofs,
                                locally_owned_dofs_simp,
                                dsp_DF_DP,
                                mpi_communicator);
    // 
    IndexSet col_one(1); col_one.add_index(0);
    DynamicSparsityPattern dsp_DR_DP(dof_handler_simp.n_dofs(),
                                     1,
                                     locally_relevant_dofs_simp);                      
    SparsityTools::distribute_sparsity_pattern(dsp_DR_DP,
                                               locally_owned_dofs_simp,
                                               mpi_communicator,
                                               locally_relevant_dofs_simp);
    jacobian_matrix_dRdP.reinit(locally_owned_dofs_simp,
                                col_one,
                                dsp_DR_DP,
                                mpi_communicator);
    MatSetType(jacobian_matrix_dRdP, MATMPIDENSE);
    //
    adjoint_mu.reinit(locally_owned_dofs_simp,mpi_communicator);

    simp_rho.reinit(locally_owned_dofs_simp,mpi_communicator);
    simp_rho_filter.reinit(locally_owned_dofs_simp,mpi_communicator);
    simp_rho_max.reinit(locally_owned_dofs_simp,mpi_communicator);
    simp_rho_min.reinit(locally_owned_dofs_simp,mpi_communicator);

    cell_volume.reinit(locally_owned_dofs_simp,mpi_communicator);
    Object_Function_Derivative.reinit(locally_owned_dofs_simp,mpi_communicator);
    Constraint_Function_Derivative.resize(1);
    Constraint_Function_Derivative[0].reinit(locally_owned_dofs_simp,mpi_communicator);
}

void Nonlinear_Time::initialize_adjoint_vector()
{
    TimerOutput::Scope t(computing_timer, "initialize_adjoint_vector");
    adjoint_lambda = 1./(double)dof_handler.n_dofs();
    adjoint_mu = 0;
}

void Nonlinear_Time::assemble_implicit_jacobian_p(const double time,
                                                  const PETScWrappers::MPI::Vector &solution,
                                                  const PETScWrappers::MPI::Vector &solution_dot)
{
    TimerOutput::Scope t(computing_timer, "assemble_implicit_jacobian_p");

    tmp_solution = solution;
    tmp_solution_dot = solution_dot;
    
    current_constraints.distribute(tmp_solution);
    homogeneous_constraints.distribute(tmp_solution_dot);

    locally_relevant_solution = tmp_solution;
    locally_relevant_solution_dot = tmp_solution_dot;

    jacobian_matrix_dFdP = 0;

    QGauss<dim> quadrature_formula(fe.degree + 1);
    FEValues<dim> fe_values(fe, quadrature_formula,
                            update_values | update_gradients 
                            | update_JxW_values | update_quadrature_points);
    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points = quadrature_formula.size();

    QGauss<dim> quadrature_formula_simp(fe_simp.degree + 1);
    FEValues<dim> fe_values_simp(fe_simp, quadrature_formula_simp,
                                 update_values | update_quadrature_points);

    Vector<double> cell_jacobian_dfdp(dofs_per_cell);

    std::vector<double>        solution_values(n_q_points);
    std::vector<Tensor<1,dim>> solution_gradients(n_q_points);
    std::vector<double>        solution_dot_values(n_q_points);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices_simp(1);

    std::vector<double>        simp_rho_values(1);

    double cell_heat_capacity = 0;
    double cell_density = 0;

    double cell_conbine_value = 0;
    double cell_conbine_value_drho = 0;
    double cell_muti_value_drho = 0;
    double cell_muti_grad_value_drho = 0;

    double cell_heat_conductivity_drho = 0;
    double cell_heat_capacity_drho = 0;
    double cell_density_drho = density[1] - density[0];

    auto cell = dof_handler.begin_active();
    auto cell_simp = dof_handler_simp.begin_active();
    auto cell_end = dof_handler.end();

    for (;cell != cell_end; cell++, cell_simp++)
    {        
        if (cell->is_locally_owned())
        {
            cell_jacobian_dfdp = 0;

            fe_values.reinit(cell);
            fe_values.get_function_values(locally_relevant_solution,solution_values);
            fe_values.get_function_gradients(locally_relevant_solution,
                                             solution_gradients);
            fe_values.get_function_values(locally_relevant_solution_dot,
                                          solution_dot_values);

            fe_values_simp.reinit(cell_simp);
            fe_values_simp.get_function_values(simp_rho,
                                               simp_rho_values);

            cell_density = density[1] * simp_rho_values[0] + (1 - simp_rho_values[0]) * density[0];

            cell_heat_conductivity_drho = (heat_conductivity[1] - heat_conductivity[0]) *
                                           MySimp.simp_grad_value(simp_rho_values[0]);

            cell_conbine_value = heat_capacity[1] * simp_rho_values[0] * density[1]/cell_density;

            cell_conbine_value_drho = heat_capacity[1] * density[1] *
                                (cell_density - simp_rho_values[0] * cell_density_drho)/
                                 (cell_density * cell_density);

            cell_muti_value_drho = - density[0] * 
                                (cell_density + (1 - simp_rho_values[0]) * cell_density_drho)/
                                (cell_density * cell_density);
            
            cell_muti_grad_value_drho = (1 - simp_rho_values[0]) * density[0] / cell_density;
            
            for (unsigned int q = 0; q < n_q_points; ++q)
            {
                cell_heat_capacity = MyPcm.cp_t_x_value(solution_values[q],simp_rho_values[0]);

                cell_heat_capacity_drho = cell_conbine_value_drho + 
                                          cell_muti_value_drho * cell_heat_capacity +
                                          cell_muti_grad_value_drho * MyPcm.dcp_t_dx_value(solution_values[q],simp_rho_values[0]);
                                          
                cell_heat_capacity = cell_conbine_value + cell_heat_capacity * cell_muti_grad_value_drho;

                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                    cell_jacobian_dfdp(i) += (fe_values.shape_grad(i, q) *
                                                cell_heat_conductivity_drho *
                                                solution_gradients[q] 
                                              +
                                                fe_values.shape_value(i, q) *
                                                cell_heat_capacity_drho *
                                                cell_density *
                                                solution_dot_values[q]
                                              + 
                                                fe_values.shape_value(i, q) *
                                                cell_heat_capacity *
                                                cell_density_drho *
                                                solution_dot_values[q]
                                             ) * 
                                             fe_values.JxW(q);
                }
            }

            cell->get_dof_indices(local_dof_indices);
            cell_simp->get_dof_indices(local_dof_indices_simp);
            
            for (unsigned int i = 0; i < dofs_per_cell; i++)
            {
                if (!current_constraints.is_constrained(local_dof_indices[i]))
                {
                jacobian_matrix_dFdP.add(local_dof_indices[i],
                                         local_dof_indices_simp[0],
                                         cell_jacobian_dfdp(i));
                }
            }
        }
    }
    jacobian_matrix_dFdP.compress(VectorOperation::add);
}

void Nonlinear_Time::integral_funtion(const PETScWrappers::MPI::Vector &solution,
                                      PETScWrappers::MPI::Vector &integral)
{
    TimerOutput::Scope t(computing_timer, "integral_funtion");
    integral[0] = 0;
    integral.compress(VectorOperation::insert);
}

void Nonlinear_Time::assemble_integral_jacobian_p()
{
    TimerOutput::Scope t(computing_timer, "assemble_integral_jacobian_p");
    jacobian_matrix_dRdP = 0;
}

void Nonlinear_Time::assemble_integral_jacobian_u(const PETScWrappers::MPI::Vector &solution)
{
    TimerOutput::Scope t(computing_timer, "assemble_integral_jacobian_u");
    jacobian_matrix_dRdU = 0;
}

void Nonlinear_Time::output_adjoint_results(const double time, const unsigned int step)
{
    TimerOutput::Scope t(computing_timer, "output_adjoint_results");
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler_simp);
    data_out.add_data_vector(dof_handler_simp, adjoint_mu, "adjoint_mu");
    data_out.add_data_vector(dof_handler_simp, simp_rho, "simp_rho");
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(adjoint_lambda, "adjoint_lambda");
    data_out.build_patches();
    std::string filename = "adjoint_sensitivity_" + Utilities::int_to_string(step, 4) + ".vtu";
    data_out.write_vtu_in_parallel(filename, mpi_communicator);
}

void Nonlinear_Time::setup_adjoint_solver()
{
    TimerOutput::Scope t(computing_timer, "setup_adjoint_solver");

    /* setup dfdp */ 
    const auto ts_ijacobian_p = 
               [](TS ts, PetscReal t, Vec y, Vec y_dot, PetscReal alpha, Mat A, void *ctx) 
               -> PetscErrorCode {
                    PetscFunctionBegin;
                    auto user = static_cast<Nonlinear_Time *>(ctx);
                    user->pcout<<"implicit_jacobian_p"<<std::endl;
                    PETScWrappers::MPI::Vector ydealii(y);
                    PETScWrappers::MPI::Vector ydotdealii(y_dot);
                    user->assemble_implicit_jacobian_p(t, ydealii, ydotdealii);
                    PetscFunctionReturn(PETSC_SUCCESS);
                };
    TSSetIJacobianP(time_stepper.petsc_ts(),
                    jacobian_matrix_dFdP.petsc_matrix(),
                    ts_ijacobian_p,
                    this);
    /* create quadts */ 
    TSCreateQuadratureTS(time_stepper.petsc_ts(), PETSC_TRUE, &quadts);   
    /* setup r */
    const auto ts_integral_funtion = 
                [](TS ts, PetscReal t, Vec y, Vec intergal, void *ctx) 
                -> PetscErrorCode {
                    PetscFunctionBegin;
                    auto user = static_cast<Nonlinear_Time *>(ctx);
                    PETScWrappers::MPI::Vector ydealii(y);
                    PETScWrappers::MPI::Vector intergaldealii(intergal);
                    user->pcout<<"integral_funtion"<<std::endl;
                    user->integral_funtion(ydealii, intergaldealii);
                    PetscFunctionReturn(PETSC_SUCCESS);
                };
    TSSetRHSFunction(quadts,NULL,ts_integral_funtion,this); 
    /* setup drdu */
    const auto ts_integral_u =
               [](TS ts, PetscReal t, Vec y, Mat DRDY, Mat P, void *ctx) 
               -> PetscErrorCode {
                    PetscFunctionBegin;
                    auto user = static_cast<Nonlinear_Time *>(ctx);
                    user->pcout<<"integral_jacobian_u"<<std::endl;
                    PETScWrappers::MPI::Vector ydealii(y);
                    user->assemble_integral_jacobian_u(ydealii);
                    PetscFunctionReturn(PETSC_SUCCESS);
               };
    TSSetRHSJacobian(quadts,
                     jacobian_matrix_dRdU.petsc_matrix(),
                     jacobian_matrix_dRdU.petsc_matrix(),
                     ts_integral_u,
                     this);
    /* setup drdp */
    const auto ts_integral_p =
               [](TS ts, PetscReal t, Vec y, Mat DRDP, void *ctx) 
               -> PetscErrorCode {
                    PetscFunctionBegin;
                    auto user = static_cast<Nonlinear_Time *>(ctx);
                    user->pcout<<"integral_jacobian_p"<<std::endl;
                    user->assemble_integral_jacobian_p();
                    PetscFunctionReturn(PETSC_SUCCESS);
               };
    TSSetRHSJacobianP(quadts,
                      jacobian_matrix_dRdP.petsc_matrix(),
                      ts_integral_p,
                      this);
    /* monitor */ 
    const auto ts_monitor = 
                [](TS ts, PetscInt steps, PetscReal time, Vec u, PetscInt numcost, Vec *lambda, Vec *mu, void *adjointmctx) 
                -> PetscErrorCode {
                    PetscFunctionBegin;
                    auto user = static_cast<Nonlinear_Time *>(adjointmctx);                    
                    // user->pcout<<"output_adjoint_results"<<std::endl;
                    user->pcout << "Adjoint Time step " << steps << " at t=" << time << std::endl;                    
                    // user->output_adjoint_results(time, steps);
                    PetscFunctionReturn(PETSC_SUCCESS);
                };
    TSAdjointMonitorSet(time_stepper.petsc_ts(),
                        ts_monitor,
                        this,
                        NULL);
    /* set solution save */
    TSSetSaveTrajectory(time_stepper.petsc_ts());
    TSGetTrajectory(time_stepper.petsc_ts(), &tj);
    TSTrajectorySetType(tj,
                        time_stepper.petsc_ts(),
                        TSTRAJECTORYMEMORY);
}