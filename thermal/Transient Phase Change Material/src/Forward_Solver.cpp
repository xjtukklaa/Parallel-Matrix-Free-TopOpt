#include "../include/Nonlinear_Time.h"

class Heat_Source : public Function<dim>
{
public:
    virtual double value(const Point<dim> &p,
                         const unsigned int component = 0) const override
    {
        if (p[0] * p[0] + p[1] * p[1] <= 0.04)
        {
            return 1e3;            
        }else
        {
            return 0;
        }
    }
};

void Nonlinear_Time::setup_forward_system()
{
    TimerOutput::Scope t(computing_timer, "setup_forward_system");
    dof_handler.distribute_dofs(fe);
    pcout<<"dofs heat : "<<dof_handler.n_dofs()<<std::endl;

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    locally_relevant_dofs = 
    DoFTools::extract_locally_relevant_dofs(dof_handler);

    homogeneous_constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler,
                                            homogeneous_constraints);
    VectorTools::interpolate_boundary_values(dof_handler,
                                             1,
                                             Functions::ZeroFunction<dim>(),
                                             homogeneous_constraints);
    homogeneous_constraints.close();

    current_constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler,
                                            current_constraints);
    VectorTools::interpolate_boundary_values(dof_handler,
                                             1,
                                             Functions::ConstantFunction<dim>(293.15),
                                             current_constraints);
    current_constraints.close();

    DynamicSparsityPattern dsp_DF_DU(dof_handler.n_dofs(),
                                     dof_handler.n_dofs(),
                                     locally_relevant_dofs);
    DoFTools::make_sparsity_pattern(dof_handler, 
                                    dsp_DF_DU,
                                    current_constraints,
                                    false);
    SparsityTools::distribute_sparsity_pattern(dsp_DF_DU,
                                               locally_owned_dofs, 
                                               mpi_communicator,
                                               locally_relevant_dofs);
    
    jacobian_matrix_dFdU.reinit(locally_owned_dofs,
                                locally_owned_dofs,
                                dsp_DF_DU,
                                mpi_communicator);
    // 
    IndexSet col_one(1); col_one.add_index(0);
    DynamicSparsityPattern dsp_DR_DU(dof_handler.n_dofs(),
                                     1,
                                     locally_relevant_dofs);                      
    SparsityTools::distribute_sparsity_pattern(dsp_DR_DU,
                                               locally_owned_dofs,
                                               mpi_communicator,
                                               locally_relevant_dofs);
    jacobian_matrix_dRdU.reinit(locally_owned_dofs,
                                col_one,
                                dsp_DR_DU,
                                mpi_communicator);
    MatSetType(jacobian_matrix_dRdU, MATMPIDENSE);
    // 
    adjoint_lambda.reinit(locally_owned_dofs,mpi_communicator);
    //  
    tempture_solution.reinit(locally_owned_dofs,mpi_communicator);

    tmp_solution.reinit(locally_owned_dofs,mpi_communicator);
    tmp_solution_dot.reinit(locally_owned_dofs,mpi_communicator);

    locally_relevant_solution.reinit(locally_owned_dofs,
                                     locally_relevant_dofs,
                                     mpi_communicator);
    locally_relevant_solution_dot.reinit(locally_owned_dofs,
                                         locally_relevant_dofs,
                                         mpi_communicator);
}

void Nonlinear_Time::implicit_function(const double /* time */,
                                       const PETScWrappers::MPI::Vector &solution,
                                       const PETScWrappers::MPI::Vector &solution_dot,
                                       PETScWrappers::MPI::Vector &residual)
{
    TimerOutput::Scope t(computing_timer, "implicit_function");

    tmp_solution = solution;
    tmp_solution_dot = solution_dot;
    
    current_constraints.distribute(tmp_solution);
    homogeneous_constraints.distribute(tmp_solution_dot);

    locally_relevant_solution = tmp_solution;
    locally_relevant_solution_dot = tmp_solution_dot;
    
    residual = 0;
    
    QGauss<dim> quadrature_formula(fe.degree + 1);
    QGauss<dim - 1> face_quadrature_formula(fe.degree + 1);

    FEValues<dim> fe_values(fe, quadrature_formula,
                            update_values | update_gradients 
                            | update_JxW_values | update_quadrature_points);
    FEFaceValues<dim> fe_face_values(fe,
                                   face_quadrature_formula,
                                   update_values | update_quadrature_points | update_JxW_values);       
    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points = quadrature_formula.size();
    const unsigned int n_face_q_points = face_quadrature_formula.size();
    
    QGauss<dim> quadrature_formula_simp(fe_simp.degree + 1);
    FEValues<dim> fe_values_simp(fe_simp, quadrature_formula_simp,
                            update_values | update_quadrature_points);
    
    Vector<double> cell_residual(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    std::vector<double>        solution_values(n_q_points);
    std::vector<Tensor<1,dim>> solution_gradients(n_q_points);
    std::vector<double>        solution_dot_values(n_q_points);

    std::vector<double>        simp_rho_values(1);

    Heat_Source heat_source;

    double cell_heat_conductivity = 0;
    double cell_heat_capacity = 0;
    double cell_density = 0;
    double cell_conbine_value = 0;
    double cell_muti_value = 0;

    auto cell = dof_handler.begin_active();
    auto cell_simp = dof_handler_simp.begin_active();
    auto cell_end = dof_handler.end();

    for (;cell != cell_end; cell++, cell_simp++)
    {        
        if (cell->is_locally_owned())
        {
            cell_residual = 0;

            fe_values.reinit(cell);
            fe_values.get_function_values(locally_relevant_solution,solution_values);
            fe_values.get_function_gradients(locally_relevant_solution,
                                            solution_gradients);
            fe_values.get_function_values(locally_relevant_solution_dot,
                                        solution_dot_values);

            fe_values_simp.reinit(cell_simp);
            fe_values_simp.get_function_values(simp_rho_filter,
                                               simp_rho_values);
            
            cell_heat_conductivity = heat_conductivity[0] + 
                                    (heat_conductivity[1] - heat_conductivity[0]) *
                                     MySimp.simp_value(simp_rho_values[0]);

            cell_density = density[1] * simp_rho_values[0] + (1 - simp_rho_values[0]) * density[0];

            cell_conbine_value = heat_capacity[1] * simp_rho_values[0] * density[1]/cell_density;

            cell_muti_value = (1 - simp_rho_values[0]) * density[0] / cell_density;

            for (unsigned int q = 0; q < n_q_points; ++q)
            {
                cell_heat_capacity = MyPcm.cp_t_x_value(solution_values[q],simp_rho_values[0]);
                cell_heat_capacity = cell_conbine_value + cell_heat_capacity * cell_muti_value;

                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                    cell_residual(i) += (fe_values.shape_grad(i, q) *
                                            cell_heat_conductivity *
                                            solution_gradients[q] 
                                        +
                                        fe_values.shape_value(i, q) *
                                            cell_heat_capacity *
                                            cell_density *
                                            solution_dot_values[q]
                                        - 
                                        fe_values.shape_value(i, q) *
                                            heat_source.value(fe_values.quadrature_point(q)) 
                                        ) *
                                        fe_values.JxW(q);
                }
            }
            for (auto cellface : cell->face_iterators())
            {
                if (cellface->at_boundary() && cellface->boundary_id() == 2)
                {
                    fe_face_values.reinit(cell, cellface);
                    for (unsigned int q = 0; q < n_face_q_points; ++q)
                    {
                        for (unsigned int i = 0; i < dofs_per_cell; ++i)
                        {
                            cell_residual(i) -= 1e3 *
                                                fe_face_values.shape_value(i,q) *
                                                fe_face_values.JxW(q);
                        }
                    }
                }
            }
            cell->get_dof_indices(local_dof_indices);
            current_constraints.distribute_local_to_global(cell_residual,
                                                           local_dof_indices,
                                                           residual);
        }
    }
    residual.compress(VectorOperation::add);

    for (const auto &c : current_constraints.get_lines())
    {
      if (locally_owned_dofs.is_element(c.index))
        {
          if (c.entries.empty()) 
            residual[c.index] = solution[c.index] - tmp_solution[c.index];
          else 
            residual[c.index] = solution[c.index];
        }        
    }
    residual.compress(VectorOperation::insert);
}

void Nonlinear_Time::assemble_implicit_jacobian_u(const double /* time */,
                                                  const PETScWrappers::MPI::Vector &solution,
                                                  const PETScWrappers::MPI::Vector &solution_dot,
                                                  const double alpha)
{
    TimerOutput::Scope t(computing_timer, "assemble_implicit_jacobian_u");

    tmp_solution = solution;
    tmp_solution_dot = solution_dot;
    
    current_constraints.distribute(tmp_solution);
    homogeneous_constraints.distribute(tmp_solution_dot);

    locally_relevant_solution = tmp_solution;
    locally_relevant_solution_dot = tmp_solution_dot;
    
    jacobian_matrix_dFdU = 0;

    QGauss<dim> quadrature_formula(fe.degree + 1);
    FEValues<dim> fe_values(fe, quadrature_formula,
                            update_values | update_gradients 
                            | update_JxW_values | update_quadrature_points);
    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points = quadrature_formula.size();

    std::vector<double>        solution_values(n_q_points);
    std::vector<Tensor<1,dim>> solution_gradients(n_q_points);
    std::vector<double>        solution_dot_values(n_q_points);

    QGauss<dim> quadrature_formula_simp(fe_simp.degree + 1);
    FEValues<dim> fe_values_simp(fe_simp, quadrature_formula_simp,
                                 update_values | update_quadrature_points);

    FullMatrix<double> cell_jacobian_dfdu(dofs_per_cell,dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    
    std::vector<double>        simp_rho_values(1);

    double cell_heat_conductivity = 0;
    double cell_heat_capacity = 0;
    double cell_heat_capacity_dt = 0;
    double cell_density = 0;
    double cell_conbine_value = 0;
    double cell_muti_value = 0;

    auto cell = dof_handler.begin_active();
    auto cell_simp = dof_handler_simp.begin_active();
    auto cell_end = dof_handler.end();

    for (;cell != cell_end; cell++, cell_simp++)
    {        
        if (cell->is_locally_owned())
        {
            cell_jacobian_dfdu = 0;

            fe_values.reinit(cell);
            fe_values.get_function_values(locally_relevant_solution,solution_values);
            fe_values.get_function_gradients(locally_relevant_solution,
                                            solution_gradients);
            fe_values.get_function_values(locally_relevant_solution_dot,
                                        solution_dot_values);

            fe_values_simp.reinit(cell_simp);
            fe_values_simp.get_function_values(simp_rho_filter,
                                               simp_rho_values);

            cell_heat_conductivity = heat_conductivity[0] + 
                                    (heat_conductivity[1] - heat_conductivity[0]) *
                                     MySimp.simp_value(simp_rho_values[0]);

            cell_density = density[1] * simp_rho_values[0] + (1 - simp_rho_values[0]) * density[0];

            cell_conbine_value = heat_capacity[1] * simp_rho_values[0] * density[1]/cell_density;

            cell_muti_value = (1 - simp_rho_values[0]) * density[0] / cell_density;

            for (unsigned int q = 0; q < n_q_points; ++q)
            {
                cell_heat_capacity = MyPcm.cp_t_x_value(solution_values[q],simp_rho_values[0]);
                cell_heat_capacity = cell_conbine_value + cell_heat_capacity * cell_muti_value;

                cell_heat_capacity_dt = MyPcm.dcp_dt_x_value(solution_values[q],simp_rho_values[0]);
                cell_heat_capacity_dt *= cell_muti_value;

                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                    cell_jacobian_dfdu(i,j) += (fe_values.shape_grad(i, q) *
                                                cell_heat_conductivity *
                                                fe_values.shape_grad(j, q) 
                                            + 
                                            fe_values.shape_value(i, q) *
                                                cell_density *
                                                cell_heat_capacity_dt *
                                                solution_dot_values[q] *
                                                fe_values.shape_value(j, q)
                                            +
                                            fe_values.shape_value(i, q) *
                                                cell_heat_capacity *
                                                cell_density *
                                                alpha *
                                                fe_values.shape_value(j, q)
                                            ) *
                                            fe_values.JxW(q);
                    }
                }
            }
            cell->get_dof_indices(local_dof_indices);
            current_constraints.distribute_local_to_global(cell_jacobian_dfdu,
                                                           local_dof_indices,
                                                           jacobian_matrix_dFdU);
        }
    }
    jacobian_matrix_dFdU.compress(VectorOperation::add);

    for (const auto &c : current_constraints.get_lines())
    {
        jacobian_matrix_dFdU.set(c.index, c.index, 1.0);
    }
    jacobian_matrix_dFdU.compress(VectorOperation::insert);
}

void Nonlinear_Time::output_forward_results(const double time,
                                            const unsigned int timestep_number,
                                            const PETScWrappers::MPI::Vector &solution)
{
    TimerOutput::Scope t(computing_timer, "output_forward_results");
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "temperature");
    data_out.build_patches();
    data_out.set_flags(DataOutBase::VtkFlags(time, timestep_number));
    const std::string filename =
        "forward_solution_" + Utilities::int_to_string(timestep_number, 4) + ".vtu";
    data_out.write_vtu_in_parallel(filename, mpi_communicator);
}

void Nonlinear_Time::setup_forward_solver()
{
    TimerOutput::Scope t(computing_timer, "setup_forward_solver");

    time_stepper.reinit(time_stepper_data);
    
    time_stepper.set_matrices(jacobian_matrix_dFdU, jacobian_matrix_dFdU);

    time_stepper.implicit_function = [&](const double time,
                                         const PETScWrappers::MPI::Vector &solution,
                                         const PETScWrappers::MPI::Vector &solution_dot,
                                         PETScWrappers::MPI::Vector &residual)
    {
        pcout<<"implicit_function"<<std::endl;
        this->implicit_function(time, solution, solution_dot, residual);
    };

    time_stepper.setup_jacobian = [&](const double time,
                                      const PETScWrappers::MPI::Vector &solution,
                                      const PETScWrappers::MPI::Vector &solution_dot,
                                      const double alpha)
    {
        pcout<<"implicit_jacobian_u"<<std::endl;
        this->assemble_implicit_jacobian_u(time, solution, solution_dot, alpha);
    };

    time_stepper.algebraic_components = [&]()
    {
        // pcout << "algebraic_components" << std::endl;
        IndexSet algebraic_set(dof_handler.n_dofs());
        algebraic_set.add_indices(DoFTools::extract_hanging_node_dofs(dof_handler));
        algebraic_set.add_indices(DoFTools::extract_boundary_dofs(dof_handler, ComponentMask(), {1}));
        return algebraic_set;
    };

    time_stepper.update_constrained_components = [&](const double time, 
                                                     PETScWrappers::MPI::Vector &solution)
    {
        // pcout << "update_constrained_components" << std::endl;
        current_constraints.distribute(solution);
    };

    time_stepper.monitor = [&](const double time,
                               const PETScWrappers::MPI::Vector &solution,
                               const unsigned int timestep_number)
    {
        // pcout << "monitor" << std::endl;
        pcout << "Forward Time step " << timestep_number << " at t=" << time << std::endl;
        // this->output_forward_results(time, timestep_number, solution);
    };
}