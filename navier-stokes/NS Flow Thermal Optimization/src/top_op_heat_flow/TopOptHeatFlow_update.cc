#include "../../include/top_opt_heat_flow.h"

namespace TopOpt
{
    void TopOptHeatFlow::reinit_mma(const bool is_initial_step)
    {
        TimerOutput::Scope t(computing_timer, "reinit_mma");
        PetscScalar *aMMA = new PetscScalar[n_constraints];
        PetscScalar *cMMA = new PetscScalar[n_constraints];
        PetscScalar *dMMA = new PetscScalar[n_constraints];
        for (PetscInt i = 0; i < n_constraints; i++)
        {
            aMMA[i] = 0.0;
            cMMA[i] = 10000.0;
            dMMA[i] = 0.0;
        }

        if (mma == nullptr && is_initial_step)
        {
            mma = new MMA(n_variables_global, n_constraints, xval_petsc, aMMA, cMMA, dMMA);
        }
        else
        {
            delete mma;
            mma = new MMA(n_variables_global, n_constraints, loop_counter, xold1_petsc, xold2_petsc, xmax_petsc, xmin_petsc, aMMA, cMMA, dMMA);
        }

        delete[] aMMA;
        delete[] cMMA;
        delete[] dMMA;
    }

    void TopOptHeatFlow::update_function_values(const bool print_values)
    {

        QGauss<dim> quadrature_formula(quadrature_degree + 1);
        QGauss<dim - 1> face_quadrature_formula(quadrature_degree + 1);
        hp::QCollection<dim> hp_quadrature_formula(quadrature_formula);

        hp::FEValues<dim> top_fe_values(top_mapping,
                                        top_dof_handler.get_fe_collection(),
                                        hp_quadrature_formula,
                                        update_values | update_gradients | update_hessians | update_JxW_values);
        hp::FEValues<dim> top_equivalent_fe_values(top_equivalent_mapping,
                                                   top_equivalent_fe,
                                                   hp_quadrature_formula,
                                                   update_values);
        FEValues<dim> ns_fe_values(ns_fe,
                                   quadrature_formula,
                                   update_values | update_gradients);
        FEValues<dim> temp_fe_values(temp_fe,
                                     quadrature_formula,
                                     update_values | update_gradients);
        FEFaceValues<dim> ns_fe_face_values(ns_fe,
                                            face_quadrature_formula,
                                            update_values | update_normal_vectors |
                                                update_JxW_values);
        FEFaceValues<dim> temp_fe_face_values(temp_fe,
                                              face_quadrature_formula,
                                              update_values | update_JxW_values);

        const unsigned int n_q_points = quadrature_formula.size();
        const unsigned int n_face_q_points = face_quadrature_formula.size();

        const FEValuesExtractors::Vector velocities(0);
        const FEValuesExtractors::Scalar pressure(dim);

        /*预分配内存*/
        std::vector<Tensor<1, dim>> present_velocity_values(n_q_points);
        std::vector<Tensor<2, dim>> present_velocity_gradients(n_q_points);
        std::vector<double> present_temperature_values(n_q_points);
        std::vector<Tensor<1, dim>> present_temperature_gradients(n_q_points);
        std::vector<double> xphys_heaviside_values(n_q_points);
        std::vector<Tensor<1, dim>> xphys_heaviside_gradients(n_q_points);
        std::vector<Tensor<1, dim>> xphys_filter_gradients(n_q_points);
        std::vector<Tensor<2, dim>> xphys_filter_hessians(n_q_points);
        std::vector<double> xphys_equivalent_values(n_q_points);

        std::vector<double> present_face_pressure_values(n_face_q_points);
        std::vector<Tensor<1, dim>> present_face_velocity_values(n_face_q_points);
        std::vector<Tensor<1, dim>> present_face_normal_vectors(n_face_q_points);
        std::vector<double> present_face_temp_values(n_face_q_points);

        /*待计算的量*/

        double domain_int_temp = 0;
        double domain_int_heat_exchange = 0.0;
        double domain_int_xphys = 0;
        double domain_int_flow_dissipation = 0;
        double domain_int_norm_xphys_heaviside_gradients = 0;
        double domain_int_norm_xphys_filter_gradients = 0;
        double domain_int_fnorm_xphys_filter_hessians = 0;
        double domain_int_norm_u_grad_xphys = 0;
        double domain_int_abs_cos_u_grad_xphys = 0;
        double domain_int_xphys_abs_cos_u_grad_xphys = 0;
        double domain_int_abs_cos_u_grad_temp = 0;
        double domain_int_xphys_abs_cos_u_grad_temp = 0;
        double domain_int_entransy_dissipation = 0;

        double bound_int_pressure = 0;
        double bound_int_pump = 0;
        double bound_int_temp = 0;

        double volume_total = GridTools::volume(triangulation);
        double volume_design = domain_volume[0];
        double area_inlet = boundary_area[1];
        double area_heatflux = boundary_area[4];

        double average_temp = 0;
        double average_bound_temp = 0;
        double heat_exchange = 0.0;
        double flow_dissipation = 0;
        double volume_frac = 0;
        double inlet_pressure = 0;
        double pump_power = 0;
        double norm_xphys_heaviside_gradients = 0;
        double fnorm_xphys_filter_hessians = 0;
        double norm_xphys_filter_gradients = 0;
        double norm_u_grad_xphys = 0;
        double abs_cos_u_grad_xphys = 0;
        double xphys_abs_cos_u_grad_xphys = 0;
        double abs_cos_u_grad_temp = 0;
        double xphys_abs_cos_u_grad_temp = 0;
        double entransy_dissipation = 0;

        /*迭代器*/
        auto top_cell = top_dof_handler.begin_active();
        const auto endc = top_dof_handler.end();
        auto top_equivalent_cell = top_equivalent_dof_handler.begin_active();
        auto ns_cell = ns_dof_handler.begin_active();
        auto temp_cell = temp_dof_handler.begin_active();

        for (; top_cell != endc; ++top_cell, ++top_equivalent_cell, ++ns_cell, ++temp_cell)
        {
            if (top_cell->is_locally_owned())
            {
                ns_fe_values.reinit(ns_cell);
                temp_fe_values.reinit(temp_cell);
                top_fe_values.reinit(top_cell);
                top_equivalent_fe_values.reinit(top_equivalent_cell);

                /*读取高斯点处的函数值*/
                ns_fe_values[velocities].get_function_values(ns_solution_relevant,
                                                             present_velocity_values);
                ns_fe_values[velocities].get_function_gradients(ns_solution_relevant,
                                                                present_velocity_gradients);
                temp_fe_values.get_function_values(temp_solution_relevant,
                                                   present_temperature_values);
                temp_fe_values.get_function_gradients(temp_solution_relevant,
                                                      present_temperature_gradients);
                auto &present_top_fe_values = top_fe_values.get_present_fe_values();
                auto &present_top_equivalent_fe_values = top_equivalent_fe_values.get_present_fe_values();
                present_top_fe_values.get_function_values(xphys_heaviside_relevant,
                                                          xphys_heaviside_values);
                present_top_fe_values.get_function_gradients(xphys_heaviside_relevant,
                                                             xphys_heaviside_gradients);
                present_top_fe_values.get_function_gradients(xphys_filter_relevant,
                                                             xphys_filter_gradients);
                present_top_fe_values.get_function_hessians(xphys_filter_relevant,
                                                            xphys_filter_hessians);
                present_top_equivalent_fe_values.get_function_values(xphys_equivalent_relevant,
                                                                     xphys_equivalent_values);

                /*目标函数和约束函数计算*/

                for (unsigned int q = 0; q < n_q_points; ++q)
                {
                    double xphys_equivalent_values_q_point = std::max(std::min(xphys_equivalent_values[q], 1.0), 0.0);
                    permeability = material_interpolate.permeability_interpolate(xphys_equivalent_values_q_point);
                    thermal_conductivity = material_interpolate.conductivity_interpolate(xphys_equivalent_values_q_point);
                    heatsource_coeff = material_interpolate.heatsource_interpolate(xphys_equivalent_values_q_point);

                    if (top_cell->material_id() == 0)
                    {
                        domain_int_xphys += xphys_equivalent_values_q_point * present_top_fe_values.JxW(q);
                    }

                    domain_int_temp += present_temperature_values[q] * present_top_fe_values.JxW(q);

                    domain_int_heat_exchange +=
                        heatsource_coeff * (t_Q - present_temperature_values[q]) *
                        present_top_fe_values.JxW(q);

                    domain_int_flow_dissipation +=
                        (0.5 * viscosity * scalar_product(present_velocity_gradients[q], present_velocity_gradients[q]) +
                         permeability * present_velocity_values[q].norm_square()) *
                        present_top_fe_values.JxW(q);

                    domain_int_norm_xphys_heaviside_gradients += xphys_heaviside_gradients[q].norm() * present_top_fe_values.JxW(q);
                    domain_int_norm_xphys_filter_gradients += xphys_filter_gradients[q].norm() * present_top_fe_values.JxW(q);
                    domain_int_fnorm_xphys_filter_hessians += xphys_filter_hessians[q].norm() * present_top_fe_values.JxW(q);
                    domain_int_norm_u_grad_xphys += (std::abs(present_velocity_values[q] * xphys_heaviside_gradients[q])) *
                                                    present_top_fe_values.JxW(q);

                    if (std::abs(present_velocity_values[q] * xphys_heaviside_gradients[q]) > field_if_epsilon)
                    {
                        domain_int_abs_cos_u_grad_xphys +=
                            (std::abs(present_velocity_values[q] * xphys_heaviside_gradients[q]) /
                             (present_velocity_values[q].norm() * xphys_heaviside_gradients[q].norm() + field_plus_epsilon)) *
                            present_top_fe_values.JxW(q);
                        domain_int_xphys_abs_cos_u_grad_xphys +=
                            (xphys_heaviside_values[q] *
                             (std::abs(present_velocity_values[q] * xphys_heaviside_gradients[q]) /
                              (present_velocity_values[q].norm() * xphys_heaviside_gradients[q].norm() + field_plus_epsilon))) *
                            present_top_fe_values.JxW(q);
                    }

                    if (std::abs(present_velocity_values[q] * present_temperature_gradients[q]) > field_if_epsilon)
                    {
                        domain_int_abs_cos_u_grad_temp +=
                            (std::abs(present_velocity_values[q] * present_temperature_gradients[q]) /
                             (present_velocity_values[q].norm() * present_temperature_gradients[q].norm())) *
                            present_top_fe_values.JxW(q);
                        domain_int_xphys_abs_cos_u_grad_temp +=
                            (xphys_heaviside_values[q] *
                             std::abs(present_velocity_values[q] * present_temperature_gradients[q]) /
                             (present_velocity_values[q].norm() * present_temperature_gradients[q].norm())) *
                            present_top_fe_values.JxW(q);
                    }
                    domain_int_entransy_dissipation +=
                        (0.5 * thermal_conductivity * present_temperature_gradients[q].norm_square()) *
                        present_top_fe_values.JxW(q);
                }

                for (const auto face : top_cell->face_indices())
                {
                    auto face_iter = top_cell->face(face);
                    if (face_iter->at_boundary())
                    {
                        ns_fe_face_values.reinit(ns_cell, face);
                        temp_fe_face_values.reinit(temp_cell, face);
                        ns_fe_face_values[pressure].get_function_values(ns_solution_relevant,
                                                                        present_face_pressure_values);
                        ns_fe_face_values[velocities].get_function_values(ns_solution_relevant,
                                                                          present_face_velocity_values);
                        temp_fe_face_values.get_function_values(temp_solution_relevant,
                                                                present_face_temp_values);
                        present_face_normal_vectors = ns_fe_face_values.get_normal_vectors();

                        if (face_iter->boundary_id() == 1)
                        {
                            for (unsigned int q = 0; q < n_face_q_points; ++q)
                            {
                                bound_int_pressure += present_face_pressure_values[q] * ns_fe_face_values.JxW(q);
                            }
                        }

                        if (face_iter->boundary_id() == 1 || face_iter->boundary_id() == 2)
                        {
                            for (unsigned int q = 0; q < n_face_q_points; ++q)
                            {
                                bound_int_pump +=
                                    (-1.0) *
                                    (present_face_pressure_values[q] + 0.5 * present_face_velocity_values[q].norm_square()) *
                                    (present_face_velocity_values[q] * present_face_normal_vectors[q]) *
                                    ns_fe_face_values.JxW(q);
                            }
                        }

                        if (face_iter->boundary_id() == 4)
                        {
                            for (unsigned int q = 0; q < n_face_q_points; ++q)
                            {
                                bound_int_temp += present_face_temp_values[q] * temp_fe_face_values.JxW(q);
                            }
                        }
                    }
                }
            }
        }
        { /*？？是否需要每个处理器的各自设计变量对应不同的约束或不同的值？？*/

            domain_int_temp = Utilities::MPI::sum(domain_int_temp, mpi_communicator);
            domain_int_heat_exchange = Utilities::MPI::sum(domain_int_heat_exchange, mpi_communicator);
            domain_int_flow_dissipation = Utilities::MPI::sum(domain_int_flow_dissipation, mpi_communicator);
            domain_int_xphys = Utilities::MPI::sum(domain_int_xphys, mpi_communicator);
            domain_int_norm_xphys_heaviside_gradients = Utilities::MPI::sum(domain_int_norm_xphys_heaviside_gradients, mpi_communicator);
            domain_int_norm_xphys_filter_gradients = Utilities::MPI::sum(domain_int_norm_xphys_filter_gradients, mpi_communicator);
            domain_int_fnorm_xphys_filter_hessians = Utilities::MPI::sum(domain_int_fnorm_xphys_filter_hessians, mpi_communicator);
            domain_int_norm_u_grad_xphys = Utilities::MPI::sum(domain_int_norm_u_grad_xphys, mpi_communicator);
            domain_int_abs_cos_u_grad_xphys = Utilities::MPI::sum(domain_int_abs_cos_u_grad_xphys, mpi_communicator);
            domain_int_xphys_abs_cos_u_grad_xphys = Utilities::MPI::sum(domain_int_xphys_abs_cos_u_grad_xphys, mpi_communicator);
            domain_int_abs_cos_u_grad_temp = Utilities::MPI::sum(domain_int_abs_cos_u_grad_temp, mpi_communicator);
            domain_int_xphys_abs_cos_u_grad_temp = Utilities::MPI::sum(domain_int_xphys_abs_cos_u_grad_temp, mpi_communicator);
            domain_int_entransy_dissipation = Utilities::MPI::sum(domain_int_entransy_dissipation, mpi_communicator);

            bound_int_pressure = Utilities::MPI::sum(bound_int_pressure, mpi_communicator);
            bound_int_pump = Utilities::MPI::sum(bound_int_pump, mpi_communicator);
            bound_int_temp = Utilities::MPI::sum(bound_int_temp, mpi_communicator);
        }
        average_temp = domain_int_temp / volume_total;
        average_bound_temp = bound_int_temp / area_heatflux;
        heat_exchange = domain_int_heat_exchange;
        flow_dissipation = domain_int_flow_dissipation;

        norm_xphys_heaviside_gradients = domain_int_norm_xphys_heaviside_gradients / volume_total;
        norm_xphys_filter_gradients = domain_int_norm_xphys_filter_gradients / volume_total;
        fnorm_xphys_filter_hessians = domain_int_fnorm_xphys_filter_hessians / volume_total;
        norm_u_grad_xphys = domain_int_norm_u_grad_xphys / volume_total;
        abs_cos_u_grad_xphys = domain_int_abs_cos_u_grad_xphys / volume_total;
        xphys_abs_cos_u_grad_xphys = domain_int_xphys_abs_cos_u_grad_xphys / volume_total;

        abs_cos_u_grad_temp = domain_int_abs_cos_u_grad_temp / volume_total;
        xphys_abs_cos_u_grad_temp = domain_int_xphys_abs_cos_u_grad_temp / volume_total;
        entransy_dissipation = domain_int_entransy_dissipation / volume_total;

        volume_frac = domain_int_xphys / volume_design;
        inlet_pressure = bound_int_pressure / area_inlet;
        pump_power = bound_int_pump;

        if (print_values)
        {
            pcout << std::endl
                  << "******************************************************************************" << std::endl
                  << "Present topology optimization have: " << std::endl
                  << "Average Temperature = " << average_temp << std::endl
                  << "Average Temperature of Heatflux Boundary = " << average_bound_temp << std::endl
                  << "Heat Exchange = " << heat_exchange << std::endl
                  << "Flow Dissipation = " << flow_dissipation << std::endl
                  << "Norm Xphys Heaviside Gradients = " << norm_xphys_heaviside_gradients << std::endl
                  << "Norm Xphys Filter Gradients = " << norm_xphys_filter_gradients << std::endl
                  << "Fnorm Xphys Filter Hessians = " << fnorm_xphys_filter_hessians << std::endl
                  << "Norm U GradXphys = " << norm_u_grad_xphys << std::endl
                  << "Abs Cos U GradXphys = " << abs_cos_u_grad_xphys << std::endl
                  << "Xphys Abs Cos U GradXphys = " << xphys_abs_cos_u_grad_xphys << std::endl
                  << "Abs Cos U GradTemp = " << abs_cos_u_grad_temp << std::endl
                  << "Xphys Abs Cos U GradTemp = " << xphys_abs_cos_u_grad_temp << std::endl
                  << "Entransy Dissipation = " << entransy_dissipation << std::endl
                  << "Volume Fraction = " << volume_frac << std::endl
                  << "Inlet Pressure = " << inlet_pressure << std::endl
                  << "Pump Power = " << pump_power << std::endl;
        }

        switch (objective_function)
        {
        case PRM::mean_temp_objective:
        {
            obj_value = average_temp;
            break;
        }
        case PRM::heat_exchange_objective:
        {
            obj_value = heat_exchange;
            break;
        }
        case PRM::mean_boundary_temp_objective:
        {
            obj_value = average_bound_temp;
            break;
        }
        case PRM::xphys_abs_cos_u_grad_temp_objective:
        {
            obj_value = xphys_abs_cos_u_grad_temp;
            break;
        }

        default:
            break;
        }

        switch (flow_constraint_function)
        {
        case PRM::flow_dissipation_constraint:
        {
            cstr_flow_value = flow_dissipation;
            break;
        }
        case PRM::inlet_pressure_constraint:
        {
            cstr_flow_value = inlet_pressure;
            break;
        }
        case PRM::pump_power_constraint:
        {
            cstr_flow_value = pump_power;
            break;
        }
        case PRM::only_volume_constraint:
        {
            cstr_flow_value = volume_frac;
            break;
        }
        default:
            break;
        }

        cstr_vol_value = volume_frac;

        switch (field_constraint_function)
        {
        case PRM::no_field_constraints:
        {
            cstr_field_value = 0;
            break;
        }

        case PRM::norm_xphys_heaviside_gradients_constraint:
        {
            cstr_field_value = norm_xphys_heaviside_gradients;
            break;
        }
        case PRM::norm_u_grad_xphys_constraint:
        {
            cstr_field_value = norm_u_grad_xphys;
            break;
        }
        case PRM::abs_cos_u_grad_xphys_constraint:
        {
            cstr_field_value = abs_cos_u_grad_xphys;
            break;
        }
        case PRM::xphys_abs_cos_u_grad_xphys_constraint:
        {
            cstr_field_value = xphys_abs_cos_u_grad_xphys;
            break;
        }
        case PRM::fnorm_xphys_filter_hessians_constraint:
        {
            cstr_field_value = fnorm_xphys_filter_hessians;
            break;
        }
        case PRM::norm_xphys_filter_gradients_constraint:
        {
            cstr_field_value = norm_xphys_filter_gradients;
        }

        default:
            break;
        }
    }

    void TopOptHeatFlow::topology_update(double &change)
    {
        TimerOutput::Scope t(computing_timer, "MMA_Optimization");
        /*1. 确定设计变量的数量、约束函数的数量*/
        PetscInt nloc;
        VecGetLocalSize(xval_petsc, &nloc);

        /*2. 将分布式的向量转化为每个处理器独自拥有的向量分量，进行代数运算，并传递回分布式向量*/
        PetscScalar Xmin = 0.0;
        PetscScalar Xmax = 1.0;
        PetscScalar movlim = 0.1;

        PetscScalar *dfdx, **dgdx;

        VecGetArray(dfdx_petsc, &dfdx);

        VecGetArrays(dgdx_petsc, n_constraints, &dgdx);

        /* 更新计算目标和约束的初代值 */
        if (loop_counter == 0)
        {
            update_function_values(false);
            obj_cstr_init_step_value[0] = obj_value;
            obj_cstr_init_step_value[1] = cstr_flow_value;
            obj_cstr_init_step_value[2] = cstr_vol_value;
            obj_cstr_init_step_value[3] = cstr_field_value;
            pcout << std::endl
                  << "******************************************************************************" << std::endl;
            pcout << "Objective & Constraint initial values (0th step) ..." << std::endl
                  << "Thermal Objective: " << obj_cstr_init_step_value[0] << std::endl
                  << "Flow Constraint:   " << obj_cstr_init_step_value[1] << std::endl
                  << "Volume Constraint: " << obj_cstr_init_step_value[2] << std::endl
                  << "Field Constraint:  " << obj_cstr_init_step_value[3] << std::endl;
        }

        /* 不归一化 */

        double sen_obj_absmax = 1.0, sen_flow_absmax = 1.0,
               sen_vol_absmax = 1.0, sen_field_absmax = 1.0;

        switch (PRM::normalization_method)
        {
        case PRM::normalization_off:
        {
            sen_obj_absmax = 1.0;
            sen_flow_absmax = 1.0;
            sen_vol_absmax = 1.0;
            sen_field_absmax = 1.0;
            break;
        }

        case PRM::normalization_on:
        {

            /* 目标采用初代值，约束采用约束值 */
            sen_obj_absmax = std::max(1e-12, std::abs(obj_cstr_init_step_value[0]));
            sen_flow_absmax = flow_cstr;
            sen_vol_absmax = vol_cstr;
            sen_field_absmax = field_cstr;

            break;
        }

        default:
            break;
        }
        std::pair<unsigned int, unsigned int> local_range = xphys_values_distributed.local_range();

        for (unsigned int i = local_range.first, k = 0; i < local_range.second; ++i, ++k)
        {
            double xval_tmp = xphys_values_distributed[i];
            xval_tmp = std::max(std::min(xval_tmp, Xmax), Xmin);
            U_distributed[i] = std::min(Xmax, xval_tmp + movlim);
            L_distributed[i] = std::max(Xmin, xval_tmp - movlim);

            dfdx[k] = sens_obj_dfilter_relevant(i) / sen_obj_absmax;

            dgdx[0][k] = sens_flow_dfilter_relevant(i) / sen_flow_absmax;
            dgdx[1][k] = sens_vol_dfilter_relevant(i) / sen_vol_absmax;

            if (field_constraint_function != PRM::no_field_constraints)
            {
                if (loop_counter > 200)
                {
                    dgdx[2][k] = sens_field_dfilter_relevant(i) / sen_field_absmax;
                }
                else
                {
                    dgdx[2][k] = 0;
                }
            }
        }
        U_distributed.compress(VectorOperation::insert);
        L_distributed.compress(VectorOperation::insert);
        top_constraints.distribute(U_distributed);
        top_constraints.distribute(L_distributed);
        U_relevant = U_distributed;
        L_relevant = L_distributed;

        /*VecRestoreArray可以自动释放dfdx所指向的内存，由petsc自动管理内存，不需要我们手动delete[] dfdx;
        通过简单的验证即可证明，即访问dfdx[0]，如果返回值说明内存没有释放，如果程序崩溃说明内存已经释放，访问不合法*/

        VecRestoreArray(dfdx_petsc, &dfdx);
        VecRestoreArrays(dgdx_petsc, n_constraints, &dgdx);

        /*4. 计算目标函数和约束函数的值*/

        PetscScalar *fval_petsc = new PetscScalar[1];
        PetscScalar *gval_petsc = new PetscScalar[n_constraints];

        double fval_double = 0.0, gval_flow = 0.0, gval_vol = 0.0, gval_field = 0.0;
        std::vector<double> gval_std_vec;
        update_function_values();
        fval_petsc[0] = fval_double;
        gval_flow = cstr_flow_value;
        gval_vol = cstr_vol_value;
        gval_field = cstr_field_value;

        gval_petsc[0] = (gval_flow / flow_cstr) - 1.0;

        gval_petsc[1] = (gval_vol / vol_cstr) - 1.0;
        if (field_constraint_function != PRM::no_field_constraints)
        {

            gval_petsc[2] = (-1.0) * ((gval_field / field_cstr) - 1.0);
        }

        /*5. 优化器更新设计变量*/

        {

            mma->Update(xval_petsc, dfdx_petsc, gval_petsc, dgdx_petsc, xmin_petsc, xmax_petsc);
            mma->Restart(xold1_petsc, xold2_petsc, xmax_petsc, xmin_petsc);
        }
        delete[] fval_petsc;
        delete[] gval_petsc;

        /*6. 将更新后的设计变量分布到各个处理器上，并计算设计变量最大变化值*/

        xphys_values_distributed.compress(VectorOperation::insert);
        xold1_distributed.compress(VectorOperation::insert);
        xold2_distributed.compress(VectorOperation::insert);
        U_distributed.compress(VectorOperation::insert);
        L_distributed.compress(VectorOperation::insert);
        top_constraints.distribute(xphys_values_distributed);
        top_constraints.distribute(xold1_distributed);
        top_constraints.distribute(xold2_distributed);
        top_constraints.distribute(U_distributed);
        top_constraints.distribute(L_distributed);
        xphys_values_relevant = xphys_values_distributed;
        xold1_relevant = xold1_distributed;
        xold2_relevant = xold2_distributed;
        U_relevant = U_distributed;
        L_relevant = L_distributed;

        change = 0.0;
        for (unsigned int i = local_range.first; i < local_range.second; ++i)
        {
            change = std::max(change, std::abs(xphys_values_relevant(i) - xold1_relevant(i)));
        }
        change = Utilities::MPI::max(change, mpi_communicator);
        pcout << "change = " << change << std::endl;
    }

}