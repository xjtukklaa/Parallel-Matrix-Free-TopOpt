#include "../../include/top_opt_heat_flow.h"

namespace TopOpt
{

    void TopOptHeatFlow::setup_dofs()
    {
        TimerOutput::Scope t(computing_timer, "setup");

        ns_system_matrix.clear();
        temp_system_matrix.clear();
        ad_temp_system_matrix.clear();
        ad_ns_system_matrix.clear();
        filter_system_matrix.clear();

        ns_dof_handler.distribute_dofs(ns_fe);
        temp_dof_handler.distribute_dofs(temp_fe);
        ad_temp_dof_handler.distribute_dofs(ad_temp_fe);
        ad_ns_dof_handler.distribute_dofs(ad_ns_fe);

        auto cell = top_dof_handler.begin_active();
        const auto endc = top_dof_handler.end();
        auto top_equivalent_cell = top_equivalent_dof_handler.begin_active();
        auto filter_cell = filter_dof_handler.begin_active();

        for (; cell != endc; ++cell, ++top_equivalent_cell, ++filter_cell)
        {
            if (cell->is_locally_owned())
            {
                if ((cell->material_id() == 1) || (cell->material_id() == 2))
                {
                    cell->set_active_fe_index(1);
                    top_equivalent_cell->set_active_fe_index(1);
                    filter_cell->set_active_fe_index(1);
                }
                else if (cell->material_id() == 0)
                {
                    cell->set_active_fe_index(0);
                    top_equivalent_cell->set_active_fe_index(0);
                    filter_cell->set_active_fe_index(0);
                }
                else
                    Assert(false, ExcNotImplemented());
            }
        }

        if (top_degree == 0)
        {
            top_dof_handler.distribute_dofs(top_dg_fe);
            top_mapping = hp::MappingCollection<dim>(top_dg_mapping);
        }
        else
        {
            top_dof_handler.distribute_dofs(top_cg_fe);
            top_mapping = hp::MappingCollection<dim>(top_cg_mapping);
        }
        top_equivalent_dof_handler.distribute_dofs(top_equivalent_fe);
        filter_dof_handler.distribute_dofs(filter_fe);

        /*我们根据我们想要如何创建块矩阵和向量，将本地拥有的和本地相关的 DoF 的IndexSet分成
        两个 IndexSet。*/
        ns_owned_partitioning = ns_dof_handler.locally_owned_dofs();
        ns_relevant_partitioning = DoFTools::extract_locally_relevant_dofs(ns_dof_handler);

        temp_owned_partitioning = temp_dof_handler.locally_owned_dofs();
        temp_relevant_partitioning = DoFTools::extract_locally_relevant_dofs(temp_dof_handler);

        ad_temp_owned_partitioning = ad_temp_dof_handler.locally_owned_dofs();
        ad_temp_relevant_partitioning = DoFTools::extract_locally_relevant_dofs(ad_temp_dof_handler);

        ad_ns_owned_partitioning = ad_ns_dof_handler.locally_owned_dofs();
        ad_ns_relevant_partitioning = DoFTools::extract_locally_relevant_dofs(ad_ns_dof_handler);

        top_owned_partitioning = top_dof_handler.locally_owned_dofs();
        top_relevant_partitioning = DoFTools::extract_locally_relevant_dofs(top_dof_handler);

        top_equivalent_owned_partitioning = top_equivalent_dof_handler.locally_owned_dofs();
        top_equivalent_relevant_partitioning = DoFTools::extract_locally_relevant_dofs(top_equivalent_dof_handler);

        filter_owned_partitioning = filter_dof_handler.locally_owned_dofs();
        filter_relevant_partitioning = DoFTools::extract_locally_relevant_dofs(filter_dof_handler);

        /*在牛顿方案中，我们首先对初始步骤获得的解应用边界条件。为了确保牛顿迭代过程中边界条件保持满足，
        更新δuk时使用零边界条件。因此我们设置了两个不同的约束对象。*/

        /* 速度场边界-非零边界（用于速度当前值） */
        const FEValuesExtractors::Vector velocities(0);
        {
            Functions::ZeroFunction<dim> non_slip_boundary(dim + 1);
            NSBoundaryValues<dim> inlet_boundary(triangulation);
            std::map<types::boundary_id, const Function<dim> *> boundary_values;
            boundary_values[0] = &non_slip_boundary;
            boundary_values[4] = &non_slip_boundary;
            boundary_values[1] = &inlet_boundary;

            ns_nonzero_constraints.reinit(ns_owned_partitioning, ns_relevant_partitioning);

            DoFTools::make_hanging_node_constraints(ns_dof_handler, ns_nonzero_constraints);
            VectorTools::interpolate_boundary_values(ns_dof_handler,
                                                     boundary_values,
                                                     ns_nonzero_constraints,
                                                     ns_fe.component_mask(velocities));

            std::set<types::boundary_id> no_normal_flux_boundaries = {3};
            VectorTools::compute_no_normal_flux_constraints(ns_dof_handler,
                                                            0,
                                                            no_normal_flux_boundaries,
                                                            ns_nonzero_constraints);
        }
        ns_nonzero_constraints.close();

        /* 速度场边界-零边界（用于速度更新值） */
        {
            Functions::ZeroFunction<dim> non_slip_boundary(dim + 1);
            Functions::ZeroFunction<dim> inlet_boundary(dim + 1);
            std::map<types::boundary_id, const Function<dim> *> boundary_values;
            boundary_values[0] = &non_slip_boundary;
            boundary_values[4] = &non_slip_boundary;
            boundary_values[1] = &inlet_boundary;

            ns_zero_constraints.reinit(ns_owned_partitioning, ns_relevant_partitioning);

            DoFTools::make_hanging_node_constraints(ns_dof_handler, ns_zero_constraints);
            VectorTools::interpolate_boundary_values(ns_dof_handler,
                                                     boundary_values,
                                                     ns_zero_constraints,
                                                     ns_fe.component_mask(velocities));

            std::set<types::boundary_id> no_normal_flux_boundaries = {3};
            VectorTools::compute_no_normal_flux_constraints(ns_dof_handler,
                                                            0,
                                                            no_normal_flux_boundaries,
                                                            ns_zero_constraints);
        }
        ns_zero_constraints.close();

        /* 温度场边界 */
        {
            TempBoundaryValues<dim> fixed_boundary;
            std::map<types::boundary_id, const Function<dim> *> boundary_values;
            boundary_values[1] = &fixed_boundary;

            temp_constraints.reinit(temp_owned_partitioning, temp_relevant_partitioning);

            DoFTools::make_hanging_node_constraints(temp_dof_handler, temp_constraints);
            VectorTools::interpolate_boundary_values(temp_dof_handler,
                                                     boundary_values,
                                                     temp_constraints);
        }
        temp_constraints.close();

        /* 伴随温度场边界 */
        {
            Functions::ZeroFunction<dim> fixed_boundary(1);
            std::map<types::boundary_id, const Function<dim> *> boundary_values;
            boundary_values[1] = &fixed_boundary;

            ad_temp_constraints.reinit(ad_temp_owned_partitioning, ad_temp_relevant_partitioning);

            DoFTools::make_hanging_node_constraints(ad_temp_dof_handler, ad_temp_constraints);
            VectorTools::interpolate_boundary_values(ad_temp_dof_handler,
                                                     boundary_values,
                                                     ad_temp_constraints);
        }
        ad_temp_constraints.close();

        /* 伴随速度场边界 */
        {
            Functions::ZeroFunction<dim> non_slip_boundary(dim + 1);
            Functions::ZeroFunction<dim> inlet_boundary(dim + 1);
            std::map<types::boundary_id, const Function<dim> *> boundary_values;
            boundary_values[0] = &non_slip_boundary;
            boundary_values[4] = &non_slip_boundary;
            boundary_values[1] = &inlet_boundary;

            ad_ns_constraints.reinit(ad_ns_owned_partitioning, ad_ns_relevant_partitioning);

            DoFTools::make_hanging_node_constraints(ad_ns_dof_handler, ad_ns_constraints);
            VectorTools::interpolate_boundary_values(ad_ns_dof_handler,
                                                     boundary_values,
                                                     ad_ns_constraints,
                                                     ad_ns_fe.component_mask(velocities));

            std::set<types::boundary_id> no_normal_flux_boundaries = {3};
            VectorTools::compute_no_normal_flux_constraints(ad_ns_dof_handler,
                                                            0,
                                                            no_normal_flux_boundaries,
                                                            ad_ns_constraints);
        }
        ad_ns_constraints.close();

        {
            top_constraints.reinit(top_owned_partitioning, top_relevant_partitioning);
            DoFTools::make_hanging_node_constraints(top_dof_handler, top_constraints);
        }
        top_constraints.close();

        {
            top_equivalent_constraints.reinit(top_equivalent_owned_partitioning, top_equivalent_relevant_partitioning);
            DoFTools::make_hanging_node_constraints(top_equivalent_dof_handler, top_equivalent_constraints);
        }
        top_equivalent_constraints.close();

        {
            filter_constraints.reinit(filter_owned_partitioning, filter_relevant_partitioning);
            DoFTools::make_hanging_node_constraints(filter_dof_handler, filter_constraints);
        }
        filter_constraints.close();

        pcout << std::endl
              << "******************************************************************************" << std::endl;
        pcout << "The diameter of the largest active cell:   "
              << GridTools::maximal_cell_diameter(triangulation) << std::endl
              << "The diameter of the smallest active cell:   "
              << GridTools::minimal_cell_diameter(triangulation) << std::endl;

        pcout << "Number of triangulation Levels (0 ~ N-1): " << triangulation.n_levels() << std::endl
              << "Number of active cells: " << triangulation.n_global_active_cells() << std::endl
              << "Number of degrees of freedom for NS: " << ns_dof_handler.n_dofs() << std::endl
              << "Number of degrees of freedom for Temp: " << temp_dof_handler.n_dofs() << std::endl
              << "Number of degrees of freedom for Topology: " << top_dof_handler.n_dofs() << std::endl;
    }

    void TopOptHeatFlow::initialize_system()
    {
        TimerOutput::Scope t(computing_timer, "initialize");

        /*NS*/
        {
            DynamicSparsityPattern dsp(ns_relevant_partitioning);
            DoFTools::make_sparsity_pattern(ns_dof_handler, dsp, ns_zero_constraints);
            SparsityTools::distribute_sparsity_pattern(
                dsp,
                ns_owned_partitioning,
                mpi_communicator,
                ns_relevant_partitioning);
            ns_system_matrix.reinit(ns_owned_partitioning, ns_owned_partitioning, dsp, mpi_communicator);
        }

        ns_solution_distributed.reinit(ns_owned_partitioning, mpi_communicator);
        ns_solution_relevant.reinit(ns_owned_partitioning, ns_relevant_partitioning, mpi_communicator);

        ns_newton_update_distributed.reinit(ns_owned_partitioning, mpi_communicator);
        ns_newton_update_relevant.reinit(ns_owned_partitioning, ns_relevant_partitioning, mpi_communicator);

        ns_evaluation_distributed.reinit(ns_owned_partitioning, mpi_communicator);
        ns_evaluation_relevant.reinit(ns_owned_partitioning, ns_relevant_partitioning, mpi_communicator);

        ns_system_rhs.reinit(ns_owned_partitioning, mpi_communicator);

        ns_nonzero_constraints.distribute(ns_solution_distributed);
        ns_nonzero_constraints.distribute(ns_evaluation_distributed);
        ns_solution_relevant = ns_solution_distributed;
        ns_evaluation_relevant = ns_evaluation_distributed;

        {
            DynamicSparsityPattern dsp(temp_relevant_partitioning);
            DoFTools::make_sparsity_pattern(temp_dof_handler, dsp, temp_constraints);
            SparsityTools::distribute_sparsity_pattern(
                dsp,
                temp_dof_handler.locally_owned_dofs(),
                mpi_communicator,
                DoFTools::extract_locally_relevant_dofs(temp_dof_handler));
            temp_system_matrix.reinit(temp_owned_partitioning, dsp, mpi_communicator);
        }

        temp_solution_distributed.reinit(temp_owned_partitioning, mpi_communicator);
        temp_solution_relevant.reinit(temp_owned_partitioning, temp_relevant_partitioning, mpi_communicator);

        temp_system_rhs.reinit(temp_owned_partitioning, mpi_communicator);

        /*For Adjoint Temp*/
        {
            DynamicSparsityPattern dsp(ad_temp_relevant_partitioning);
            DoFTools::make_sparsity_pattern(ad_temp_dof_handler, dsp, ad_temp_constraints);
            SparsityTools::distribute_sparsity_pattern(
                dsp,
                ad_temp_dof_handler.locally_owned_dofs(),
                mpi_communicator,
                DoFTools::extract_locally_relevant_dofs(ad_temp_dof_handler));
            ad_temp_system_matrix.reinit(ad_temp_owned_partitioning, dsp, mpi_communicator);
        }

        ad_temp_solution_distributed.reinit(ad_temp_owned_partitioning, mpi_communicator);
        ad_temp_solution_relevant.reinit(ad_temp_owned_partitioning, ad_temp_relevant_partitioning, mpi_communicator);

        ad_temp_system_rhs.reinit(ad_temp_owned_partitioning, mpi_communicator);

        /*For Adjoint NS*/
        {
            DynamicSparsityPattern dsp(ad_ns_relevant_partitioning);
            DoFTools::make_sparsity_pattern(ad_ns_dof_handler, dsp, ad_ns_constraints);
            SparsityTools::distribute_sparsity_pattern(
                dsp,
                ad_ns_dof_handler.locally_owned_dofs(),
                mpi_communicator,
                DoFTools::extract_locally_relevant_dofs(ad_ns_dof_handler));
            ad_ns_system_matrix.reinit(ad_ns_owned_partitioning, dsp, mpi_communicator);
        }

        ad_ns_solution_distributed.reinit(ad_ns_owned_partitioning, mpi_communicator);
        ad_ns_solution_relevant.reinit(ad_ns_owned_partitioning, ad_ns_relevant_partitioning, mpi_communicator);

        ad_ns_flow_solution_distributed.reinit(ad_ns_owned_partitioning, mpi_communicator);
        ad_ns_flow_solution_relevant.reinit(ad_ns_owned_partitioning, ad_ns_relevant_partitioning, mpi_communicator);

        ad_ns_field_solution_distributed.reinit(ad_ns_owned_partitioning, mpi_communicator);
        ad_ns_field_solution_relevant.reinit(ad_ns_owned_partitioning, ad_ns_relevant_partitioning, mpi_communicator);

        ad_ns_system_rhs.reinit(ad_ns_owned_partitioning, mpi_communicator);

        /*For Topology Optimization*/

        xphys_values_distributed.reinit(top_owned_partitioning, mpi_communicator);
        xphys_values_relevant.reinit(top_owned_partitioning, top_relevant_partitioning, mpi_communicator);

        xphys_filter_distributed.reinit(top_owned_partitioning, mpi_communicator);
        xphys_filter_relevant.reinit(top_owned_partitioning, top_relevant_partitioning, mpi_communicator);

        xphys_heaviside_distributed.reinit(top_owned_partitioning, mpi_communicator);
        xphys_heaviside_relevant.reinit(top_owned_partitioning, top_relevant_partitioning, mpi_communicator);

        xphys_equivalent_distributed.reinit(top_equivalent_owned_partitioning, mpi_communicator);
        xphys_equivalent_relevant.reinit(top_equivalent_owned_partitioning, top_equivalent_relevant_partitioning, mpi_communicator);

        sens_obj_distributed.reinit(top_owned_partitioning, mpi_communicator);
        sens_obj_relevant.reinit(top_owned_partitioning, top_relevant_partitioning, mpi_communicator);
        sens_flow_distributed.reinit(top_owned_partitioning, mpi_communicator);
        sens_flow_relevant.reinit(top_owned_partitioning, top_relevant_partitioning, mpi_communicator);
        sens_vol_distributed.reinit(top_owned_partitioning, mpi_communicator);
        sens_vol_relevant.reinit(top_owned_partitioning, top_relevant_partitioning, mpi_communicator);
        sens_field_distributed.reinit(top_owned_partitioning, mpi_communicator);
        sens_field_relevant.reinit(top_owned_partitioning, top_relevant_partitioning, mpi_communicator);

        sens_obj_dheaviside_distributed.reinit(top_owned_partitioning, mpi_communicator);
        sens_obj_dheaviside_relevant.reinit(top_owned_partitioning, top_relevant_partitioning, mpi_communicator);
        sens_flow_dheaviside_distributed.reinit(top_owned_partitioning, mpi_communicator);
        sens_flow_dheaviside_relevant.reinit(top_owned_partitioning, top_relevant_partitioning, mpi_communicator);
        sens_vol_dheaviside_distributed.reinit(top_owned_partitioning, mpi_communicator);
        sens_vol_dheaviside_relevant.reinit(top_owned_partitioning, top_relevant_partitioning, mpi_communicator);
        sens_field_dheaviside_distributed.reinit(top_owned_partitioning, mpi_communicator);
        sens_field_dheaviside_relevant.reinit(top_owned_partitioning, top_relevant_partitioning, mpi_communicator);

        sens_obj_dfilter_distributed.reinit(top_owned_partitioning, mpi_communicator);
        sens_obj_dfilter_relevant.reinit(top_owned_partitioning, top_relevant_partitioning, mpi_communicator);
        sens_flow_dfilter_distributed.reinit(top_owned_partitioning, mpi_communicator);
        sens_flow_dfilter_relevant.reinit(top_owned_partitioning, top_relevant_partitioning, mpi_communicator);
        sens_vol_dfilter_distributed.reinit(top_owned_partitioning, mpi_communicator);
        sens_vol_dfilter_relevant.reinit(top_owned_partitioning, top_relevant_partitioning, mpi_communicator);
        sens_field_dfilter_distributed.reinit(top_owned_partitioning, mpi_communicator);
        sens_field_dfilter_relevant.reinit(top_owned_partitioning, top_relevant_partitioning, mpi_communicator);

        xold1_distributed.reinit(top_owned_partitioning, mpi_communicator);
        xold1_relevant.reinit(top_owned_partitioning, top_relevant_partitioning, mpi_communicator);
        xold2_distributed.reinit(top_owned_partitioning, mpi_communicator);
        xold2_relevant.reinit(top_owned_partitioning, top_relevant_partitioning, mpi_communicator);
        U_distributed.reinit(top_owned_partitioning, mpi_communicator);
        U_relevant.reinit(top_owned_partitioning, top_relevant_partitioning, mpi_communicator);
        L_distributed.reinit(top_owned_partitioning, mpi_communicator);
        L_relevant.reinit(top_owned_partitioning, top_relevant_partitioning, mpi_communicator);

        n_variables_global = xphys_values_distributed.size();

        if (field_constraint_function == PRM::no_field_constraints)
            n_constraints = 2;
        else
            n_constraints = 3;

        xval_petsc = xphys_values_distributed.petsc_vector();
        xold1_petsc = xold1_distributed.petsc_vector();
        xold2_petsc = xold2_distributed.petsc_vector();
        xmax_petsc = U_distributed.petsc_vector();
        xmin_petsc = L_distributed.petsc_vector();

        VecDuplicate(xval_petsc, &dfdx_petsc);

        VecDuplicateVecs(xval_petsc, n_constraints, &dgdx_petsc);

        /*For Filter*/
        {
            DynamicSparsityPattern dsp(filter_relevant_partitioning);
            DoFTools::make_sparsity_pattern(filter_dof_handler, dsp, filter_constraints);
            SparsityTools::distribute_sparsity_pattern(
                dsp,
                filter_dof_handler.locally_owned_dofs(),
                mpi_communicator,
                DoFTools::extract_locally_relevant_dofs(filter_dof_handler));
            filter_system_matrix.reinit(filter_owned_partitioning, dsp, mpi_communicator);
        }
        filter_solution_distributed.reinit(filter_owned_partitioning,
                                           mpi_communicator);
        filter_solution_relevant.reinit(filter_owned_partitioning,
                                        filter_relevant_partitioning,
                                        mpi_communicator);
        filter_system_rhs.reinit(filter_owned_partitioning, mpi_communicator);
    }

}