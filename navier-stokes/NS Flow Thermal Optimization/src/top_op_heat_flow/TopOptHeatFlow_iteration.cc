#include "../../include/top_opt_heat_flow.h"

namespace TopOpt
{
    /*拓扑优化迭代过程*/
    void TopOptHeatFlow::topology_iteration(const double top_min_change,
                                            const unsigned int top_max_n_loops,
                                            const unsigned int top_max_n_refinements,
                                            unsigned int output_interval)
    {
        double current_change = std::numeric_limits<double>::max();

        for (unsigned int top_refinement = 0; top_refinement < top_max_n_refinements + 1; ++top_refinement)
        {
            unsigned int loop = 0;

            while ((current_change > top_min_change) && loop < top_max_n_loops + 1)
            {
                pcout << std::endl
                      << "******************************************************************************" << std::endl;
                pcout << "Topology optimization loop   " << loop << "   with grid rifinement   " << top_refinement << std::endl;
                loop_counter = top_refinement * (top_max_n_loops + 1) + loop;

                bool init_step_in_global_loop = loop_counter == 0;
                bool first_step_in_local_loop = loop == 0;

                pcout << "Topology optimization global loop is " << loop_counter << std::endl;

                update_current_parameters();

                if (init_step_in_global_loop)
                {
                    setup_dofs();
                    initialize_system();
                }
                if (first_step_in_local_loop)
                {
                    reinit_mma(init_step_in_global_loop);
                }

                reset_topology(init_step_in_global_loop);

                bool is_sens = false, assemble_matrix = first_step_in_local_loop;
                filter_assemble(assemble_matrix, xphys_values_relevant);
                filter_solve();
                filter_transfer_to_top(is_sens, xphys_filter_distributed,
                                       xphys_filter_relevant);

                xphys_heaviside_conservation();

                xphys_equivalent();

                ns_nonlinear_solver(init_step_in_global_loop);

                temp_assemble();
                temp_solve();

                ad_continuous_temp_assemble();

                ad_continuous_temp_solve();

                lagrange_function_type = PRM::lagrange_obj_function;
                assemble_matrix = true;

                ad_continuous_ns_assemble_with_residual_linearization(lagrange_function_type, assemble_matrix);
                ad_continuous_ns_solve(lagrange_function_type);

                lagrange_function_type = PRM::lagrange_flow_cstr_function;
                assemble_matrix = false;
                ad_continuous_ns_assemble(lagrange_function_type, assemble_matrix);

                ad_continuous_ns_solve(lagrange_function_type);

                if (field_constraint_function != PRM::no_field_constraints)
                {
                    lagrange_function_type = PRM::lagrange_field_cstr_function;
                    assemble_matrix = false;
                    ad_continuous_ns_assemble(lagrange_function_type, assemble_matrix);

                    ad_continuous_ns_solve(lagrange_function_type);
                }

                sens_assemble();

                is_sens = true;
                assemble_matrix = false;

                filter_assemble(assemble_matrix, sens_obj_relevant);
                filter_solve();
                filter_transfer_to_top(is_sens, sens_obj_dfilter_distributed,
                                       sens_obj_dfilter_relevant);

                filter_assemble(assemble_matrix, sens_flow_relevant);
                filter_solve();
                filter_transfer_to_top(is_sens, sens_flow_dfilter_distributed,
                                       sens_flow_dfilter_relevant);

                filter_assemble(assemble_matrix, sens_vol_relevant);
                filter_solve();
                filter_transfer_to_top(is_sens, sens_vol_dfilter_distributed,
                                       sens_vol_dfilter_relevant);

                if (field_constraint_function != PRM::no_field_constraints)
                {
                    filter_assemble(assemble_matrix, sens_field_relevant);
                    filter_solve();
                    filter_transfer_to_top(is_sens, sens_field_dfilter_distributed,
                                           sens_field_dfilter_relevant);
                }

                if ((loop % output_interval) == 0)
                {
                    output_results(top_max_n_refinements, loop_counter);
                }

                /*
                 * 示例：只在第 0 代检查。
                 *
                 * 选取物理区域：
                 * x ∈ [2 mm, 3 mm]
                 * y ∈ [5 mm, 6 mm]
                 *
                 * 最多检查 1000 个单元。
                 */

                topology_update(current_change);

                pcout << std::endl
                      << "******************************************************************************" << std::endl;
                computing_timer.print_summary();

                pcout << std::endl;

                /* 流动约束收敛太快时，因此适当关闭流动约束 */
                loop++;
            }

            if (top_refinement < top_max_n_refinements)
            {
                refine_mesh();
            }
        }
    }

}