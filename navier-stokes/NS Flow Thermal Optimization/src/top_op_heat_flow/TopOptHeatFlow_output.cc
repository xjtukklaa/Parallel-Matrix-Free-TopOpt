#include "../../include/top_opt_heat_flow.h"

namespace TopOpt
{
    /*TopOptHeatFlow::output_results
    该函数与步骤 22中的相同，只是我们为输出文件选择一个名称，该名称还包含雷诺数
    （即当前上下文中粘度的倒数）。*/

    void TopOptHeatFlow::output_results(const unsigned int refinement_cycle,
                                        const unsigned int loop_num)
    {
        TimerOutput::Scope t(computing_timer, "output_results");

        std::vector<std::string> solution_names(dim, "velocity");
        solution_names.emplace_back("pressure");

        std::vector<DataComponentInterpretation::DataComponentInterpretation>
            data_component_interpretation(
                dim, DataComponentInterpretation::component_is_part_of_vector);
        data_component_interpretation.push_back(
            DataComponentInterpretation::component_is_scalar);

        DataOut<dim> data_out;

        data_out.attach_dof_handler(ns_dof_handler);
        data_out.add_data_vector(ns_solution_relevant,
                                 solution_names,
                                 DataOut<dim>::type_dof_data,
                                 data_component_interpretation);

        std::vector<std::string> solution_names_temp(1, "temperature");
        std::vector<DataComponentInterpretation::DataComponentInterpretation>
            data_component_interpretation_temp(
                1, DataComponentInterpretation::component_is_scalar);
        data_out.add_data_vector(temp_dof_handler,
                                 temp_solution_relevant,
                                 solution_names_temp,
                                 data_component_interpretation_temp);

        std::vector<std::string> solution_names_ad_temp(1, "ad_temperature");
        std::vector<DataComponentInterpretation::DataComponentInterpretation>
            data_component_interpretation_ad_temp(
                1, DataComponentInterpretation::component_is_scalar);
        data_out.add_data_vector(ad_temp_dof_handler,
                                 ad_temp_solution_relevant,
                                 solution_names_ad_temp,
                                 data_component_interpretation_ad_temp);

        std::vector<std::string> solution_names_ad_ns(dim, "ad_velocity");
        solution_names_ad_ns.emplace_back("ad_pressure");
        std::vector<DataComponentInterpretation::DataComponentInterpretation>
            data_component_interpretation_ad_ns(
                dim, DataComponentInterpretation::component_is_part_of_vector);
        data_component_interpretation_ad_ns.push_back(
            DataComponentInterpretation::component_is_scalar);
        data_out.add_data_vector(ad_ns_dof_handler,
                                 ad_ns_solution_relevant,
                                 solution_names_ad_ns,
                                 data_component_interpretation_ad_ns);

        std::vector<std::string> solution_names_ad_ns_flow(dim, "ad_ns_flow_velocity");
        solution_names_ad_ns_flow.emplace_back("ad_ns_flow_pressure");
        std::vector<DataComponentInterpretation::DataComponentInterpretation>
            data_component_interpretation_ad_ns_flow(
                dim, DataComponentInterpretation::component_is_part_of_vector);
        data_component_interpretation_ad_ns_flow.push_back(
            DataComponentInterpretation::component_is_scalar);
        data_out.add_data_vector(ad_ns_dof_handler,
                                 ad_ns_flow_solution_relevant,
                                 solution_names_ad_ns_flow,
                                 data_component_interpretation_ad_ns_flow);

        std::vector<DataComponentInterpretation::DataComponentInterpretation>
            data_component_interpretation_top(
                1, DataComponentInterpretation::component_is_scalar);

        data_out.add_data_vector(top_dof_handler,
                                 xphys_values_relevant,
                                 "xphys",
                                 data_component_interpretation_top);

        data_out.add_data_vector(top_dof_handler,
                                 xphys_filter_relevant,
                                 "xphys_filter",
                                 data_component_interpretation_top);

        data_out.add_data_vector(top_dof_handler,
                                 xphys_heaviside_relevant,
                                 "xphys_heaviside",
                                 data_component_interpretation_top);

        data_out.add_data_vector(top_equivalent_dof_handler,
                                 xphys_equivalent_relevant,
                                 "xphys_equivalent",
                                 data_component_interpretation_top);

        data_out.add_data_vector(top_dof_handler,
                                 sens_obj_relevant,
                                 "sens_obj",
                                 data_component_interpretation_top);
        data_out.add_data_vector(top_dof_handler,
                                 sens_flow_relevant,
                                 "sens_cstr",
                                 data_component_interpretation_top);
        data_out.add_data_vector(top_dof_handler,
                                 sens_vol_relevant,
                                 "sens_vol",
                                 data_component_interpretation_top);
        data_out.add_data_vector(top_dof_handler,
                                 sens_field_relevant,
                                 "sens_field",
                                 data_component_interpretation_top);

        data_out.add_data_vector(top_dof_handler,
                                 sens_obj_dheaviside_relevant,
                                 "sens_obj_dheaviside",
                                 data_component_interpretation_top);
        data_out.add_data_vector(top_dof_handler,
                                 sens_flow_dheaviside_relevant,
                                 "sens_cstr_dheaviside",
                                 data_component_interpretation_top);
        data_out.add_data_vector(top_dof_handler,
                                 sens_vol_dheaviside_relevant,
                                 "sens_vol_dheaviside",
                                 data_component_interpretation_top);
        data_out.add_data_vector(top_dof_handler,
                                 sens_field_dheaviside_relevant,
                                 "sens_field_dheaviside",
                                 data_component_interpretation_top);

        data_out.add_data_vector(top_dof_handler,
                                 sens_obj_dfilter_relevant,
                                 "sens_obj_dfilter",
                                 data_component_interpretation_top);
        data_out.add_data_vector(top_dof_handler,
                                 sens_flow_dfilter_relevant,
                                 "sens_cstr_dfilter",
                                 data_component_interpretation_top);
        data_out.add_data_vector(top_dof_handler,
                                 sens_vol_dfilter_relevant,
                                 "sens_vol_dfilter",
                                 data_component_interpretation_top);
        data_out.add_data_vector(top_dof_handler,
                                 sens_field_dfilter_relevant,
                                 "sens_field_dfilter",
                                 data_component_interpretation_top);

        data_out.add_data_vector(top_dof_handler,
                                 xold1_relevant,
                                 "xold1",
                                 data_component_interpretation_top);
        data_out.add_data_vector(top_dof_handler,
                                 xold2_relevant,
                                 "xold2",
                                 data_component_interpretation_top);
        data_out.add_data_vector(top_dof_handler,
                                 U_relevant,
                                 "U",
                                 data_component_interpretation_top);
        data_out.add_data_vector(top_dof_handler,
                                 L_relevant,
                                 "L",
                                 data_component_interpretation_top);

        Vector<float> subdomain(triangulation.n_active_cells());
        for (unsigned int i = 0; i < subdomain.size(); ++i)
            subdomain(i) = triangulation.locally_owned_subdomain();
        data_out.add_data_vector(subdomain, "subdomain");

        unsigned int n_divisions = 0;
        if (top_degree > 1)
            n_divisions = 6;
        else
            n_divisions = 1;

        data_out.build_patches(n_divisions);

        std::stringstream ss1;
        ss1 << std::setw(4) << std::setfill('0') << loop_num;
        std::string file_name = std::to_string(Re_refer) + "_solution_" +
                                "refinement_" + std::to_string(refinement_cycle) +
                                "_loop_" + ss1.str() + ".vtu";

        data_out.write_vtu_in_parallel(file_name, mpi_communicator);
    }

}