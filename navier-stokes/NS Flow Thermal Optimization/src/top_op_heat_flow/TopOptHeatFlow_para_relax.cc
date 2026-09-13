#include "../../include/top_opt_heat_flow.h"

namespace TopOpt
{

    void TopOptHeatFlow::update_current_parameters()
    {
        Da_refer = relax_da_refer.parameter_update(loop_counter);
        heaviside_beta = relax_heaviside_beta.parameter_update(loop_counter);
        heaviside_beta_equivalent = relax_heaviside_beta_equivalent.parameter_update(loop_counter);

        switch (PRM::dimensionless_method)
        {
        case PRM::dimensionless_on:
        {
            permeability_solid = (1.0 + (1.0 / Re_refer)) / Da_refer;
            break;
        }

        case PRM::dimensionless_off:
        {
            permeability_solid = (1.0 + (1.0 / Re_refer)) / Da_refer *
                                 rho_refer * U_refer / L_refer;
            break;
        }

        default:
            break;
        }
        material_interpolate = MaterialInterpolate(permeability_fluid, permeability_solid,
                                                   permeability_penal, thermal_cond_fluid,
                                                   thermal_cond_solid, thermal_cond_penal,
                                                   heatsource_coeff_solid, heatsource_coeff_penal,
                                                   convection_coeff_fluid, convection_coeff_penal);
        heaviside = Heaviside(heaviside_beta, heaviside_yita);
        heaviside_equivalent = Heaviside(heaviside_beta_equivalent, heaviside_yita_equivalent);

        pcout << std::endl
              << "******************************************************************************" << std::endl;
        pcout << "Parameter relaxation...." << std::endl
              << "Da = " << Da_refer
              << "     Heavside beta = " << heaviside_beta
              << "     Heavside beta equivalent = " << heaviside_beta_equivalent << std::endl;
    }

}