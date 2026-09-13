#ifndef STABILIZATION_PARAMETER_H
#define STABILIZATION_PARAMETER_H

#include "../include/dealiipackage.h"

namespace TopOpt
{
    using namespace dealii;

    template <int dim, typename NumberType = double>
    class StabilizationParameter
    {
    public:
        StabilizationParameter();
        void value(const std::vector<Tensor<1, dim, NumberType>> &present_velocity_values,
                   const NumberType kinematic_viscosity,
                   const NumberType element_size,
                   std::vector<NumberType> &tauSUPG,
                   std::vector<NumberType> &tauPSPG,
                   std::vector<NumberType> &vLSIC);
        void value_temperature(
            const std::vector<Tensor<1, dim, NumberType>> &present_velocity_values,
            const NumberType thermal_diffusivity,
            const NumberType element_size,
            std::vector<NumberType> &tauSUPGT);
        void partial(const std::vector<Tensor<1, dim, NumberType>> &present_velocity_values,
                     const NumberType kinematic_viscosity,
                     const NumberType thermal_diffusivity,
                     const NumberType element_size,
                     std::vector<Tensor<1, dim, NumberType>> &delta_tauSUPG,
                     std::vector<Tensor<1, dim, NumberType>> &delta_tauPSPG,
                     std::vector<Tensor<1, dim, NumberType>> &delta_vLSIC,
                     std::vector<Tensor<1, dim, NumberType>> &delta_tauSUPGT);

        void value_from_element_matrix_vector(
            const std::vector<Tensor<1, dim, NumberType>> &present_velocity_values,
            const std::vector<Tensor<2, dim, NumberType>> &present_velocity_gradients,
            const std::vector<NumberType> &permeability_values,
            const std::vector<std::vector<Tensor<1, dim, NumberType>>> &grad_phi_u_scalar,
            const NumberType kinematic_viscosity,
            std::vector<NumberType> &tauSUPG,
            std::vector<NumberType> &tauPSPG,
            std::vector<NumberType> &vLSIC);

        void value_from_element_matrix_vector(
            const std::vector<Tensor<1, dim, NumberType>> &present_velocity_values,
            const std::vector<Tensor<1, dim, NumberType>> &present_temperature_gradients,
            const std::vector<std::vector<Tensor<1, dim, NumberType>>> &grad_phi_t_scalar,
            const NumberType k_divd_rho_cp,
            std::vector<NumberType> &tauSUPGT);

        void value_high_reynolds(const std::vector<Tensor<1, dim, NumberType>> &present_velocity_values,
                                 const std::vector<std::vector<Tensor<1, dim, NumberType>>> &grad_phi_u_scalar,
                                 const NumberType kinematic_viscosity,
                                 const NumberType element_size,
                                 const NumberType u_global,
                                 std::vector<NumberType> &tauSUPG,
                                 std::vector<NumberType> &tauPSPG,
                                 std::vector<NumberType> &vLSIC);

    private:
        NumberType penal;
    };

    template <int dim, typename NumberType>
    StabilizationParameter<dim, NumberType>::StabilizationParameter()
        : penal(2.0)
    {
    }

    template <int dim, typename NumberType>
    void StabilizationParameter<dim, NumberType>::value(
        const std::vector<Tensor<1, dim, NumberType>> &present_velocity_values,
        const NumberType kinematic_viscosity,
        const NumberType element_size,
        std::vector<NumberType> &tauSUPG,
        std::vector<NumberType> &tauPSPG,
        std::vector<NumberType> &vLSIC)
    {

        for (unsigned int i = 0; i < present_velocity_values.size(); ++i)
        {

            NumberType velocity_norm_square = present_velocity_values[i].norm_square();
            tauSUPG[i] = 1.0 / std::sqrt(((4.0 * velocity_norm_square) / (element_size * element_size)) +
                                         ((144.0 * kinematic_viscosity * kinematic_viscosity) /
                                          std::pow(element_size, 4)));
            tauPSPG[i] = tauSUPG[i];
            vLSIC[i] = tauSUPG[i] * velocity_norm_square;
        }
    }

    template <int dim, typename NumberType>
    void StabilizationParameter<dim, NumberType>::value_temperature(
        const std::vector<Tensor<1, dim, NumberType>> &present_velocity_values,
        const NumberType thermal_diffusivity,
        const NumberType element_size,
        std::vector<NumberType> &tauSUPGT)
    {
        for (unsigned int i = 0; i < present_velocity_values.size(); ++i)
        {
            const NumberType velocity_norm_square = present_velocity_values[i].norm_square();
            tauSUPGT[i] = 1.0 / std::sqrt(4.0 * velocity_norm_square / (element_size * element_size) +
                                          144.0 * thermal_diffusivity * thermal_diffusivity /
                                              std::pow(element_size, 4));
        }
    }

    template <int dim, typename NumberType>
    void StabilizationParameter<dim, NumberType>::partial(
        const std::vector<Tensor<1, dim, NumberType>> &present_velocity_values,
        const NumberType kinematic_viscosity,
        const NumberType thermal_diffusivity,
        const NumberType element_size,
        std::vector<Tensor<1, dim, NumberType>> &delta_tauSUPG,
        std::vector<Tensor<1, dim, NumberType>> &delta_tauPSPG,
        std::vector<Tensor<1, dim, NumberType>> &delta_vLSIC,
        std::vector<Tensor<1, dim, NumberType>> &delta_tauSUPGT)
    {
        for (unsigned int i = 0; i < present_velocity_values.size(); ++i)
        {

            std::vector<NumberType> tauSUPG(present_velocity_values.size());
            std::vector<NumberType> tauPSPG(present_velocity_values.size());
            std::vector<NumberType> vLSIC(present_velocity_values.size());
            std::vector<NumberType> tauSUPGT(present_velocity_values.size());
            value(present_velocity_values, kinematic_viscosity, element_size,
                  tauSUPG, tauPSPG, vLSIC);
            value_temperature(present_velocity_values, thermal_diffusivity, element_size, tauSUPGT);

            for (unsigned int i = 0; i < present_velocity_values.size(); ++i)
            {
                const NumberType velocity_norm_square = present_velocity_values[i].norm_square();

                delta_tauSUPG[i] = (-4.0 * std::pow(tauSUPG[i], 3) / (element_size * element_size)) *
                                   present_velocity_values[i];
                delta_tauPSPG[i] = delta_tauSUPG[i];
                delta_vLSIC[i] = velocity_norm_square * delta_tauSUPG[i] +
                                 (2.0 * tauSUPG[i]) * present_velocity_values[i];
                delta_tauSUPGT[i] = (-4.0 * std::pow(tauSUPGT[i], 3) / (element_size * element_size)) *
                                    present_velocity_values[i];
            }
        }
    }

    template <int dim, typename NumberType>
    void StabilizationParameter<dim, NumberType>::value_from_element_matrix_vector(
        const std::vector<Tensor<1, dim, NumberType>> &present_velocity_values,
        const std::vector<Tensor<2, dim, NumberType>> &present_velocity_gradients,
        const std::vector<NumberType> &permeability_values,
        const std::vector<std::vector<Tensor<1, dim, NumberType>>> &grad_phi_u_scalar,
        const NumberType kinematic_viscosity,
        std::vector<NumberType> &tauSUPG,
        std::vector<NumberType> &tauPSPG,
        std::vector<NumberType> &vLSIC)
    {
        Tensor<1, dim, NumberType> grad_norm_u, ru;
        for (unsigned int q = 0; q < present_velocity_values.size(); ++q)
        {
            NumberType velocity_norm = present_velocity_values[q].norm();
            NumberType velocity_norm_square = velocity_norm * velocity_norm;
            NumberType tau1 = 0.0, tau3 = 0.0;

            NumberType hRGN = 0.0;
            grad_norm_u.clear();
            ru.clear();

            for (unsigned int c = 0; c < dim; ++c)
                for (unsigned int d = 0; d < dim; ++d)
                    grad_norm_u[d] += present_velocity_values[q][c] * present_velocity_gradients[q][c][d] /
                                      velocity_norm;
            ru = grad_norm_u / grad_norm_u.norm();

            for (unsigned int i = 0; i < grad_phi_u_scalar[q].size(); ++i)
            {
                tau1 += std::abs(present_velocity_values[q] * grad_phi_u_scalar[q][i]);
                hRGN += std::abs(ru * grad_phi_u_scalar[q][i]);
            }
            tau1 = 1.0 / tau1;

            hRGN = tau1 * 2.0 * velocity_norm;

            tau3 = hRGN * hRGN / (4.0 * kinematic_viscosity);
            NumberType tmp = 0;
            tmp = 1.0 / std::pow(tau1, penal) + 1.0 / std::pow(tau3, penal) +
                  std::pow(permeability_values[q], penal);
            tauSUPG[q] = 1.0 / std::pow(tmp, 1.0 / penal);
            tauPSPG[q] = tauSUPG[q];
            vLSIC[q] = tauSUPG[q] * velocity_norm_square;
        }
    }

    template <int dim, typename NumberType>
    void StabilizationParameter<dim, NumberType>::value_from_element_matrix_vector(
        const std::vector<Tensor<1, dim, NumberType>> &present_velocity_values,
        const std::vector<Tensor<1, dim, NumberType>> &present_temperature_gradients,
        const std::vector<std::vector<Tensor<1, dim, NumberType>>> &grad_phi_t,
        const NumberType k_divd_rho_cp,
        std::vector<NumberType> &tauSUPGT)
    {
        Tensor<1, dim, NumberType> rt;
        for (unsigned int q = 0; q < present_velocity_values.size(); ++q)
        {

            NumberType tau1_t = 0.0, tau3_t = 0.0;

            NumberType hRGNT = 0.0;

            rt = present_temperature_gradients[q] / present_temperature_gradients[q].norm();

            for (unsigned int i = 0; i < grad_phi_t[q].size(); ++i)
            {
                tau1_t += std::abs(present_velocity_values[q] * grad_phi_t[q][i]);
                hRGNT += std::abs(rt * grad_phi_t[q][i]);
            }
            tau1_t = 1.0 / tau1_t;
            hRGNT = 2.0 / hRGNT;
            tau3_t = hRGNT * hRGNT / (4.0 * k_divd_rho_cp);
            NumberType tmp = 0;
            tmp = 1.0 / std::pow(tau1_t, penal) + 1.0 / std::pow(tau3_t, penal);
            tauSUPGT[q] = 1.0 / std::pow(tmp, 1.0 / penal);
        }
    }

    template <int dim, typename NumberType>
    void StabilizationParameter<dim, NumberType>::value_high_reynolds(
        const std::vector<Tensor<1, dim, NumberType>> &present_velocity_values,
        const std::vector<std::vector<Tensor<1, dim, NumberType>>> &grad_phi_u_scalar,
        const NumberType kinematic_viscosity,
        const NumberType element_size,
        const NumberType u_global,
        std::vector<NumberType> &tauSUPG,
        std::vector<NumberType> &tauPSPG,
        std::vector<NumberType> &vLSIC)
    {
        Tensor<1, dim, NumberType> unit_vector;
        for (unsigned int q = 0; q < present_velocity_values.size(); ++q)
        {
            NumberType velocity_norm = present_velocity_values[q].norm();

            NumberType h_upwind = 0.0;
            unit_vector = present_velocity_values[q] / velocity_norm;
            for (unsigned int i = 0; i < grad_phi_u_scalar[q].size(); ++i)
            {
                h_upwind += std::abs(unit_vector * grad_phi_u_scalar[q][i]);
            }
            h_upwind = 2.0 / h_upwind;

            NumberType Re_u = velocity_norm * h_upwind / (2.0 * kinematic_viscosity);
            NumberType Re_u_global = u_global * element_size / (2.0 * kinematic_viscosity);

            if (Re_u < 3)
            {
                tauSUPG[q] = h_upwind / (2.0 * velocity_norm) * (Re_u / 3.0);
            }
            else
            {
                tauSUPG[q] = h_upwind / (2.0 * velocity_norm);
            }

            if (Re_u_global < 3)
            {
                tauPSPG[q] = element_size / (2.0 * u_global) * (Re_u_global / 3.0);
            }
            else
            {
                tauPSPG[q] = element_size / (2.0 * u_global);
            }

            vLSIC[q] = 0.0;
        }
    }
}

#endif