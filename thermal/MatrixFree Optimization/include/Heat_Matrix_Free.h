#pragma once
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/generic_linear_algebra.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/solution_transfer.h>

#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/fe_evaluation.h>

#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_matrix.h>

#include "Deal_II_MMA.h"

namespace Heat_Matrix_Free
{
	using namespace dealii;
	const unsigned int dimension = 3;
	const unsigned int degree_finite_element = 1;
	const unsigned int filter_degree_finite_element = 1;
	const double epsimin = 1e-2;
	const double penal = 3.;
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/**
	 * @brief Class representing a matrix-free Laplace operator.
	 *
	 * This class inherits from MatrixFreeOperators::Base and provides matrix-free operations related to the Laplace operator.
	 *
	 * @tparam dim Dimension of the problem.
	 * @tparam fe_degree Degree of the finite element.
	 * @tparam number Numerical type used for computation.
	 */
	template <int dim, int fe_degree, typename number>
	class Laplace_Matrix_Free : public MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<number>>
	{
	public:
		/**
		 * @brief Type alias for the value type.
		 */
		using value_type = number;

		/**
		 * @brief Constructor for the Laplace_Matrix_Free class.
		 */
		Laplace_Matrix_Free();

		/**
		 * @brief Clear internal data structures.
		 */
		void clear() override;

		/**
		 * @brief Evaluate the coefficients of the Laplace operator.
		 *
		 * @param Simp_Rho Vector representing the coefficients to be evaluated.
		 */
		void evaluate_coefficient(LinearAlgebra::distributed::Vector<number> &Simp_Rho,
		                          Tensor<2, dim, VectorizedArray<number>> Material_Thermal_Conductivity_Matrix);

		/**
		 * @brief Compute the diagonal of the matrix-free Laplace operator.
		 */
		virtual void compute_diagonal() override;

		/**
		 * @brief Table storing the coefficient tensors of the Laplace matrix after adding the interpolation model.
		 */
		Table<2, Tensor<2, dim, VectorizedArray<number>>> coefficient;

	private:
		/**
		 * @brief Apply the matrix-free Laplace operator and add the result to the destination vector.
		 *
		 * @param dst Destination vector to which the result is added.
		 * @param src Source vector on which the operation is applied.
		 */
		virtual void apply_add(LinearAlgebra::distributed::Vector<number> &dst,
							   const LinearAlgebra::distributed::Vector<number> &src) const override;

		/**
		 * @brief Apply the local matrix-free Laplace operator.
		 *
		 * @param data Matrix-free data structure.
		 * @param dst Destination vector to which the result is added.
		 * @param src Source vector on which the operation is applied.
		 * @param cell_range Range of cells on which the operation is applied.
		 */
		void local_apply(const MatrixFree<dim, number> &data,
						 LinearAlgebra::distributed::Vector<number> &dst,
						 const LinearAlgebra::distributed::Vector<number> &src,
						 const std::pair<unsigned int, unsigned int> &cell_range) const;

		/**
		 * @brief Compute the local diagonal of the matrix-free Laplace operator.
		 *
		 * @param data Matrix-free data structure.
		 * @param dst Destination vector for computing the diagonal.
		 * @param dummy A dummy parameter.
		 * @param cell_range Range of cells for computing the diagonal.
		 */
		void local_compute_diagonal(const MatrixFree<dim, number> &data,
									LinearAlgebra::distributed::Vector<number> &dst,
									const unsigned int &dummy,
									const std::pair<unsigned int, unsigned int> &cell_range) const;
	};
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////
	template <int dim, int fe_degree, typename number>
	Laplace_Matrix_Free<dim, fe_degree, number>::Laplace_Matrix_Free()
			: MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<number>>()
	{}

	template <int dim, int fe_degree, typename number>
	void Laplace_Matrix_Free<dim, fe_degree, number>::clear()
	{
		coefficient.reinit(0, 0);
		MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<number>>::clear();
	}

	template <int dim, int fe_degree, typename number>
	void Laplace_Matrix_Free<dim, fe_degree, number>::evaluate_coefficient(LinearAlgebra::distributed::Vector<number> &Simp_Rho,
																			Tensor<2, dim, VectorizedArray<number>> Material_Thermal_Conductivity_Matrix)
	{
		const unsigned int n_cells = this->data->n_cell_batches();
		FEEvaluation<dim,fe_degree,fe_degree+1,1,number> fe_values(*this->data);
		coefficient.reinit(n_cells, 1);
		for (unsigned int cell = 0; cell < n_cells; cell++)
		{
			fe_values.reinit(cell);
			fe_values.read_dof_values_plain(Simp_Rho);
			coefficient(cell,0) = epsimin * Material_Thermal_Conductivity_Matrix + 
								  pow<number>(fe_values.get_dof_value(0), penal) * 
								  (Material_Thermal_Conductivity_Matrix * (1. - epsimin));
		}
	}

	template <int dim, int fe_degree, typename number>
	void Laplace_Matrix_Free<dim, fe_degree, number>::compute_diagonal()
	{
		this->inverse_diagonal_entries.reset(new DiagonalMatrix<LinearAlgebra::distributed::Vector<number>>());
		LinearAlgebra::distributed::Vector<number> &inverse_diagonal = this->inverse_diagonal_entries->get_vector();
		this->data->initialize_dof_vector(inverse_diagonal);
		unsigned int dummy = 0;
		this->data->cell_loop(&Laplace_Matrix_Free::local_compute_diagonal,
								this,
								inverse_diagonal,
								dummy);
		this->set_constrained_entries_to_one(inverse_diagonal);
		for (unsigned int i = 0; i < inverse_diagonal.locally_owned_size(); i++)
		{
			inverse_diagonal.local_element(i) = 1./inverse_diagonal.local_element(i);
		}
	}

	template <int dim, int fe_degree, typename number>
	void Laplace_Matrix_Free<dim, fe_degree, number>::apply_add(LinearAlgebra::distributed::Vector<number> &dst,
														const LinearAlgebra::distributed::Vector<number> &src) const
	{
		this->data->cell_loop(&Laplace_Matrix_Free::local_apply, this, dst, src);
	}

	template <int dim, int fe_degree, typename number>
	void Laplace_Matrix_Free<dim, fe_degree, number>::local_apply(const MatrixFree<dim, number> &data,
													LinearAlgebra::distributed::Vector<number> &dst,
													const LinearAlgebra::distributed::Vector<number> &src,
													const std::pair<unsigned int, unsigned int> &cell_range) const
	{
		FEEvaluation<dim,fe_degree,fe_degree+1,1,number> fe_values(data);
		for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
		{
			fe_values.reinit(cell);
			fe_values.read_dof_values(src);
			fe_values.evaluate(EvaluationFlags::gradients);
			for (const unsigned int q_point : fe_values.quadrature_point_indices())
			{
				fe_values.submit_gradient(coefficient(cell, 0) * fe_values.get_gradient(q_point), q_point);              
			}
			fe_values.integrate(EvaluationFlags::gradients);
			fe_values.distribute_local_to_global(dst);
		}
	}
	template <int dim, int fe_degree, typename number>
	void Laplace_Matrix_Free<dim, fe_degree, number>::local_compute_diagonal(const MatrixFree<dim, number> &data,
														LinearAlgebra::distributed::Vector<number> &dst,
														const unsigned int &,
														const std::pair<unsigned int, unsigned int> &cell_range) const
	{
		FEEvaluation<dim,fe_degree,fe_degree+1,1,number> fe_values(data);
		AlignedVector<VectorizedArray<number>> diagonal(fe_values.dofs_per_cell);
		for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
		{
			fe_values.reinit(cell);
			for (unsigned int i = 0; i < fe_values.dofs_per_cell; ++i)
			{
				for (unsigned int j = 0; j < fe_values.dofs_per_cell; ++j)
				{
					fe_values.submit_dof_value(VectorizedArray<number>(), j);                    
				}
				fe_values.submit_dof_value(make_vectorized_array<number>(1.), i);

				fe_values.evaluate(EvaluationFlags::gradients);
				for (const unsigned int q : fe_values.quadrature_point_indices())
				{
					fe_values.submit_gradient(coefficient(cell, 0) * fe_values.get_gradient(q),q);                    
				}
				fe_values.integrate(EvaluationFlags::gradients);
				diagonal[i] = fe_values.get_dof_value(i);
			}
			for (unsigned int i = 0; i < fe_values.dofs_per_cell; ++i)
			{
				fe_values.submit_dof_value(diagonal[i], i);                
			}
			fe_values.distribute_local_to_global(dst);
		}
	}
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Helmholtz matrix-free method
	template <int dim, int fe_degree, typename number>
	class Helmholtz_Matrix_Free : public MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<number>>
	{
	public : 
		using value_type = number;

		Helmholtz_Matrix_Free();

		void clear() override;

		virtual void compute_diagonal() override;

		number Rmin_Pow2;
	private :
		virtual void apply_add(LinearAlgebra::distributed::Vector<number> &dst,
							   const LinearAlgebra::distributed::Vector<number> &src) const override;

		void local_apply(const MatrixFree<dim, number> &data,
						 LinearAlgebra::distributed::Vector<number> &dst,
						 const LinearAlgebra::distributed::Vector<number> &src,
						 const std::pair<unsigned int, unsigned int> &cell_range) const;

		void local_compute_diagonal(const MatrixFree<dim, number> &data,
									LinearAlgebra::distributed::Vector<number> &dst,
									const unsigned int &dummy,
									const std::pair<unsigned int, unsigned int> &cell_range) const;
	};

	template <int dim, int fe_degree, typename number>
	Helmholtz_Matrix_Free<dim, fe_degree, number>::Helmholtz_Matrix_Free()
	: MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<number>>()
	{}

	template <int dim, int fe_degree, typename number>
	void  Helmholtz_Matrix_Free<dim, fe_degree, number>::clear() 
	{
		MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<number>>::clear();
	}

	template <int dim, int fe_degree, typename number>
	void Helmholtz_Matrix_Free<dim, fe_degree, number>::compute_diagonal()
	{
		this->inverse_diagonal_entries.reset(new DiagonalMatrix<LinearAlgebra::distributed::Vector<number>>());
		LinearAlgebra::distributed::Vector<number> &inverse_diagonal = this->inverse_diagonal_entries->get_vector();
		this->data->initialize_dof_vector(inverse_diagonal);
		unsigned int dummy = 0;
		this->data->cell_loop(&Helmholtz_Matrix_Free::local_compute_diagonal,
							this,
							inverse_diagonal,
							dummy);
		this->set_constrained_entries_to_one(inverse_diagonal);
		for (unsigned int i = 0; i < inverse_diagonal.locally_owned_size(); i++)
		{
			inverse_diagonal.local_element(i) = 1./inverse_diagonal.local_element(i);
		}
	}

	template <int dim, int fe_degree, typename number>
	void Helmholtz_Matrix_Free<dim, fe_degree, number>::apply_add(LinearAlgebra::distributed::Vector<number> &dst,
															const LinearAlgebra::distributed::Vector<number> &src) const
	{
		this->data->cell_loop(&Helmholtz_Matrix_Free::local_apply, this, dst, src);
	}

	template <int dim, int fe_degree, typename number>
	void Helmholtz_Matrix_Free<dim, fe_degree, number>::local_apply(const MatrixFree<dim, number> &data,
											LinearAlgebra::distributed::Vector<number> &dst,
											const LinearAlgebra::distributed::Vector<number> &src,
											const std::pair<unsigned int, unsigned int> &cell_range) const
	{
		FEEvaluation<dim,fe_degree,fe_degree+1,1,number> fe_values(data);
		for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
		{
			fe_values.reinit(cell);
			fe_values.read_dof_values(src);
			fe_values.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
			for (const unsigned int q : fe_values.quadrature_point_indices())
			{
				fe_values.submit_value(fe_values.get_value(q),q);
				fe_values.submit_gradient(Rmin_Pow2 * fe_values.get_gradient(q),q);                    
			}
			fe_values.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
			fe_values.distribute_local_to_global(dst);
		}
	}

	template <int dim, int fe_degree, typename number>
	void Helmholtz_Matrix_Free<dim, fe_degree, number>::local_compute_diagonal(const MatrixFree<dim, number> &data,
														LinearAlgebra::distributed::Vector<number> &dst,
														const unsigned int &,
														const std::pair<unsigned int, unsigned int> &cell_range) const
	{
		FEEvaluation<dim,fe_degree,fe_degree+1,1,number> fe_values(data);
		AlignedVector<VectorizedArray<number>> diagonal(fe_values.dofs_per_cell);
		for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
		{
			fe_values.reinit(cell);
			for (unsigned int i = 0; i < fe_values.dofs_per_cell; ++i)
			{
				for (unsigned int j = 0; j < fe_values.dofs_per_cell; ++j)
				{
					fe_values.submit_dof_value(VectorizedArray<number>(), j);                    
				}
				fe_values.submit_dof_value(make_vectorized_array<number>(1.), i);

				fe_values.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
				for (const unsigned int q : fe_values.quadrature_point_indices())
				{
					fe_values.submit_value(fe_values.get_value(q),q);
					fe_values.submit_gradient(Rmin_Pow2 * fe_values.get_gradient(q),q);                    
				}
				fe_values.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
				diagonal[i] = fe_values.get_dof_value(i);
			}
			for (unsigned int i = 0; i < fe_values.dofs_per_cell; ++i)
			{
				fe_values.submit_dof_value(diagonal[i], i);                
			}
			fe_values.distribute_local_to_global(dst);
		}  
	}
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Main class for computation
	class Heat_Solver
	{
	public:
		Heat_Solver();
		void run();

	private:
		// Generate mesh
		void make_grid();
		// Laplace
		void setup_laplace_system();
		void setup_multigrid_laplace_system();
		// Filter
		void setup_filter_system();
		void setup_multigrid_filter_system();
		// SIMP
		void setup_simp_system();			
		// Solve filter field
		void assemble_filter_rhs(LinearAlgebra::distributed::Vector<double> input);
		void solve_filter_chebyshev();
		void generate_average_vector(LinearAlgebra::distributed::Vector<double> &output);
		// Laplace solve and sensitivity		
		void interpolate_simp_vector();		
		void transfer_simp_vector();
		void assemble_laplace_rhs();
		void solve_chebyshev();
		void generate_object_discrete_derivative();
		// 
		double mma_optimizer(Vector<double> input);
		void get_cell_volume();
		void output_results();
		void time();
		// 
		MPI_Comm mpi_communicator;
		parallel::distributed::Triangulation<dimension> triangulation;
		// Mapping relation
		MappingFE<dimension> mapping;
		// 
		MappingFE<dimension> mapping_filter;		
		// Heat transfer finite element
		const FE_Q<dimension> fe;
		// Design variable finite element
		const FE_DGQ<dimension> fe_simp;
		// Filter finite element
		const FE_Q<dimension> fe_filter;
		// Degree of freedom handling
		DoFHandler<dimension> dof_handler;
		// Design variable field
		DoFHandler<dimension> dof_handler_simp;
		// Filter field
		DoFHandler<dimension> dof_handler_filter;
		// Handle periodic boundary conditions and single-point constraints
		AffineConstraints<double> constraints;
		// Constraint handling for filter field
		AffineConstraints<double> constraints_filter;
		// 
		using VectorType = LinearAlgebra::distributed::Vector<double>;
		using LeveLVectorType = LinearAlgebra::distributed::Vector<float>;

		// Define matrix types
		using LaplaceMatrixType = Laplace_Matrix_Free<dimension, degree_finite_element, double>;
		using LeveLLaplaceMatrixType = Laplace_Matrix_Free<dimension, degree_finite_element, float>;		
		// Laplace matrix for heat transfer
		LaplaceMatrixType laplace_matrix;
		// Right-hand side formed by the test heat flux of the entire structure
		VectorType        laplace_rhs;
		// Temperature distribution obtained by solving the test strain of the entire structure
		VectorType        laplace_temperature;
		// Constraints on the multigrid for the heat transfer field
		MGConstrainedDoFs 		  mg_constrained_dofs;
		// Matrices on the multigrid for the heat transfer field
		MGLevelObject<LeveLLaplaceMatrixType> mg_matrices;

		// Matrix for filter field
		using FilterMatrixType = Helmholtz_Matrix_Free<dimension, filter_degree_finite_element, double>;
		using LeveLFilterMatrixType = Helmholtz_Matrix_Free<dimension, filter_degree_finite_element, float>;
		FilterMatrixType filter_matrix;
		// Right-hand side of filter field
		VectorType       filter_rhs;
		// Solution of filter field
		VectorType       filter_solution;
		// Constraints on the multigrid for the filter field
		MGConstrainedDoFs 		  mg_constrained_dofs_filter;
		// Matrices on the multigrid for the filter field
		MGLevelObject<LeveLFilterMatrixType> mg_matrices_filter;

		// Continuous design variables, taking only the value at the first quadrature point; this vector is used for solving the physical field
		VectorType 			           Simp_Rho_Continuous;
		// Continuous design variables on the multigrid levels
		MGLevelObject<LeveLVectorType> Simp_Rho_Continuous_Level;		

		// Discrete design variables
		VectorType                     Simp_Rho_Discrete;
		// Discrete design variables
		VectorType                     Simp_Rho_Continuous_Filter;
		// Filtered discrete design variables
		VectorType                     Simp_Rho_Discrete_Filter;
		// Discrete design variables
		VectorType                     Simp_Rho_Discrete_Max;
		// Discrete design variables
		VectorType                     Simp_Rho_Discrete_Min;

		// Derivative of the objective function
		VectorType                     Object_Discrete_Derivative;
		// Derivative of the constraint function
		std::vector<VectorType>        Constraint_Discrete_Derivative;

		// Cell volume
		VectorType 			           Cell_Volume_Discrete;

		// Thermal conductivity properties of the material
		Tensor<2, dimension, double>  SumMaterial_Thermal_Conductivity_Matrix;
		Tensor<2, dimension, VectorizedArray<float>>   LeveLMaterial_Thermal_Conductivity_Matrix;
		Tensor<2, dimension, VectorizedArray<double>>  Material_Thermal_Conductivity_Matrix;

		// MMA optimizer
		Deal_II_MMA_Solver             MMA_Optimizer;

		// MPI output
		ConditionalOStream pcout;
		// Timer
		TimerOutput computing_timer;
		// Filter radius
		double Rmin;
	};
};