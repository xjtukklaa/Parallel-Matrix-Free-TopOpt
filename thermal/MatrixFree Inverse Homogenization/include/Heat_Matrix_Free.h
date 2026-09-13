#pragma once
#include <stdlib.h>
#include <fstream>
#include <iostream>

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
	const unsigned int dimension = 2;
	const unsigned int degree_finite_element = 1;
	const unsigned int filter_degree_finite_element = 1;
	const double epsimin = 1e-2;
	const double penal = 5.;
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/**
	 * @brief 表示无矩阵拉普拉斯算子的类。
	 *
	 * 该类继承自 MatrixFreeOperators::Base，并提供与拉普拉斯算子相关的无矩阵操作功能。
	 *
	 * @tparam dim 问题的维度。
	 * @tparam fe_degree 有限元的阶数。
	 * @tparam number 用于计算的数值类型。
	 */
	template <int dim, int fe_degree, typename number>
	class Laplace_Matrix_Free : public MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<number>>
	{
	public:
		/**
		 * @brief 值类型的类型别名。
		 */
		using value_type = number;

		/**
		 * @brief Laplace_Matrix_Free 类的构造函数。
		 */
		Laplace_Matrix_Free();

		/**
		 * @brief 清除内部数据结构。
		 */
		void clear() override;

		/**
		 * @brief 评估拉普拉斯算子的系数。
		 *
		 * @param Simp_Rho 表示要评估的系数的向量。
		 */
		void evaluate_coefficient(LinearAlgebra::distributed::Vector<number> &Simp_Rho,
		                          Tensor<2, dim, VectorizedArray<number>> Material_Thermal_Conductivity_Matrix);

		/**
		 * @brief 计算无矩阵拉普拉斯算子的对角线。
		 */
		virtual void compute_diagonal() override;

		/**
		 * @brief 存储添加插值模型后的拉普拉斯矩阵系数张量的表。
		 */
		Table<2, Tensor<2, dim, VectorizedArray<number>>> coefficient;

	private:
		/**
		 * @brief 应用无矩阵拉普拉斯算子并将结果添加到目标向量。
		 *
		 * @param dst 添加结果的目标向量。
		 * @param src 应用操作的源向量。
		 */
		virtual void apply_add(LinearAlgebra::distributed::Vector<number> &dst,
							   const LinearAlgebra::distributed::Vector<number> &src) const override;

		/**
		 * @brief 应用局部无矩阵拉普拉斯算子。
		 *
		 * @param data 无矩阵数据结构。
		 * @param dst 添加结果的目标向量。
		 * @param src 应用操作的源向量。
		 * @param cell_range 应用操作的单元格范围。
		 */
		void local_apply(const MatrixFree<dim, number> &data,
						 LinearAlgebra::distributed::Vector<number> &dst,
						 const LinearAlgebra::distributed::Vector<number> &src,
						 const std::pair<unsigned int, unsigned int> &cell_range) const;

		/**
		 * @brief 计算无矩阵拉普拉斯算子的局部对角线。
		 *
		 * @param data 无矩阵数据结构。
		 * @param dst 计算对角线的目标向量。
		 * @param dummy 一个虚拟参数。
		 * @param cell_range 计算对角线的单元格范围。
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
			coefficient(cell,0) = Material_Thermal_Conductivity_Matrix * epsimin + 
								  pow<number>(fe_values.get_dof_value(0), penal) * 
								  (Material_Thermal_Conductivity_Matrix * (1.-epsimin));
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
														const unsigned int &dummy,
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
	// Helmholtz无矩阵方法
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
														const unsigned int &dummy,
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
	// 计算的主类
	class Heat_Solver
	{
	public:
		Heat_Solver();
		void run();

	private:
		// 划分网格
		void make_grid();
		// laplace
		void setup_laplace_system();
		void setup_multigrid_laplace_system();
		// filter
		void setup_filter_system();
		void setup_multigrid_filter_system();
		// simp
		void setup_simp_system();
		void setup_init_simp_value();						
		// 过滤场求解
		void assemble_filter_rhs(LinearAlgebra::distributed::Vector<double> input);
		void solve_filter_chebyshev();
		void generate_average_vector(LinearAlgebra::distributed::Vector<double> &output);
		void generate_solution_laplace(LinearAlgebra::distributed::Vector<double> &input);
		// laplace求解以及敏度		
		void interpolate_simp_vector();		
		void transfer_simp_vector();
		void assemble_laplace_rhs();
		void solve_chebyshev();
		void generate_homogenization_tensor();
		void generate_object_discrete_derivative();
		void keep_boundary_shape(LinearAlgebra::distributed::Vector<double> &input);
		// 
		double mma_optimizer(Vector<double> input);
		void refine_grid();
		void get_cell_volume();
		void output_results();
		void output_simp();
		void time();
		// 
		MPI_Comm mpi_communicator;
		parallel::distributed::Triangulation<dimension> triangulation;
		// 映射关系
		MappingFE<dimension> mapping;
		// 
		MappingFE<dimension> mapping_filter;		
		// 传热有限元
		const FE_Q<dimension> fe;
		// 设计变量有限元
		const FE_DGQ<dimension> fe_simp;
		// 过滤有限元
		const FE_Q<dimension> fe_filter;
		// 自由度处理
		DoFHandler<dimension> dof_handler;
		// 设计变量场
		DoFHandler<dimension> dof_handler_simp;
		// 过滤场
		DoFHandler<dimension> dof_handler_filter;
		// 处理周期性边界条件约束和单点约束
		AffineConstraints<double> constraints;
		// 过滤场的约束处理
		AffineConstraints<double> constraints_filter;
		// 
		using VectorType = LinearAlgebra::distributed::Vector<double>;
		using LeveLVectorType = LinearAlgebra::distributed::Vector<float>;

		// 定义矩阵类型
		using LaplaceMatrixType = Laplace_Matrix_Free<dimension, degree_finite_element, double>;
		using LeveLLaplaceMatrixType = Laplace_Matrix_Free<dimension, degree_finite_element, float>;		
		// 传热的laplace矩阵
		LaplaceMatrixType laplace_matrix;
		// 整个结构的测试热流形成的右手项
		std::vector<VectorType>   unit_test_rhs;
		// 整个结构的测试应变求解得到的温度分布
		std::vector<VectorType>   unit_test_temperature;
		// 传热场多重网格上的约束
		MGConstrainedDoFs 		  mg_constrained_dofs;
		// 传热场多重网格上的矩阵
		MGLevelObject<LeveLLaplaceMatrixType> mg_matrices;

		// 过滤场的矩阵
		using FilterMatrixType = Helmholtz_Matrix_Free<dimension, filter_degree_finite_element, double>;
		using LeveLFilterMatrixType = Helmholtz_Matrix_Free<dimension, filter_degree_finite_element, float>;
		FilterMatrixType filter_matrix;
		// 过滤场的右端项
		VectorType       filter_rhs;
		// 过滤场的解
		VectorType       filter_solution;
		// 过滤场多重网格上的约束
		MGConstrainedDoFs 		  mg_constrained_dofs_filter;
		// 过滤场多重网格上的矩阵
		MGLevelObject<LeveLFilterMatrixType> mg_matrices_filter;

		// 连续的设计变量,只取第一个正交点上的值,这个向量用于进行物理场求解
		VectorType 			           Simp_Rho_Continuous;
		// 多级网格上的连续的设计变量
		MGLevelObject<LeveLVectorType> Simp_Rho_Continuous_Level;		

		// 离散的设计变量
		VectorType                     Simp_Rho_Discrete;
		// 过滤完成后的连续设计变量，用于输出重建网格
		VectorType                     Simp_Rho_Continuous_Filter;
		// 过滤后的离散设计变量
		VectorType                     Simp_Rho_Discrete_Filter;
		// 离散的设计变量
		VectorType                     Simp_Rho_Discrete_Max;
		// 离散的设计变量
		VectorType                     Simp_Rho_Discrete_Min;
		// Simp Laplace
		VectorType                     Simp_Laplace;

		// 目标函数的导数
		VectorType                     Object_Discrete_Derivative;
		// 约束函数的导数
		std::vector<VectorType>        Constraint_Discrete_Derivative;

		// 单元体积
		VectorType 			           Cell_Volume_Discrete;

		// 计算出来的热导率矩阵
		Tensor<2, dimension, double>   Opt_Thermal_Conductivity_Matrix;
		// 作为目标的热导率矩阵
		Tensor<2, dimension, double>   Obj_Thermal_Conductivity_Matrix;

		// 材料的热导率属性
		Tensor<2, dimension, double>  SumMaterial_Thermal_Conductivity_Matrix;
		Tensor<2, dimension, VectorizedArray<float>>   LeveLMaterial_Thermal_Conductivity_Matrix;
		Tensor<2, dimension, VectorizedArray<double>>  Material_Thermal_Conductivity_Matrix;

		// MMA优化器
		Deal_II_MMA_Solver             MMA_Optimizer;

		// MPI输出
		ConditionalOStream pcout;
		// 计时器
		TimerOutput computing_timer;
		// 过滤半径
		double Rmin;
		// 正则化参数
		double neta;
	};
};