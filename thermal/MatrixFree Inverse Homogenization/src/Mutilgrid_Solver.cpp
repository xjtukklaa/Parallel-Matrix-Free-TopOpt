#include "../include/Heat_Matrix_Free.h"

using namespace Heat_Matrix_Free;

/**
 * @brief 为热求解器设置拉普拉斯系统。
 *
 * 该函数执行以下步骤：
 * - 清除现有的拉普拉斯矩阵。
 * - 分配自由度（DoFs）和多重网格自由度。
 * - 输出单元格和自由度的数量。
 * - 为悬挂节点和周期性边界条件创建约束。
 * - 收集周期性面并应用周期性约束。
 * - 将第一个自由度约束为零。
 * - 初始化拉普拉斯矩阵的MatrixFree系统。
 * - 初始化单元测试的右手边（RHS）和温度向量。
 * - 初始化Simp_Rho_Continuous的自由度向量。
 */
void Heat_Solver::setup_laplace_system()
{
    TimerOutput::Scope t(computing_timer, "setup_laplace_system");
    //
    laplace_matrix.clear();
    //
    dof_handler.distribute_dofs(fe);
    dof_handler.distribute_mg_dofs();
    //
    constraints.clear();
    constraints.reinit(dof_handler.locally_owned_dofs(),
                   DoFTools::extract_locally_relevant_dofs(dof_handler));
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    Tensor<1, dimension> offset;
    std::vector<GridTools::PeriodicFacePair<typename DoFHandler<dimension>::cell_iterator>> periodicity_vector;
    GridTools::collect_periodic_faces(dof_handler,
                                      1,
                                      2,
                                      0,
                                      periodicity_vector,
                                      offset);
    GridTools::collect_periodic_faces(dof_handler,
                                      3,
                                      4,
                                      1,
                                      periodicity_vector,
                                      offset);
    if (dimension == 3)
    {
        GridTools::collect_periodic_faces(dof_handler,
                                          5,
                                          6,
                                          2,
                                          periodicity_vector,
                                          offset);
    }
    DoFTools::make_periodicity_constraints<dimension, dimension>(periodicity_vector, constraints);
    //
    constraints.constrain_dof_to_zero(0);        
    constraints.close();
    //
    typename MatrixFree<dimension, double>::AdditionalData additional_data;
    additional_data.tasks_parallel_scheme = MatrixFree<dimension, double>::AdditionalData::none;
    additional_data.mapping_update_flags = (update_gradients | update_JxW_values | update_quadrature_points | update_values);
    std::shared_ptr<MatrixFree<dimension, double>> system_mf_storage(new MatrixFree<dimension, double>());
    system_mf_storage->reinit(mapping,
                              dof_handler,
                              constraints,
                              QGauss<1>(fe.degree + 1),
                              additional_data);
    laplace_matrix.initialize(system_mf_storage);
    unit_test_rhs.resize(dimension);
    unit_test_temperature.resize(dimension);
    for (unsigned int i = 0; i < dimension; i++)
    {
        laplace_matrix.initialize_dof_vector(unit_test_rhs[i]);
        laplace_matrix.initialize_dof_vector(unit_test_temperature[i]);
    }
    laplace_matrix.initialize_dof_vector(Simp_Rho_Continuous);
}

/**
 * @brief 设置多重网格拉普拉斯系统。
 *
 * 该函数执行以下步骤：
 * - 清除现有的多重网格矩阵。
 * - 初始化多重网格矩阵的层次结构。
 * - 为多重网格上的周期性边界条件创建约束。
 * - 约束第一个自由度为零。
 * - 初始化每个层次的多重网格矩阵。
 * - 为每个层次的矩阵设置周期性约束。
 * - 初始化每个层次的MatrixFree系统。
 * - 初始化每个层次的多重网格矩阵。
 */
void Heat_Solver::setup_multigrid_laplace_system()
{
    TimerOutput::Scope t(computing_timer, "setup_multigrid_laplace_system");
    // 清理多重网格
    mg_matrices.clear_elements();
    mg_constrained_dofs.clear();
    //
    const unsigned int nlevels = triangulation.n_levels();
    mg_matrices.resize(0, nlevels - 1);
    // 多级网格上的周期性边界条件
    mg_constrained_dofs.initialize(dof_handler);
    // 约束0号自由度
    IndexSet boundary_index(1);
    boundary_index.add_index(0);
    for (unsigned int i = 0; i < nlevels; i++)
    {
        mg_constrained_dofs.add_boundary_indices(dof_handler,
                                                 i,
                                                 boundary_index);
    }
    //
    const IdentityMatrix transformation(dof_handler.get_fe().n_dofs_per_face());
    const ComponentMask component_mask;
    const float periodicity_factor = 1.0;
    for (unsigned int level = 0; level < nlevels; level++)
    {    
        AffineConstraints<float> level_constraints(dof_handler.locally_owned_mg_dofs(level),
                                 DoFTools::extract_locally_relevant_level_dofs(dof_handler, level));

        for (const auto &[first_cell, second_cell] :
             dof_handler.get_triangulation().get_periodic_face_map())
        {
            if (first_cell.first->is_artificial_on_level())
                continue;
            if (second_cell.first.first->is_artificial_on_level())
                continue;
            if (first_cell.first->level() != second_cell.first.first->level())
                continue;
            if (first_cell.first->level() != (int)level)
                continue;

            DoFTools::internal::set_periodicity_constraints(
                first_cell.first->as_dof_handler_level_iterator(dof_handler)->face(first_cell.second),
                second_cell.first.first->as_dof_handler_level_iterator(dof_handler)->face(second_cell.first.second),
                transformation,
                level_constraints,
                component_mask,
                second_cell.second,
                periodicity_factor,
                first_cell.first->level());
        }
        for (const types::global_dof_index dof_index : mg_constrained_dofs.get_boundary_indices(level))
        {
            level_constraints.constrain_dof_to_zero(dof_index);
        } 
        level_constraints.close();

        typename MatrixFree<dimension, float>::AdditionalData additional_data;
        additional_data.tasks_parallel_scheme = MatrixFree<dimension, float>::AdditionalData::none;
        additional_data.mapping_update_flags = (update_gradients | update_JxW_values | update_quadrature_points | update_values);
        additional_data.mg_level = level;
        std::shared_ptr<MatrixFree<dimension, float>> mg_mf_storage_level(std::make_shared<MatrixFree<dimension, float>>());
        mg_mf_storage_level->reinit(mapping,
                                    dof_handler,
                                    level_constraints,
                                    QGauss<1>(fe.degree + 1),
                                    additional_data);

        mg_matrices[level].initialize(mg_mf_storage_level,
                                      mg_constrained_dofs,
                                      level);
    }
}

/**
 * @brief 将值从离散向量传输到连续向量。
 *
 * 该函数将 `Simp_Rho_Discrete` 向量中的值传输到 `Simp_Rho_Continuous` 向量中。
 * 它遍历 `dof_handler` 和 `dof_handler_simp` 对象中的所有活动单元，对于每个本地拥有的单元，
 * 它将离散向量中的值添加到连续向量中的相应条目。然后压缩连续向量并更新幽灵值。
 *
 * @note 该函数使用计时器来测量传输操作所需的时间
 * @note 使用的设计变量为离散变量，即 `Simp_Rho_Discrete`，传入连续设计变量为 `Simp_Rho_Continuous`的第一个正交点
 */
void Heat_Solver::transfer_simp_vector()
{
    TimerOutput::Scope t(computing_timer, "transfer_simp_vector");
    Simp_Rho_Continuous = 0;
    //
    std::vector<types::global_dof_index> local_dof_indices(fe.n_dofs_per_cell());
    std::vector<types::global_dof_index> local_rho_dof_indices(fe_simp.n_dofs_per_cell());
    //
    auto cell_iter_begin = dof_handler.begin_active();
    auto cell_iter_end = dof_handler.end();
    auto simp_iter = dof_handler_simp.begin_active();
    //
    for (; cell_iter_begin != cell_iter_end; cell_iter_begin++, simp_iter++)
    {
        if (cell_iter_begin->is_locally_owned())
        {
            cell_iter_begin->get_dof_indices(local_dof_indices);
            simp_iter->get_dof_indices(local_rho_dof_indices);
            Simp_Rho_Continuous[local_dof_indices[0]] += Simp_Rho_Discrete[local_rho_dof_indices[0]];
        }
    }
    Simp_Rho_Continuous.compress(VectorOperation::add);
    Simp_Rho_Continuous.update_ghost_values();
}

/**
 * @brief 在多个网格层次上插值向量。
 *
 * 该函数执行以下步骤：
 * 1. 清除并调整 `Simp_Rho_Continuous_Level` 向量的大小以匹配三角剖分的层次数。
 * 2. 初始化 `mg_matrices` 中每个层次的自由度（DoF）向量。
 * 3. 将当前的设计变量传输到拉普拉斯矩阵中。
 * 4. 使用 `Simp_Rho_Continuous` 向量评估拉普拉斯矩阵的系数。
 * 5. 初始化多重网格传递算子。
 * 6. 将设计变量插值到多重网格层次。
 * 7. 更新每个层次的幽灵值并评估每个多重网格矩阵的系数。
*/
void Heat_Solver::interpolate_simp_vector()
{
    TimerOutput::Scope t(computing_timer, "interpolate_simp_vector");
    //
    const unsigned int nlevels = triangulation.n_levels();
    Simp_Rho_Continuous_Level.clear();
    Simp_Rho_Continuous_Level.resize(0, nlevels - 1);
    for (unsigned int level = 0; level < nlevels; level++)
    {
        mg_matrices[level].initialize_dof_vector(Simp_Rho_Continuous_Level[level]);
    }
    //
    transfer_simp_vector();
    laplace_matrix.evaluate_coefficient(Simp_Rho_Continuous, Material_Thermal_Conductivity_Matrix);
    laplace_matrix.compute_diagonal();
    //
    MGTransferMatrixFree<dimension, float> mg_vector_transfer;
    mg_vector_transfer.build(dof_handler);
    mg_vector_transfer.interpolate_to_mg(dof_handler,
                                         Simp_Rho_Continuous_Level,
                                         Simp_Rho_Continuous);
    for (unsigned int i = 0; i < nlevels; i++)
    {
        Simp_Rho_Continuous_Level[i].update_ghost_values();
        mg_matrices[i].evaluate_coefficient(Simp_Rho_Continuous_Level[i], LeveLMaterial_Thermal_Conductivity_Matrix);
        mg_matrices[i].compute_diagonal();
    }
}

/**
 * @brief 组装拉普拉斯方程的右手边（RHS）。
 *
 * 该函数将 RHS 向量初始化为零，然后遍历所有单元和积分点以计算梯度对 RHS 的贡献。
 * 计算的值随后被积分并分配到全局 RHS 向量中。
 *
 * 该函数使用无矩阵评估来执行组装，这对于大规模问题是高效的。
 *
 * @note 该函数假设 `laplace_matrix` 和 `unit_test_rhs` 已正确初始化，并且 `dimension` 和 `degree_finite_element` 已定义。
 */
void Heat_Solver::assemble_laplace_rhs()
{
    TimerOutput::Scope t(computing_timer, "assemble_laplace_rhs");
    for (unsigned int i = 0; i < dimension; i++)
    {
        unit_test_rhs[i] = 0;
    }
    QGauss<dimension> quadrature_formula(fe.degree + 1);
    FEValues<dimension> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

    unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    
    std::vector<Vector<double>> cell_rhs(dimension);

    Tensor<2, dimension, double> Thermal_Conductivity_Matrix;
    
    std::vector<double> rho_values(1);

    for (unsigned int i = 0; i < dimension; i++)
    {
        cell_rhs[i].reinit(dofs_per_cell);
    }
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    std::vector<types::global_dof_index> local_rho_dof_indices(1);
    //
    auto cell_begin = dof_handler.begin_active();
    auto cell_end = dof_handler.end();
    auto rho_begin = dof_handler_simp.begin_active();
    //
    for (; cell_begin != cell_end; ++cell_begin, ++rho_begin)
    {
        if (cell_begin->is_locally_owned() && cell_begin->level() == rho_begin->level())
        {
            for (unsigned int i = 0; i < dimension; i++)
            {
                cell_rhs[i] = 0;
            }
            fe_values.reinit(cell_begin);
            rho_begin->get_dof_indices(local_rho_dof_indices);
            rho_values[0] = Simp_Rho_Discrete_Filter[local_rho_dof_indices[0]];
            Thermal_Conductivity_Matrix =  epsimin * SumMaterial_Thermal_Conductivity_Matrix +
                                           (1. - epsimin) * SumMaterial_Thermal_Conductivity_Matrix 
                                           * pow(rho_values[0], penal);
            cell_begin->get_dof_indices(local_dof_indices);

            for (unsigned int index = 0; index < dimension; index++)
            {
                for (const unsigned int q_index : fe_values.quadrature_point_indices())
                {
                    for (const unsigned int i : fe_values.dof_indices())
                    {
                        cell_rhs[index][i] += (fe_values.shape_grad(i, q_index) *
                                               Thermal_Conductivity_Matrix[index] *
                                               fe_values.JxW(q_index));
                    }
                }
                constraints.distribute_local_to_global(cell_rhs[index],
                                                       local_dof_indices,
                                                       unit_test_rhs[index]);                
            }
        }
    }
    for (unsigned int i = 0; i < dimension; i++)
    {
        unit_test_rhs[i].compress(VectorOperation::add);
        unit_test_rhs[i].update_ghost_values();
    }
}

/**
 * @brief 使用多重网格方法求解热方程。
 *
 * 该函数使用多重网格求解器设置并求解热方程。
 * 它初始化多重网格传输、平滑器和矩阵，然后使用带有多重网格预处理器的BiCGStab求解器来求解每个维度的系统。
 *
 * 该函数执行以下步骤：
 * 1. 初始化多重网格传输算子。
 * 2. 为多重网格层次结构的每个层次设置平滑器数据。
 * 3. 初始化多重网格平滑器。
 * 4. 设置粗网格求解器。
 * 5. 初始化多重网格矩阵和接口矩阵。
 * 6. 使用边缘矩阵设置多重网格求解器。
 * 7. 使用多重网格求解器创建预处理器。
 * 8. 使用带有多重网格预处理器的BiCGStab求解器求解每个维度的系统。
 *
 * @note 该函数假设在调用此函数之前，必要的数据结构（例如dof_handler、mg_matrices、constraints等）已正确初始化。
 */
void Heat_Solver::solve_chebyshev()
{
    TimerOutput::Scope t(computing_timer, "solve_chebyshev");
    MGTransferMatrixFree<dimension, float> mg_transfer(mg_constrained_dofs);
    mg_transfer.build(dof_handler);

    using SmootherType = PreconditionChebyshev<LeveLLaplaceMatrixType, LeveLVectorType>;
    mg::SmootherRelaxation<SmootherType, LeveLVectorType> mg_smoother;
    MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
    smoother_data.resize(0, triangulation.n_global_levels() - 1);
    for (unsigned int level = 0; level < triangulation.n_global_levels(); ++level)
    {
        if (level > 0)
        {
            smoother_data[level].smoothing_range = 15.;
            smoother_data[level].degree = 5;
            smoother_data[level].eig_cg_n_iterations = 10;
        }
        else
        {
            smoother_data[0].smoothing_range = 1e-3;
            smoother_data[0].degree = numbers::invalid_unsigned_int;
            smoother_data[0].eig_cg_n_iterations = mg_matrices[0].m();
        }
        mg_matrices[level].compute_diagonal();
        smoother_data[level].preconditioner = mg_matrices[level].get_matrix_diagonal_inverse();
    }
    mg_smoother.initialize(mg_matrices, smoother_data);

    MGCoarseGridApplySmoother<LeveLVectorType> mg_coarse;
    mg_coarse.initialize(mg_smoother);

    mg::Matrix<LeveLVectorType> mg_matrix(mg_matrices);

    MGLevelObject<MatrixFreeOperators::MGInterfaceOperator<LeveLLaplaceMatrixType>> mg_interface_matrices;
    mg_interface_matrices.resize(0, triangulation.n_global_levels() - 1);
    for (unsigned int level = 0; level < triangulation.n_global_levels(); ++level)
    {
        mg_interface_matrices[level].initialize(mg_matrices[level]);
    }

    mg::Matrix<LeveLVectorType> mg_interface(mg_interface_matrices);

    Multigrid<LeveLVectorType> mg(mg_matrix, mg_coarse, mg_transfer, mg_smoother, mg_smoother);
    mg.set_edge_matrices(mg_interface, mg_interface);

    PreconditionMG<dimension, LeveLVectorType, MGTransferMatrixFree<dimension, float>> preconditioner(dof_handler, mg, mg_transfer);

    for (unsigned int index = 0; index < dimension; index++)
    {
        SolverControl solve_control(dof_handler.n_dofs(), 1e-12 * unit_test_rhs[index].l2_norm());
        unit_test_temperature[index] = 0;
        SolverFGMRES<VectorType> solver(solve_control);
        solver.solve(laplace_matrix,
                     unit_test_temperature[index],
                     unit_test_rhs[index],
                     preconditioner);
        constraints.distribute(unit_test_temperature[index]);
        unit_test_temperature[index].update_ghost_values();
        // pcout<<"                             "<< "Laplace Solver Last Steps : "<< solve_control.last_step() << std::endl
        //      <<"                             "<< "Laplace Solver Last Value : "<< solve_control.last_value() << std::endl;
    }
}

/**
 * @brief 生成热求解器的均匀化张量。
 *
 * 该函数计算均匀化张量，用于表示异质材料的有效热导率。张量是使用无矩阵有限元评估和温度场的梯度计算的。
 *
 * 该函数执行以下步骤：
 * 1. 初始化最优热导率矩阵的临时张量。
 * 2. 设置有限元评估对象。
 * 3. 遍历矩阵自由对象中的所有单元批次。
 * 4. 对于每个单元批次，重新初始化有限元评估对象。
 * 5. 读取温度场的自由度值，并在积分点处评估梯度。
 * 6. 计算每个积分点对最优热导率矩阵的贡献。
 * 7. 汇总所有积分点和单元的贡献。
 * 8. 在所有 MPI 进程中归约计算的张量。
 * 9. 通过离散单元体积对张量进行归一化。
 *
 * @note 该函数使用向量化数组和 MPI 进行并行计算。
 */
void Heat_Solver::generate_homogenization_tensor()
{
    TimerOutput::Scope t(computing_timer, "generate_homogenization_tensor");
    Opt_Thermal_Conductivity_Matrix = 0;
    //
    Tensor<2, dimension, VectorizedArray<double>> Opt_Thermal_Conductivity_Matrix_Tmp;
    Opt_Thermal_Conductivity_Matrix_Tmp = make_vectorized_array(0.);
    //
    FEEvaluation<dimension, degree_finite_element, degree_finite_element + 1, 1, double> fe_values(*laplace_matrix.get_matrix_free());
    //
    std::vector<Tensor<2, dimension, VectorizedArray<double>>> Temperature_Tensor(fe_values.n_q_points);
    //
    for (unsigned int cell = 0; cell < laplace_matrix.get_matrix_free()->n_cell_batches(); ++cell)
    {
        fe_values.reinit(cell);
        //
        for (unsigned int index = 0; index < dimension; index++)
        {
            fe_values.read_dof_values(unit_test_temperature[index]);
            fe_values.evaluate(EvaluationFlags::gradients);
            for (unsigned int q_point : fe_values.quadrature_point_indices())
            {
                Temperature_Tensor[q_point][index] = fe_values.get_gradient(q_point);
            }
        }
        for (unsigned int q_point : fe_values.quadrature_point_indices())
        {
            Opt_Thermal_Conductivity_Matrix_Tmp += (laplace_matrix.coefficient(cell, 0) - 
                                                    laplace_matrix.coefficient(cell, 0) * Temperature_Tensor[q_point]) *
                                                    fe_values.JxW(q_point);
        }
    }
    for (unsigned int i = 0; i < dimension; i++)
    {
        for (unsigned int j = 0; j < dimension; j++)
        {
            Opt_Thermal_Conductivity_Matrix[i][j] = Opt_Thermal_Conductivity_Matrix_Tmp[i][j].sum();
        }
    }
    Opt_Thermal_Conductivity_Matrix = Utilities::MPI::sum(Opt_Thermal_Conductivity_Matrix, mpi_communicator);
    Opt_Thermal_Conductivity_Matrix.operator/=(Cell_Volume_Discrete.l1_norm());
}

void Heat_Solver::generate_object_discrete_derivative()
{
    TimerOutput::Scope t(computing_timer, "generate_object_discrete_derivative");
    Object_Discrete_Derivative = 0;
    QGauss<dimension> quadrature_formula(fe.degree + 1);
    FEValues<dimension> fe_values(fe,
                                  quadrature_formula,
                                  update_values | update_gradients |
                                  update_quadrature_points | update_JxW_values);

    std::vector<types::global_dof_index> local_simp_dof_indices(fe_simp.n_dofs_per_cell());
    std::vector<double> rho_values(1);

    std::vector<std::vector<Tensor<1, dimension, double>>> solution_gradients(dimension);
    for (unsigned int i = 0; i < dimension; i++)
    {
        solution_gradients[i].resize(fe_values.n_quadrature_points);
    }
    // 
    Tensor<2, dimension, double> Thermal_Conductivity_Derivative_Matrix;
    Tensor<2, dimension, double> Temperature_Tensor;
    Tensor<2, dimension, double> Object_Discrete_Derivative_Cell;
    //
    auto cell_begin = dof_handler.begin_active();
    auto cell_end = dof_handler.end();
    auto base_begin = dof_handler_simp.begin_active();
    //
    for (; cell_begin != cell_end; ++cell_begin, ++base_begin)
    {
        if (cell_begin->is_locally_owned() && cell_begin->level() == base_begin->level())
        {
            fe_values.reinit(cell_begin);
            // Simp
            base_begin->get_dof_indices(local_simp_dof_indices);
            rho_values[0] = Simp_Rho_Discrete_Filter[local_simp_dof_indices[0]];
            Thermal_Conductivity_Derivative_Matrix = penal * (1. - epsimin) * SumMaterial_Thermal_Conductivity_Matrix 
                                                     * pow(rho_values[0], penal - 1); 
            // nabla Temperature
            for (unsigned int i = 0; i < dimension; i++)
            {
                fe_values.get_function_gradients(unit_test_temperature[i], solution_gradients[i]);
            } 
            // 
            Object_Discrete_Derivative_Cell = 0;
            for (unsigned int q_point : fe_values.quadrature_point_indices())
            {
                for (unsigned int i = 0; i < dimension; i++)
                {
                    Temperature_Tensor[i] = solution_gradients[i][q_point];
                }
                Object_Discrete_Derivative_Cell += (Thermal_Conductivity_Derivative_Matrix - 
                                                    2. * Thermal_Conductivity_Derivative_Matrix * Temperature_Tensor +
                                                    Temperature_Tensor * Thermal_Conductivity_Derivative_Matrix * Temperature_Tensor) * 
                                                    fe_values.JxW(q_point);
            }
            // 
            for (unsigned int i = 0; i < dimension; i++)
            {
                for (unsigned int j = 0; j < dimension; j++)
                {
                    if (i == j)
                    {
                    Object_Discrete_Derivative[local_simp_dof_indices[0]] += 
                                    2. * (Opt_Thermal_Conductivity_Matrix[i][j] - Obj_Thermal_Conductivity_Matrix[i][j]) *
                                    Object_Discrete_Derivative_Cell[i][j];                        
                    }
                }
            }
        }
    }
    Object_Discrete_Derivative.compress(VectorOperation::add);
    Object_Discrete_Derivative.operator/=(Cell_Volume_Discrete.l1_norm());
}

void Heat_Solver::refine_grid()
{
	Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
	KellyErrorEstimator<dimension>::estimate(dof_handler,
                                             QGauss<dimension - 1>(fe.degree + 1),
                                             std::map<types::boundary_id, const Function<dimension> *>(),
                                             unit_test_temperature[0],
                                             estimated_error_per_cell);
	parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(triangulation, 
	                                                                       estimated_error_per_cell, 
                                                                          0.3, 
                                                                          0.03);

  if (triangulation.n_levels() > 6)
  {
      for (auto &cell : triangulation.active_cell_iterators_on_level(6))
      {    
          cell->clear_refine_flag();             
      }
  }
  // 细化网格后需要重新插值的向量
  const std::vector<const VectorType *> 
                grid_refine_vectors_in = {&Simp_Rho_Discrete,
                                          &MMA_Optimizer.Moving_Low_Boundary,
                                          &MMA_Optimizer.Moving_Upp_Boundary,
                                          &MMA_Optimizer.Old_Design_Variables[0],
                                          &MMA_Optimizer.Old_Design_Variables[1],
                                          &Simp_Rho_Discrete_Max,
                                          &Simp_Rho_Discrete_Min};
  std::vector<VectorType *> grid_refine_vectors_out;

  parallel::distributed::SolutionTransfer<dimension,VectorType> solution_transfer(dof_handler_simp);

  triangulation.prepare_coarsening_and_refinement();    
  solution_transfer.prepare_for_coarsening_and_refinement(grid_refine_vectors_in);
  triangulation.execute_coarsening_and_refinement();

  setup_laplace_system();
  setup_multigrid_laplace_system();

  setup_filter_system();
  setup_multigrid_filter_system();

  setup_simp_system();
  get_cell_volume();

  Rmin = 1.5 * pow(Cell_Volume_Discrete.linfty_norm(), 1./(double)dimension) / (2. * sqrt(3.));

  MMA_Optimizer.Deal_II_MMA_Reinit(Simp_Rho_Discrete);
  
  pcout << "\n";
  pcout << "Number Of Refine Grid Cells : " << triangulation.n_global_active_cells() << std::endl;
  pcout << "Number Of Refine Grid Dofs : " << dof_handler.n_dofs() << std::endl;
  pcout << "\n";
  
  // 5个需要在细化网格后插值的向量
  grid_refine_vectors_out = {&Simp_Rho_Discrete,
                            &MMA_Optimizer.Moving_Low_Boundary,
                            &MMA_Optimizer.Moving_Upp_Boundary,
                            &MMA_Optimizer.Old_Design_Variables[0],
                            &MMA_Optimizer.Old_Design_Variables[1],
                            &Simp_Rho_Discrete_Max,
                            &Simp_Rho_Discrete_Min};
  // 
  solution_transfer.interpolate(grid_refine_vectors_out); 
}