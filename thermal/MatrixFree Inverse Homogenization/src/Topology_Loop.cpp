#include "../include/Heat_Matrix_Free.h"

using namespace Heat_Matrix_Free;

/**
 * @brief Heat_Solver 类的构造函数。
 *
 * 该构造函数初始化 Heat_Solver 对象，具体包括：
 * - 设置 MPI 通信器。
 * - 使用 MPI 通信器、顶点处的层次差异限制和多重网格层次结构构建初始化三角剖分。
 * - 使用指定的度初始化有限元对象。
 * - 初始化主、简化和过滤三角剖分的自由度处理程序。
 * - 为根 MPI 进程设置并行输出流。
 * - 使用 MPI 通信器、输出流和计时选项初始化计算计时器。
 */
Heat_Solver::Heat_Solver()
    : mpi_communicator(MPI_COMM_WORLD),
      triangulation(mpi_communicator,
                    Triangulation<dimension>::limit_level_difference_at_vertices,
                    parallel::distributed::Triangulation<dimension>::construct_multigrid_hierarchy),
      mapping(FE_Q<dimension>(degree_finite_element)),
      mapping_filter(FE_Q<dimension>(filter_degree_finite_element)),
      fe(degree_finite_element),
      fe_simp(0),
      fe_filter(filter_degree_finite_element),
      dof_handler(triangulation),
      dof_handler_simp(triangulation),
      dof_handler_filter(triangulation),
      pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0),
      computing_timer(mpi_communicator,
                      pcout,
                      TimerOutput::never,
                      TimerOutput::wall_times)
{
    for (unsigned int i = 0; i < dimension; i++)
    {
        Material_Thermal_Conductivity_Matrix[i][i] = make_vectorized_array<double>(10.);
        LeveLMaterial_Thermal_Conductivity_Matrix[i][i] = make_vectorized_array<float>(10.);
        SumMaterial_Thermal_Conductivity_Matrix[i][i] = 10.;
    }
    Obj_Thermal_Conductivity_Matrix[0][0] = 3.5;
    Obj_Thermal_Conductivity_Matrix[1][1] = 3.5;
}

/**
 * @brief 为热求解器生成网格并设置周期性边界条件。
 *
 * 该函数在每个维度的范围 [0, 1] 内创建一个超立方体网格。
 * 然后，它遍历网格的活动面，并根据面中心的位置分配边界 ID。对于 3D 网格，还会为 z 维度的面分配额外的边界 ID。
 *
 * 在设置边界 ID 之后，该函数收集每个维度的周期性面对，并将它们添加到三角剖分中。最后，网格被全局细化六次。
 *
 * @note 该函数使用 TimerOutput::Scope 来测量网格生成所需的时间。
 */
void Heat_Solver::make_grid()
{
    TimerOutput::Scope t(computing_timer, "make_grid");
    GridGenerator::hyper_cube(triangulation, -1, 1);
    for (auto &face : triangulation.active_face_iterators())
    {
        if (face->center()(0) == -1)
        {
            face->set_boundary_id(1);
        }
        if (face->center()(0) == 1)
        {
            face->set_boundary_id(2);
        }
        if (face->center()(1) == -1)
        {
            face->set_boundary_id(3);
        }
        if (face->center()(1) == 1)
        {
            face->set_boundary_id(4);
        }
        if (dimension == 3)
        {
            if (face->center()(2) == -1)
            {
                face->set_boundary_id(5);
            }
            if (face->center()(2) == 1)
            {
                face->set_boundary_id(6);
            }
        }
    }
    Tensor<1, dimension> offset;
    std::vector<GridTools::PeriodicFacePair<typename Triangulation<dimension>::cell_iterator>> periodicity_vector;

    GridTools::collect_periodic_faces(triangulation,
                                      1,
                                      2,
                                      0,
                                      periodicity_vector,
                                      offset);
    GridTools::collect_periodic_faces(triangulation,
                                      3,
                                      4,
                                      1,
                                      periodicity_vector,
                                      offset);
    if (dimension == 3)
    {
        GridTools::collect_periodic_faces(triangulation,
                                          5,
                                          6,
                                          2,
                                          periodicity_vector,
                                          offset);
    }
    triangulation.add_periodicity(periodicity_vector);
    triangulation.refine_global(7);
}

/**
 * @brief 初始化热求解器中使用的简化向量结构。
 *
 * 该函数设置简化有限元（FE）的自由度（DoFs），并初始化简化密度（Simp_Rho_Discrete）、
 * 单元体积（Cell_Volume_Discrete）和过滤后的简化密度（Simp_Rho_Discrete_Filter）的离散向量。
 *
 * 它使用 TimerOutput::Scope 来测量初始化过程所需的时间。
 *
 * @note 该函数假设有限元（fe_simp）和 MPI 通信器（mpi_communicator）已经设置好。
 */
void Heat_Solver::setup_simp_system()
{
    TimerOutput::Scope t(computing_timer, "setup_simp_system");
    dof_handler_simp.distribute_dofs(fe_simp);

    Simp_Rho_Discrete.reinit(dof_handler_simp.locally_owned_dofs(), mpi_communicator);
    Simp_Rho_Discrete_Filter.reinit(dof_handler_simp.locally_owned_dofs(), mpi_communicator);
    Simp_Rho_Discrete_Max.reinit(dof_handler_simp.locally_owned_dofs(), mpi_communicator);
    Simp_Rho_Discrete_Min.reinit(dof_handler_simp.locally_owned_dofs(), mpi_communicator);
    Simp_Laplace.reinit(dof_handler_simp.locally_owned_dofs(), mpi_communicator);

    Cell_Volume_Discrete.reinit(dof_handler_simp.locally_owned_dofs(), mpi_communicator);
    Object_Discrete_Derivative.reinit(dof_handler_simp.locally_owned_dofs(), mpi_communicator);

    Constraint_Discrete_Derivative.resize(1);
    Constraint_Discrete_Derivative[0].reinit(dof_handler_simp.locally_owned_dofs(), mpi_communicator);
}

/**
 * @brief 为热求解器设置简化向量的值。
 *
 * 该函数根据域中单元的位置初始化 `Simp_Rho_Discrete` 向量。
 * 对于每个本地拥有的单元，如果单元的中心在距离原点0.4的半径内，则分配值0.5，否则分配值1.0。
 *
 * 该函数使用计时器来测量设置过程的执行时间。
 */
void Heat_Solver::setup_init_simp_value()
{
    TimerOutput::Scope t(computing_timer, "setup_init_simp_value");
    // double iner_rad = 0.3;
    // double outer_rad = 0.7;
    //
    std::vector<types::global_dof_index> local_rho_dof_indices(fe_simp.n_dofs_per_cell());
    for (auto &cell_iter : dof_handler_simp.active_cell_iterators())
    {
        if (cell_iter->is_locally_owned())
        {
            cell_iter->get_dof_indices(local_rho_dof_indices);
            // 设定边界形状
            // if (cell_iter->at_boundary())
            // {
            //     // X方向
            //     if (pow(cell_iter->center()(1), 2.) + pow(cell_iter->center()(2) ,2.) > iner_rad * iner_rad &&
            //         pow(cell_iter->center()(1), 2.) + pow(cell_iter->center()(2) ,2.) < outer_rad * outer_rad)
            //     {
            //         Simp_Rho_Discrete[local_rho_dof_indices[0]] = 1.;
            //     }
            //     // Y方向
            //     else if (pow(cell_iter->center()(0), 2.) + pow(cell_iter->center()(2) ,2.) > iner_rad * iner_rad &&
            //              pow(cell_iter->center()(0), 2.) + pow(cell_iter->center()(2) ,2.) < outer_rad * outer_rad)
            //     {
            //         Simp_Rho_Discrete[local_rho_dof_indices[0]] = 1.;
            //     }
            //     // Z方向
            //     else if (pow(cell_iter->center()(0), 2.) + pow(cell_iter->center()(1) ,2.) > iner_rad * iner_rad &&
            //              pow(cell_iter->center()(0), 2.) + pow(cell_iter->center()(1) ,2.) < outer_rad * outer_rad)
            //     {
            //         Simp_Rho_Discrete[local_rho_dof_indices[0]] = 1.;
            //     } 
            //     else
            //     {
            //         Simp_Rho_Discrete[local_rho_dof_indices[0]] = 1.e-3;
            //     }
            // }
            // else
            // {
            //     // X方向
            //     if (pow(cell_iter->center()(1), 2.) + pow(cell_iter->center()(2) ,2.) <= iner_rad * iner_rad)
            //     {
            //         Simp_Rho_Discrete[local_rho_dof_indices[0]] = 0.;
            //     }
            //     // Y方向
            //     else if (pow(cell_iter->center()(0), 2.) + pow(cell_iter->center()(2) ,2.) <= iner_rad * iner_rad)
            //     {
            //         Simp_Rho_Discrete[local_rho_dof_indices[0]] = 0.;
            //     }
            //     // Z方向
            //     else if (pow(cell_iter->center()(0), 2.) + pow(cell_iter->center()(1) ,2.) <= iner_rad * iner_rad)
            //     {
            //         Simp_Rho_Discrete[local_rho_dof_indices[0]] = 0.;
            //     } 
            //     else
            //     {
            //         Simp_Rho_Discrete[local_rho_dof_indices[0]] = 1.;
            //     }
            // }

            // 多重网格计算和自适应网格
            if (pow(cell_iter->center()(0), 2.) + pow(cell_iter->center()(1), 2.) < 0.2 * 0.2)
            {
                Simp_Rho_Discrete[local_rho_dof_indices[0]] = 0.9;
            }
            else
            {
                Simp_Rho_Discrete[local_rho_dof_indices[0]] = 1.;
            }
        }
    }
    Simp_Rho_Discrete.compress(VectorOperation::insert);
    Simp_Rho_Discrete.update_ghost_values();
}

/**
 * @brief 计算网格中每个单元的体积，并将其存储在 Cell_Volume_Discrete 向量中。
 *
 * 该函数遍历 dof_handler_simp 对象中的所有活动单元。对于每个本地拥有的单元，
 * 它检索自由度（DOF）索引，并将单元的测度（体积）添加到 Cell_Volume_Discrete 向量中的相应条目。
 * 在处理完所有单元后，向量被压缩以合并来自不同进程的贡献（如果在并行运行），并更新幽灵值。
 * 最后，该函数计算过滤半径（Rmin），其值为 Cell_Volume_Discrete 向量的无穷范数的平方根。
 *
 * @note 该函数使用 TimerOutput::Scope 对象来测量执行所需的时间。
 */
void Heat_Solver::get_cell_volume()
{
    TimerOutput::Scope t(computing_timer, "get_cell_volume");
    Cell_Volume_Discrete = 0;
    //
    std::vector<types::global_dof_index> local_rho_dof_indices(fe_simp.n_dofs_per_cell());
    for (auto &cell_iter : dof_handler_simp.active_cell_iterators())
    {
        if (cell_iter->is_locally_owned())
        {
            cell_iter->get_dof_indices(local_rho_dof_indices);
            Cell_Volume_Discrete[local_rho_dof_indices[0]] += cell_iter->measure();
        }
    }
    Cell_Volume_Discrete.compress(VectorOperation::add);
    Cell_Volume_Discrete.update_ghost_values();
}

/**
 * @brief 将热求解器的结果输出到 VTU 文件。
 *
 * 该函数处理与热求解器相关的各种数据向量的输出。
 * 它将自由度（DoF）处理程序附加到 DataOut 对象，并添加多个数据向量
 * 用于连续和离散变量以及过滤相关的数据。然后将结果写入
 * 带有适当压缩设置的 VTU 文件。
 *
 * 输出以下数据向量：
 * - Simp_Rho_Continuous: 连续简化密度。
 * - unit_test_rhs: 单元测试的右手边向量（每个维度一个）。
 * - unit_test_temperature: 单元测试的温度向量（每个维度一个）。
 * - Simp_Rho_Discrete: 离散简化密度。
 * - Cell_Volume_Discrete: 离散单元体积。
 * - filter_rhs: 过滤器的右手边向量。
 * - filter_solution: 过滤器的解向量。
 *
 * 该函数还设置输出文件的压缩级别以实现最佳速度。
 *
 * @note 该函数使用 TimerOutput::Scope 来测量输出过程所需的时间。
 */
void Heat_Solver::output_results()
{
    TimerOutput::Scope t(computing_timer, "output_results");
    // Simp Output
    DataOut<dimension> data_out;
    data_out.attach_dof_handler(dof_handler_simp);
    data_out.add_data_vector(dof_handler_simp, Simp_Rho_Discrete, "Simp_Rho_Discrete");
    data_out.add_data_vector(dof_handler_simp, Simp_Rho_Discrete_Filter, "Simp_Rho_Discrete_Filter");
    data_out.add_data_vector(dof_handler_simp, Cell_Volume_Discrete, "Cell_Volume_Discrete");
    data_out.add_data_vector(dof_handler_simp, Object_Discrete_Derivative, "Object_Discrete_Derivative");
    data_out.add_data_vector(dof_handler_simp, Constraint_Discrete_Derivative[0], "Constraint_Discrete_Derivative");
    data_out.add_data_vector(dof_handler_simp, Simp_Laplace, "Simp_Laplace");
    data_out.build_patches();
    DataOutBase::VtkFlags flags;
    flags.compression_level = DataOutBase::CompressionLevel::best_speed;
    data_out.set_flags(flags);
    data_out.write_vtu_with_pvtu_record("./Simp_Out/", "simp", MMA_Optimizer.Loop_Iter, mpi_communicator);

    // // Solution Output
    // DataOut<dimension> data_out_solution;
    // data_out_solution.attach_dof_handler(dof_handler);
    // for (unsigned int index = 0; index < dimension; index++)
    // {
    //     data_out_solution.add_data_vector(unit_test_temperature[index], "unit_test_temperature_" + std::to_string(index));
    //     data_out_solution.add_data_vector(unit_test_rhs[index], "unit_test_rhs_" + std::to_string(index));
    // }
    // data_out_solution.build_patches(mapping);
    // data_out_solution.set_flags(flags);
    // data_out_solution.write_vtu_with_pvtu_record("./Solution_Out/", "solution", MMA_Optimizer.Loop_Iter, mpi_communicator);

    // Filter Output
    // DataOut<dimension> data_out_filter;
    // data_out_filter.attach_dof_handler(dof_handler_filter);
    // data_out_filter.add_data_vector(Simp_Rho_Continuous_Filter, "Simp_Rho_Continuous_Filter");
    // data_out_filter.build_patches(mapping);
    // data_out_filter.set_flags(flags);
    // data_out_filter.write_vtu_with_pvtu_record("./Filter_Out/", "filter", MMA_Optimizer.Loop_Iter, mpi_communicator);
}

void Heat_Solver::output_simp()
{
    TimerOutput::Scope t(computing_timer, "output_simp");
    std::ofstream dataFile("dataFile.txt", std::ios::app);

    std::vector<types::global_dof_index> local_rho_dof_indices(fe_simp.n_dofs_per_cell());
    for (auto &cell_iter : dof_handler_simp.active_cell_iterators())
    {
        if (cell_iter->is_locally_owned())
        {
            cell_iter->get_dof_indices(local_rho_dof_indices);
            dataFile << std::fixed<<std::setprecision(10)
                     << cell_iter->center()(0) << " " 
                     << cell_iter->center()(1) << " "
                     << Simp_Rho_Discrete[local_rho_dof_indices[0]] << std::endl;
        }
    }
    MPI_Barrier(mpi_communicator);
    dataFile.close();
}

double Heat_Solver::mma_optimizer(Vector<double> input)
{
    TimerOutput::Scope t(computing_timer, "mma_optimizer");
    MMA_Optimizer.Deal_II_MMA_Dual_Problem_Solve(Simp_Rho_Discrete,
                                                 Object_Discrete_Derivative,
                                                 input,
                                                 Constraint_Discrete_Derivative,
                                                 Simp_Rho_Discrete_Max,
                                                 Simp_Rho_Discrete_Min);
    return MMA_Optimizer.Deal_II_MMA_Get_Change(Simp_Rho_Discrete);
}

void Heat_Solver::keep_boundary_shape(LinearAlgebra::distributed::Vector<double> &input)
{
    TimerOutput::Scope t(computing_timer, "keep_boundary_shape");
    //
    std::vector<types::global_dof_index> local_rho_dof_indices(fe_simp.n_dofs_per_cell());
    for (auto &cell_iter : dof_handler_simp.active_cell_iterators())
    {
        if (cell_iter->is_locally_owned() && cell_iter->at_boundary())
        {
            cell_iter->get_dof_indices(local_rho_dof_indices);
            input[local_rho_dof_indices[0]] = 0.;
        }
    }
    input.compress(VectorOperation::insert);
    input.update_ghost_values();
}

/**
 * @brief 执行热求解器的主要操作序列。
 *
 * 该函数执行以下步骤：
 * - 打印向量化信息。
 * - 初始化计算网格。
 * - 设置拉普拉斯系统及其多重网格变体。
 * - 设置过滤系统及其多重网格变体。
 * - 初始化并处理 SIMP（固体各向同性材料惩罚）向量。
 * - 组装拉普拉斯方程的右手边并求解。
 * - 生成均匀化张量。
 * - 组装过滤方程的右手边并求解。
 * - 生成离散 SIMP 密度的平均向量。
 * - 输出结果。
 * - 打印优化的热导率矩阵。
 * - 记录执行时间。
 */
void Heat_Solver::run()
{
    const unsigned int n_vect_doubles = VectorizedArray<double>::size();
    const unsigned int n_vect_bits = 8 * sizeof(double) * n_vect_doubles;

    pcout << "Vectorization over " << n_vect_doubles
          << " doubles = " << n_vect_bits << " bits ("
          << Utilities::System::get_current_vectorization_level() << ')'
          << std::endl;
    //
    Vector<double> Object_Function(1);
    Vector<double> Constraint_Function(1);
    double change = 1e10;
    double volfrac = 0.5;
    double move = 0.2;
    neta = 1e-3;
    double value_const = 0;
    // 初始化网格
    // pcout << "+---------------------------------------------------------------------------+" << std::endl;
    // pcout << "|                             INIT OPTIMIZATION                             |" << std::endl;
    // pcout << "+---------------------------------------------------------------------------+" << std::endl;
    // pcout << std::endl;
    make_grid();

    setup_simp_system();
    get_cell_volume();

    // 计算过滤半径
    Rmin = 2 * pow(Cell_Volume_Discrete.linfty_norm(), 1. / (double)dimension) / (2. * sqrt(3.));

    MMA_Optimizer.Deal_II_MMA_Init(1, 0, 1e3, Simp_Rho_Discrete);
    MMA_Optimizer.Loop_Iter = 0;

    setup_laplace_system();
    setup_multigrid_laplace_system();

    setup_filter_system();
    setup_multigrid_filter_system();

    assemble_filter_rhs(Cell_Volume_Discrete);
    solve_filter_chebyshev();
    generate_average_vector(Constraint_Discrete_Derivative[0]);
    value_const = Constraint_Discrete_Derivative[0].linfty_norm();
    Constraint_Discrete_Derivative[0] /= Constraint_Discrete_Derivative[0].linfty_norm();
    // keep_boundary_shape(Constraint_Discrete_Derivative[0]);
    //

    // pcout << std::endl;
    // pcout << "+---------------------------------------------------------------------------+" << std::endl;
    // pcout << "|                             DEGREE OF FREEDOM                             |" << std::endl;
    // pcout << "+---------------------------------------------------------------------------+" << std::endl;
    // pcout << std::endl;

    // pcout << "+---------------------------------------------------------------------------+" << std::endl;
    // pcout << "                             " << "Simp    : " << triangulation.n_global_active_cells() << std::endl;
    // pcout << "                             " << "Laplace : " << dof_handler.n_dofs() << std::endl;
    // pcout << "                             " << "Filter  : " << dof_handler_filter.n_dofs() << std::endl;
    // pcout << "+---------------------------------------------------------------------------+" << std::endl;
    // pcout << std::endl;

    // pcout << "+---------------------------------------------------------------------------+" << std::endl;
    // pcout << "|                            START OPTIMIZATION                             |" << std::endl;
    // pcout << "+---------------------------------------------------------------------------+" << std::endl;
    // MMA优化循环
    while (MMA_Optimizer.Loop_Iter < 50 )//&& change > 1e-4 * dof_handler_simp.n_dofs())
    {
        MMA_Optimizer.Loop_Iter++;

        if (MMA_Optimizer.Loop_Iter == 1)
        {
            setup_init_simp_value();
        }

        if (MMA_Optimizer.Loop_Iter > 3)
        {
            neta = 0.96 * neta;
        }

        for (unsigned int i : dof_handler_simp.locally_owned_dofs())
        {
            Simp_Rho_Discrete_Max[i] = std::min(1., Simp_Rho_Discrete[i] + move);
            Simp_Rho_Discrete_Min[i] = std::max(1.e-3, Simp_Rho_Discrete[i] - move);
        }

        // pcout << std::endl;
        // pcout << "+---------------------------------------------------------------------------+" << std::endl;
        // pcout << "|                             SOLVER INFOMATION                             |" << std::endl;
        // pcout << "+---------------------------------------------------------------------------+" << std::endl;
        // pcout << std::endl;

        // pcout << "+---------------------------------------------------------------------------+" << std::endl;
        assemble_filter_rhs(Simp_Rho_Discrete);
        solve_filter_chebyshev();
        Simp_Rho_Continuous_Filter = filter_solution;
        generate_average_vector(Simp_Rho_Discrete_Filter);
        generate_solution_laplace(Simp_Laplace);

        interpolate_simp_vector();
        assemble_laplace_rhs();

        solve_chebyshev();
        generate_homogenization_tensor();

        generate_object_discrete_derivative();
        // 目标敏度过滤
        // Object_Discrete_Derivative.operator-=(Simp_Laplace);

        assemble_filter_rhs(Object_Discrete_Derivative);
        solve_filter_chebyshev();
        generate_average_vector(Object_Discrete_Derivative);
        Object_Discrete_Derivative /= Object_Discrete_Derivative.linfty_norm();
        // keep_boundary_shape(Object_Discrete_Derivative);

        // pcout << "+---------------------------------------------------------------------------+" << std::endl;
        // pcout << std::endl;

        // 计算体积约束
        Constraint_Function[0] = Simp_Rho_Discrete * Cell_Volume_Discrete - volfrac * Cell_Volume_Discrete.l1_norm();
        Constraint_Function[0] /= value_const;

        // 更新设计变量
        change = mma_optimizer(Constraint_Function);

        // pcout << "+---------------------------------------------------------------------------+" << std::endl;
        // pcout << "|                              OUTPUT  RESULTS                              |" << std::endl;
        // pcout << "+---------------------------------------------------------------------------+" << std::endl;
        Object_Function[0] = 0;
        for (unsigned int i = 0; i < dimension; i++)
        {
            for (unsigned int j = 0; j < dimension; j++)
            {
                Object_Function[0] += pow(Opt_Thermal_Conductivity_Matrix[i][j] - Obj_Thermal_Conductivity_Matrix[i][j], 2.);
                // pcout << std::setprecision(4) << std::fixed << std::scientific
                //       << "                             "
                //       << "Opt[" << Utilities::to_string(i, 1) << "][" << Utilities::to_string(j, 1) << "] = "
                //       << Opt_Thermal_Conductivity_Matrix[i][j]
                //       << std::endl
                //       << "                             "
                //       << std::setprecision(4) << std::fixed << std::scientific
                //       << "Obj[" << Utilities::to_string(i, 1) << "][" << Utilities::to_string(j, 1) << "] = "
                //       << Obj_Thermal_Conductivity_Matrix[i][j]
                //       << std::endl;
                // pcout << std::endl;
            }
        }
        // pcout << std::endl;
        // pcout << "                             " << "Iter   : " << MMA_Optimizer.Loop_Iter << "                            " << std::endl
        //       << "                             " << "Object : " << Object_Function[0] << "                             " << std::endl
        //       << "                             " << "Change : " << change << "                             " << std::endl
        //       << "                             " << "Volume : " << (Simp_Rho_Discrete * Cell_Volume_Discrete) / Cell_Volume_Discrete.l1_norm() << "                             " << std::endl;
        // pcout << "+---------------------------------------------------------------------------+" << std::endl;
        pcout << "Iter : " << MMA_Optimizer.Loop_Iter 
              << " Object : " << Object_Function[0] 
              << " Change : " << change
              << " Volume : " << (Simp_Rho_Discrete * Cell_Volume_Discrete) / Cell_Volume_Discrete.l1_norm()
              << std::endl;
        // if (MMA_Optimizer.Loop_Iter % 30 == 0)
        // {
        //     refine_grid();
        //     assemble_filter_rhs(Cell_Volume_Discrete);
        //     solve_filter_chebyshev();
        //     generate_average_vector(Constraint_Discrete_Derivative[0]);
        // }
        output_results();
    }
    // output_simp();
    time();
}

void Heat_Solver::time()
{
    computing_timer.print_summary();
    computing_timer.reset();
}