#include "../../include/top_opt_heat_flow.h"

namespace TopOpt
{
    /*TopOptHeatFlow::运行
    这是该程序的最后一步。在这一部分中，我们分别生成网格并运行其他函数。最大细化可以通过参数设置。*/
    void TopOptHeatFlow::run(const unsigned int loop, const unsigned int refinement)
    {
        make_grid();

        topology_iteration(0.0001, loop, refinement, 1);
    }
}