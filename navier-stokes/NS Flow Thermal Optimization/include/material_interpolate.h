#ifndef MATERIAL_INTERPOLATE_H
#define MATERIAL_INTERPOLATE_H

#include <cmath>

namespace TopOpt
{
    class MaterialInterpolate
    {
    public:
        MaterialInterpolate(double af, double as, double penal_a,
                            double kf, double ks, double penal_k,
                            double hs, double penal_h,
                            double th_f, double penal_th);
        /*插值公式-渗透率-正向插值*/
        double permeability_interpolate(double xphys);

        /*插值公式-热导率-正向插值*/
        double conductivity_interpolate(double xphys);

        /*插值公式-热源系数-正向插值*/
        double heatsource_interpolate(double xphys);

        /*插值公式-热对流系数-正向插值*/
        double convection_interpolate(double xphys);

        /*插值公式-渗透率-反向求导*/
        double permeability_partial(double xphys);

        /*插值公式-热导率-反向求导*/
        double conductivity_partial(double xphys);

        /*插值公式-热源系数-反向求导*/
        double heatsource_partial(double xphys);

        /*插值公式-热对流系数-反向求导*/
        double convection_partial(double xphys);

    private:
        double af;
        double as;
        double kf;
        double ks;
        double hs;
        double th_f;
        double penal_a;
        double penal_k;
        double penal_h;
        double penal_th;
    };
}

#endif