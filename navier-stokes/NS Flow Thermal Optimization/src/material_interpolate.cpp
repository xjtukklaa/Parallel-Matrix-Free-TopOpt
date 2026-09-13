#include "../include/material_interpolate.h"

namespace TopOpt
{
    MaterialInterpolate::MaterialInterpolate(double af, double as, double penal_a,
                                             double kf, double ks, double penal_k,
                                             double hs, double penal_h,
                                             double th_f, double penal_th)
    {
        this->af = af;
        this->as = as;
        this->penal_a = penal_a;
        this->penal_k = penal_k;
        this->kf = kf;
        this->ks = ks;
        this->hs = hs;
        this->penal_h = penal_h;
        this->th_f = th_f;
        this->penal_th = penal_th;
    }
    /*插值公式-渗透率-RAMP*/
    double MaterialInterpolate::permeability_interpolate(double xphys)
    {
        return af + (as - af) * (penal_a * (1.0 - xphys)) / (penal_a + xphys);
    }



















    /*插值公式-热导率-RAMP*/
    double MaterialInterpolate::conductivity_interpolate(double xphys)
    {
        return kf + (ks - kf) * ((penal_k * (1.0 - xphys)) / (penal_k + xphys));
    }

    /*插值公式-热源系数-RAMP*/
    double MaterialInterpolate::heatsource_interpolate(double xphys)
    {
        return 0.0 + (hs - 0.0) * ((penal_h * (1.0 - xphys)) / (penal_h + xphys));
    }

    /*插值公式-热对流系数-RAMP*/
    double MaterialInterpolate::convection_interpolate(double xphys)
    {
        return th_f + (0.0 - th_f) * ((penal_th * (1.0 - xphys)) / (penal_th + xphys));
    }

    /*插值公式-渗透率-RAMP*/
    double MaterialInterpolate::permeability_partial(double xphys)
    {


        return (as - af) * (-penal_a * (penal_a + xphys) - penal_a * (1.0 - xphys)) /
               ((penal_a + xphys) * (penal_a + xphys));
    }



















    /*插值公式-热导率-RAMP*/
    double MaterialInterpolate::conductivity_partial(double xphys)
    {
        return (ks - kf) * (-penal_k * (penal_k + xphys) - penal_k * (1.0 - xphys)) /
               ((penal_k + xphys) * (penal_k + xphys));
    }

    /*插值公式-热源系数-RAMP*/
    double MaterialInterpolate::heatsource_partial(double xphys)
    {
        return (hs - 0.0) * (-penal_h * (penal_h + xphys) - penal_h * (1.0 - xphys)) /
               ((penal_h + xphys) * (penal_h + xphys));
    }

    /*插值公式-热对流系数-RAMP*/
    double MaterialInterpolate::convection_partial(double xphys)
    {
        return (0.0 - th_f) * (-penal_th * (penal_th + xphys) - penal_th * (1.0 - xphys)) /
               ((penal_th + xphys) * (penal_th + xphys));
    }

}