"""Independent 70-digit OU/GBM moments; print the frozen campaign as JSON."""
from decimal import Decimal as D, localcontext
from math import comb, factorial
import json


def campaign():
    with localcontext() as context:
        context.prec = 70
        count, growth, volatility, duration, initial = D(4096), D(".25"), D(".5"), D(1), D(1)

        def joint(k, ell, steps):
            h = duration / D(steps)
            shifted = 1 + growth * h + D(ell) * volatility**2 * h
            gaussian = sum(
                D(comb(k, 2*j)) * shifted**(k-2*j) * (volatility**2*h)**j
                * D(factorial(2*j) // (2**j * factorial(j)))
                for j in range(k//2+1)
            )
            return initial**(k+ell) * (
                D(ell)*growth*duration + D(ell*ell-ell)*volatility**2*duration/2
            ).exp() * gaussian**steps

        def bounds(mean, variance, fourth):
            return {
                "mean": str(mean), "variance": str(variance),
                "mean_six_se": str(6*(variance/count).sqrt()),
                "variance_six_se": str(6*((fourth-(count-3)/(count-1)*variance**2)/count).sqrt()),
            }

        rows = []
        for depth in (3, 5, 7):
            steps = 2**depth
            h = duration / D(steps)
            r = 1 - 2*h
            mean = D(".5") + D(".5")*r**steps
            variance = D(".25")**2*h*(1-r**(2*steps))/(1-r*r)
            ou = bounds(mean, variance, 3*variance*variance)
            ou["continuous_mean"] = str(D(".5")+D(".5")*(-2*duration).exp())
            ou["continuous_variance"] = str(D(".25")**2*(1-(-4*duration).exp())/4)
            raw = [joint(k, 0, steps) for k in range(5)]
            variance = raw[2] - raw[1]**2
            fourth = raw[4]-4*raw[1]*raw[3]+6*raw[1]**2*raw[2]-3*raw[1]**4
            gbm = bounds(raw[1], variance, fourth)
            gbm["continuous_mean"] = str((growth*duration).exp())
            gbm["continuous_variance"] = str((2*growth*duration).exp()*((volatility**2*duration).exp()-1))
            mse = sum(D((-1)**(2-k)*comb(2,k))*joint(k,2-k,steps) for k in range(3))
            fourth_error = sum(D((-1)**(4-k)*comb(4,k))*joint(k,4-k,steps) for k in range(5))
            gbm["path_mse"] = str(mse)
            gbm["path_mse_six_se"] = str(6*((fourth_error-mse*mse)/count).sqrt())
            rows.append({"depth": depth, "steps": steps, "ou": ou, "gbm": gbm})
        return {
            "sample_count": 4096, "master_seed_byte": 83,
            "sample_ids": "0..4096 exclusive, u32 little endian",
            "weak_channel": "sde-weak", "strong_channel": "sde-strong", "duration_s": 1,
            "ou": {"initial": 1, "mean": ".5", "reversion_rate": 2, "volatility": ".25"},
            "gbm": {"initial": 1, "growth_rate": ".25", "volatility": ".5"},
            "criterion": "absolute error from exact finite-grid moment <= six analytically derived standard errors; weak continuum error separately bounded by exact finite-grid bias plus sampling bound",
            "rows": rows,
        }


if __name__ == "__main__":
    print(json.dumps(campaign(), indent=2))
