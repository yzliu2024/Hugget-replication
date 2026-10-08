import numpy as np
from pathlib import Path

beta = 0.99322
sigma = 1.5
da = 0.1
e_grid = np.array([1.0,0.1])
P_trans_matrix = np.array([
    [0.925,0.5],
    #ij表示j转换到i的概率
    [0.075,0.5]
])
theta = np.array([0.0,0.5,1.0])


def c(a, e, a_prime, q):
    c = a + e - a_prime * q
    return c

def c(a, e, a_prime, q, theta, b):
    c = a + e - a_prime * q
    return c

def Du(c):
    return c ** (-sigma)

def bisection(left, right, ideal_numb, func, result_tol, section_tol, diff_left = None, diff_right = None):

    if diff_left is None:
        diff_left = func(left)
    if diff_right is None:
        diff_right = func(right)


    if np.isnan(diff_left) or np.isnan(diff_right):
        raise ValueError(
            f"端点函数值出现NaN："
            f"{left=}, {diff_left=}, "
            f"{right=}, {diff_right=}"
        )

    error_left = diff_left - ideal_numb
    error_right = diff_right - ideal_numb

    if error_left == 0:
        return left

    if error_right == 0:
        return right

    # 只要求目标值被左右端点夹住，不限制函数递增或递减
    if error_left * error_right > 0:
        raise ValueError(
            f"没有形成有效二分区间："
            f"{left=}, {right=}, "
            f"{diff_left=}, {diff_right=}, "
            f"{ideal_numb=}"
        )

    for _ in range(1000):
        mid = (left + right) / 2
        diff_mid = func(mid)

        if np.isnan(diff_mid):
            raise ValueError(
                f"中点函数值出现NaN：{mid=}"
            )

        error_mid = diff_mid - ideal_numb

        if abs(error_mid) < result_tol:
            return mid

        if abs(right - left) < section_tol:
            candidates = [
                (left, diff_left),
                (mid, diff_mid),
                (right, diff_right)
            ]

            return min(
                candidates,
                key=lambda x: abs(
                    x[1] - ideal_numb
                )
            )[0]

        # 如果目标值位于[left, mid]之间，保留左半边
        if error_left * error_mid < 0:
            right = mid
            diff_right = diff_mid
            error_right = error_mid

        # 否则目标值位于[mid, right]之间
        else:
            left = mid
            diff_left = diff_mid
            error_left = error_mid
    raise ValueError(f"在{func=},"
                     f"{left=}, {diff_left=}, {right=}, {diff_right=}"
                     f"的区间未能用二分法求得合法解。")

def policy_grid_gen(DV_grid, q, a_grid):

    Edv_grid = DV_grid @ P_trans_matrix

    policy_grid = np.zeros_like(
        DV_grid,
        dtype=float
    )

    for ia, a in enumerate(a_grid):
        for ie, e in enumerate(e_grid):

            def diff_func(a_given):
                if a_given < a_grid[0]:
                    raise ValueError(
                        f"生成policy grid时，通过二分法寻找a_policy时，左界或者右界低于网格下界。"
                        f"下一期a值:{a_given=},当期状态:{ie=}"
                    )

                if a_given > a_grid[-1]:
                    raise ValueError(
                        f"生成policy grid时，通过二分法寻找a_policy时，左界或者右界超出网格上界。"
                        f"下一期a值:{a_given=},当期状态:{ie=}"
                    )

                du = Du(
                    c(a, e, a_given, q)
                )

                idx = int(np.floor((a_given - a_grid[0]) / da)) + 1
                #a为a_grid[0]的时候，返回0，最后的值时，返回N 相当于right
                if idx == len(a_grid):
                    Edv = Edv_grid[-1, ie]
                else:
                    lamb = (a_given - a_grid[idx - 1])/ da
                    Edv = lamb * Edv_grid[idx, ie] + (1-lamb) * Edv_grid[idx - 1, ie]

                diff = du * q - beta * Edv

                return diff

            c_floor = 1e-10
            if ia == 0:
                feasible_left = a_grid[0]
            else:
                feasible_left = policy_grid[ia-1,ie]
            feasible_right = min(
                a_grid[-1],
                (a + e - c_floor) / q
            )

            diff_left = diff_func(feasible_left)

            if diff_left >= 0:
                policy_grid[ia, ie] = feasible_left
                continue

            diff_right = diff_func(feasible_right)

            if diff_right < 0:
                raise ValueError(
                    f"最优a_prime超过网格上界："
                    f"{q=}, {ia=}, {a=}, "
                    f"{ie=}, {e=}, "
                    f"{feasible_right=}, {diff_right=}"
                )

            a_policy = bisection(
                feasible_left,
                feasible_right,
                0,
                diff_func,
                1e-6,
                1e-7,
                diff_left,
                diff_right
            )

            policy_grid[ia, ie] = a_policy
            continue
    return policy_grid

def Bellman(DV0_grid, q, a_grid):
    DVold_grid = DV0_grid.copy()

    DVnew_grid = np.zeros_like(
        DVold_grid,
        dtype=float
    )

    for iteration in range(100000):
        policy_grid = policy_grid_gen(
            DVold_grid,
            q,
            a_grid
        )

        for ia, a in enumerate(a_grid):
            for ie, e in enumerate(e_grid):
                DVnew_grid[ia, ie] = Du(
                    c(a, e, policy_grid[ia, ie], q)
                )

        rel_error_DV = np.max(
            np.abs(DVnew_grid - DVold_grid)
            / np.maximum(
                1.0,
                np.abs(DVold_grid)
            )
        )
        #print(iteration, rel_error_DV)

        if rel_error_DV < 5 * 1e-6:
            DVstar_grid = DVnew_grid.copy()

            policy_star_grid = policy_grid_gen(
                DVstar_grid,
                q,
                a_grid
            )

            return DVstar_grid, policy_star_grid

        DVold_grid = DVnew_grid.copy()

    raise RuntimeError(
        f"Bellman迭代在{100000}次后仍未收敛，"
        f"最后相对误差为{rel_error_DV}"
    )


def inverse_policy_grid_gen(policy_grid, a_grid):
    inverse_policy_grid = np.zeros_like(
        policy_grid,
        dtype=float
    )

    for ia_target, a_target in enumerate(a_grid):
        for ie, e in enumerate(e_grid):
            idx = np.searchsorted(
                policy_grid[:, ie],
                a_target,
                side='right'
            )

            if idx == 0:
                inverse_policy_grid[
                    ia_target,
                    ie
                ] = np.nan

                continue

            if idx == len(policy_grid):
                inverse_policy_grid[
                    ia_target,
                    ie
                ] = a_grid[-1]

                continue

            lamb = (
                abs(
                    a_target
                    - policy_grid[idx - 1, ie]
                )
                / abs(
                    policy_grid[idx, ie]
                    - policy_grid[idx - 1, ie]
                )
            )

            inverse_policy_grid[
                ia_target,
                ie
            ] = (
                a_grid[idx - 1]
                + lamb * da
            )

    return inverse_policy_grid


def distribution_iteration(policy_grid, a_grid, Fguess_grid):
    F_old = Fguess_grid.copy()

    inverse_policy_grid = inverse_policy_grid_gen(
        policy_grid,
        a_grid
    )

    for _ in range(100000):
        F_new = np.zeros_like(
            F_old,
            dtype=float
        )

        for ia, a in enumerate(a_grid):
            for ie, e in enumerate(e_grid):
                for ie_F, e_F in enumerate(e_grid):
                    if np.isnan(
                        inverse_policy_grid[ia, ie_F]
                    ):
                        F_old_inter = 0.0

                    else:
                        idx = np.searchsorted(
                            a_grid,
                            inverse_policy_grid[
                                ia,
                                ie_F
                            ],
                            side='right'
                        )

                        if idx == 0:
                            raise ValueError(
                                "分布迭代的inversepolicy中，"
                                "某个节点的前推政策值低于"
                                "agrid最低值。"
                                f"该节点为{ia=}, {a=}, "
                                f"{ie=}, {ie_F=}, {e_F=}"
                            )

                        elif idx == len(a_grid):
                            F_old_inter = F_old[
                                -1,
                                ie_F
                            ]

                        else:
                            lamb = (
                                abs(
                                    inverse_policy_grid[
                                        ia,
                                        ie_F
                                    ]
                                    - a_grid[idx - 1]
                                )
                                / abs(
                                    a_grid[idx]
                                    - a_grid[idx - 1]
                                )
                            )

                            F_old_inter = (
                                (1 - lamb)
                                * F_old[idx - 1, ie_F]
                                + lamb
                                * F_old[idx, ie_F]
                            )

                    F_new[ia, ie] += (
                        P_trans_matrix[ie, ie_F]
                        * F_old_inter
                    )

        total_probability = np.sum(
            F_new[-1, :]
        )

        if not np.isclose(
            total_probability,
            1.0,
            atol=1e-8
        ):
            raise ValueError(
                f"分布总概率不为1："
                f"{total_probability=}"
            )

        cdf_diff = np.diff(
            F_new,
            axis=0
        )

        if np.any(cdf_diff < -1e-10):
            bad_position = np.argwhere(
                cdf_diff < -1e-10
            )[0]

            ia_bad = bad_position[0]
            ie_bad = bad_position[1]

            raise ValueError(
                f"F_new不满足CDF单调性："
                f"{ia_bad=}, {ie_bad=}, "
                f"F_current="
                f"{F_new[ia_bad, ie_bad]}, "
                f"F_next="
                f"{F_new[ia_bad + 1, ie_bad]}"
            )

        if np.max(
            np.abs(F_new - F_old)
        ) < 1e-8:
            return F_new

        F_old = F_new.copy()

    raise RuntimeError(
        "分布循环10万次未得到稳定分布。"
    )

def aggregate_asset(F_star, a_grid):
    interval_mass = np.diff(F_star,axis=0)

    a_mid = (a_grid[:-1] + a_grid[1:])/2

    lower_bound_asset = (
        a_grid[0] * np.sum(F_star[0, :])
    )

    # 每个区间：资产中点 × 区间概率
    interval_asset = np.sum(
        a_mid[:, None] * interval_mass
    )

    return lower_bound_asset + interval_asset

def Asset_equilibrium(q, a_grid, DVguess_grid, Fguess_grid):

    DVstar_grid, policy_star_grid = Bellman(
        DVguess_grid,
        q,
        a_grid
    )

    F_star_grid = distribution_iteration(
        policy_star_grid,
        a_grid,
        Fguess_grid
    )

    Aggre_asset = aggregate_asset(
        F_star_grid,
        a_grid
    )
    print(f"完成对{q =}的循环")
    return DVstar_grid, policy_star_grid, F_star_grid, Aggre_asset

def q_bounds_gen(a_grid, e_grid, beta):
    a_lower = a_grid[0]
    e_lower = np.min(e_grid)

    if a_lower < 0:
        # 保证最低状态存在 c > 0 的政策
        feasibility_lower = (
            1 + e_lower / a_lower
        )

        q_lower = max(
            0.0,
            beta,
            feasibility_lower
        )

        # 严格大于下界
        q_lower = np.nextafter(
            q_lower,
            np.inf
        )

        q_upper = np.inf

    elif a_lower == 0:
        q_lower = np.nextafter(
            beta,
            np.inf
        )

        q_upper = np.inf

    else:
        # a_lower > 0 时，可行性反而会给出q的上界
        q_lower = np.nextafter(
            beta,
            np.inf
        )

        q_upper = np.nextafter(
            1 + e_lower / a_lower,
            -np.inf
        )

        if q_lower >= q_upper:
            raise ValueError(
                f"不存在可行q区间："
                f"{a_lower=}, {e_lower=}, "
                f"{q_lower=}, {q_upper=}"
            )

    return q_lower, q_upper

def main():
    all_results = []

    for a_min in [-4]:
        q_step = 0.005
        N = 1000
        a_grid = a_min + da * np.arange(N)

        DVguess_grid = np.zeros(
            (len(a_grid), len(e_grid)),
            dtype=float
        )

        Fguess_grid = np.zeros(
            (len(a_grid), len(e_grid)),
            dtype=float
        )
        for ia in range(len(a_grid)):
            Fguess_grid[ia, :] = (
                                         ia / (len(a_grid) - 1)
                                 ) / len(e_grid)

        def asset_at_q(q_test):
            nonlocal DVguess_grid, Fguess_grid

            result = Asset_equilibrium(
                q_test,
                a_grid,
                DVguess_grid,
                Fguess_grid
            )

            DVguess_grid = result[0].copy()
            Fguess_grid = result[2].copy()

            return result[3]

        q_lower, q_upper = q_bounds_gen(
            a_grid,
            e_grid,
            beta
        )

        # 不直接在理论下界上求解
        q = q_lower + q_step
        Aggre_asset_old = asset_at_q(q)

        print(
            f"初始：{a_min=}, {q=}, "
            f"{Aggre_asset_old=}"
        )

        if abs(Aggre_asset_old) < 1e-10:
            print(f"{a_min=}, q_star={q}")
            continue

        if Aggre_asset_old < 0:
            q_step = -q_step

        q_star = None

        for _ in range(10000):
            if q_step > 0:
                q_new = q + q_step
            if q_step < 0:
                q_step = 0.9 * q_step
                q_new = q + q_step

            if q_new < 0.9941:
                q_new = 0.9941
            #热补丁，q太小导致需要的循环网格太大导致找不到max policy。 否则需要用迭代Bellman找最优a上界。该处没太大必要。
            #其他条件不变的条件下，提高借款下限会使家庭的precautionary saving提高，进而提高每期Bellman循环中的max a_prime值。
            #但是由于下限提高，precautionary saving需求总体提高，家庭渴望更多存款，需要更低利率，更高q进行抵消。所以q——star相对更高。此时我的二分法不用取到那么低的区间，也就不会循环到特别低的a_prime。
            if q_new >= q_upper:
                raise ValueError(
                    f"已经到达q的可行上界："
                    f"{q_upper=}"
                )
            if q_new <= q_lower:
                raise ValueError(
                    f"已经到达q的可行下界："
                    f"{q_lower=}"
                )

            Aggre_asset_new = asset_at_q(q_new)

            print(
                f"{a_min=}, {q_new=}, "
                f"{Aggre_asset_new=}"
            )

            if Aggre_asset_new * Aggre_asset_old < 0:
                q_star = bisection(
                    q,
                    q_new,
                    0,
                    asset_at_q,
                    1e-6,
                    1e-12,
                    Aggre_asset_old,
                    Aggre_asset_new,
                )

                print(
                    f"{a_min=}, {q_star=}, {Aggre_asset_new=}, {Aggre_asset_old=}"
                )

                # 找到这个a_min对应的q后，
                # 结束内层q搜索
                break

            q = q_new
            Aggre_asset_old = Aggre_asset_new

        if q_star is None:
            raise ValueError(
                f"{a_min=}，从{q_lower=}开始"
                f"仍未找到异号区间"
            )

        (
            DVstar_grid,
            policy_star_grid,
            F_star_grid,
            Aggre_asset_star
        ) = Asset_equilibrium(
            q_star,
            a_grid,
            DVguess_grid,
            Fguess_grid
        )

        print(
            f"\n最终结果："
            f"{a_min=}, "
            f"{q_star=}, "
            f"{Aggre_asset_star=}\n"
        )

        # 保存这个 a_min 的最终结果
        all_results.append({
            "a_min": a_min,
            "q_star": q_star,
            "a_grid": a_grid.copy(),
            "policy": policy_star_grid.copy(),
            "F": F_star_grid.copy(),
            "Aggre_asset": Aggre_asset_star
        })

    return all_results

results = main()

# 保存文件夹
save_dir = Path(
    "/Users/hrlo/Desktop/papers/model_results"
)

save_dir.mkdir(
    parents=True,
    exist_ok=True
)


for result in results:

    a_min = result["a_min"]
    q_star = result["q_star"]
    a_grid = result["a_grid"]
    policy = result["policy"]
    F = result["F"]
    Aggre_asset = result["Aggre_asset"]

    # 例如 -8 -> minus_8
    a_min_name = str(a_min).replace("-", "minus_")

    file_path = (
        save_dir
        / f"result_{a_min_name}.npz"
    )

    np.savez_compressed(
        file_path,

        # 最终均衡结果
        a_min=a_min,
        q_star=q_star,
        Aggre_asset=Aggre_asset,

        # 之后画图需要的数据
        a_grid=a_grid,
        policy=policy,
        F=F,

        # 顺手把模型参数一起保存
        e_grid=e_grid,
        beta=beta,
        sigma=sigma,
        da=da,
        P_trans_matrix=P_trans_matrix
    )

    print(
        f"结果已保存：{file_path}\n"
        f"a_min = {a_min}\n"
        f"q_star = {q_star}\n"
        f"Aggre_asset = {Aggre_asset}\n"
    )
