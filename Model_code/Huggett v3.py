import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

beta = 0.99322

# ===== Comparative statics settings =====
SIGMA_SET = [3.0, 1.5]
CREDIT_LIMITS = [-8, -6, -4, -2]

# Du() reads this global value; main() updates it for each sigma.
sigma = SIGMA_SET[0]
da = 0.1
e_grid = np.array([1.0,0.1])
P_trans_matrix = np.array([
    [0.925,0.5],
    #ij表示j转换到i的概率
    [0.075,0.5]
])


def c(a, e, a_prime, q):
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
"""
def DV_grid_gen(V_grid, a_grid):
    DV_grid = np.zeros_like(V_grid, dtype=float)

    for ie, e in enumerate(e_grid):
        for ia, a in enumerate(a_grid):
            if ia == 0:
                DV_grid[ia, ie] = (
                    V_grid[ia + 1, ie] - V_grid[ia, ie]
                ) / da

            elif ia == len(a_grid) - 1:
                DV_grid[ia, ie] = (
                    V_grid[ia, ie] - V_grid[ia - 1, ie]
                ) / da

            else:
                DV_grid[ia, ie] = (
                    V_grid[ia + 1, ie]
                    - V_grid[ia - 1, ie]
                ) / (2 * da)

    return DV_grid


def DV_func_gen(DV_grid, a_grid):
    def DV_func(a_func, ie_func):
        idx = np.searchsorted(
            a_grid,
            a_func,
            side='left'
        )

        if a_func < a_grid[0]:
            raise ValueError(
                f"DV_func 接受的a_func值低于最低a约束，"
                f"下一期a值:{a_func},当期状态:{ie_func}"
            )

        if a_func > a_grid[-1]:
            raise ValueError(
                f"DV_func 接受的a_func值高于最高a约束，"
                f"下一期a值:{a_func},当期状态:{ie_func}"
            )

        if idx == 0:
            DV = DV_grid[0, ie_func]
        else:
            lamb = (
                a_func - a_grid[idx - 1]
            ) / da

            DV = (
                lamb * DV_grid[idx, ie_func]
                + (1 - lamb) * DV_grid[idx - 1, ie_func]
            )

        return DV

    return DV_func

def E(a,ie,DV_func):
    Edv = 0
    for ie_prime, e_prime in enumerate(e_grid):
        Edv += P_trans_matrix[ie_prime,ie] * DV_func(a,ie_prime)
    return Edv
"""

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

"""
def policy_func_gen(policy_grid, a_grid):
    def policy_func(a_func, ie_func):
        idx = np.searchsorted(
            a_grid,
            a_func,
            side='left'
        )

        if a_func < a_grid[0]:
            raise ValueError(
                f"policy_func 接受的a_func值低于最低a约束，"
                f"下一期a值:{a_func},当期状态:{ie_func}"
            )

        if a_func > a_grid[-1]:
            raise ValueError(
                f"policy_func 接受的a_func值高于最高a约束，"
                f"下一期a值:{a_func},当期状态:{ie_func}"
            )

        if idx == 0:
            policy = policy_grid[0, ie_func]
        else:
            lamb = (
                a_func - a_grid[idx - 1]
            ) / da

            policy = (
                lamb * policy_grid[idx, ie_func]
                + (1 - lamb)
                * policy_grid[idx - 1, ie_func]
            )

        if policy < a_grid[0]:
            raise ValueError(
                f"policy_func_连续化错误，"
                f"{idx=}, {a_func=}, {ie_func=}"
            )

        return policy

    return policy_func
"""

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
                    c(
                        a,
                        e,
                        policy_grid[ia, ie],
                        q
                    )
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

def solve_credit_limit(a_min):
    """Solve one equilibrium for the current global sigma and one borrowing limit."""
    q_start_gap = 0.005
    q_step = 0.0005
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

        # q 搜索内部继续使用热启动
        DVguess_grid = result[0].copy()
        Fguess_grid = result[2].copy()

        return result[3]

    q_lower, q_upper = q_bounds_gen(
        a_grid,
        e_grid,
        beta
    )

    # 不直接在理论下界上求解
    q = q_lower + q_start_gap
    Aggre_asset_old = asset_at_q(q)

    print(
        f"初始：sigma={sigma}, {a_min=}, {q=}, "
        f"{Aggre_asset_old=}"
    )

    # 如果初始点已经非常接近市场清算，就直接用它
    if abs(Aggre_asset_old) < 1e-10:
        q_star = q
    else:
        if Aggre_asset_old < 0:
            q_step = -q_step

        q_star = None

        for _ in range(10000):
            if q_step > 0:
                q_new = q + q_step
            else:
                q_step = 0.9 * q_step
                q_new = q + q_step

            if q_new < 0.9941:
                q_new = 0.9941
            # 热补丁：q 太小会让所需资产网格上界变得很大。
            # 如果之后你扩大 a_grid，可以考虑删掉这两行 floor。

            if q_new >= q_upper:
                raise ValueError(
                    f"已经到达q的可行上界：{q_upper=}"
                )
            if q_new <= q_lower:
                raise ValueError(
                    f"已经到达q的可行下界：{q_lower=}"
                )

            Aggre_asset_new = asset_at_q(q_new)

            print(
                f"sigma={sigma}, {a_min=}, {q_new=}, "
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
                break

            q = q_new
            Aggre_asset_old = Aggre_asset_new

        if q_star is None:
            raise ValueError(
                f"sigma={sigma}, {a_min=}，从{q_lower=}开始"
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

    # 一期无风险债券：q = 1 / (1 + r)
    interest_rate = 1.0 / q_star - 1.0

    print(
        f"\n最终结果：sigma={sigma}, {a_min=}, "
        f"q_star={q_star:.8f}, "
        f"interest_rate={100 * interest_rate:.4f}%, "
        f"Aggre_asset={Aggre_asset_star:.6g}\n"
    )

    return {
        "sigma": sigma,
        "a_min": a_min,
        "q_star": q_star,
        "interest_rate": interest_rate,
        "a_grid": a_grid.copy(),
        "policy": policy_star_grid.copy(),
        "F": F_star_grid.copy(),
        "Aggre_asset": Aggre_asset_star,
    }


def main():
    global sigma
    all_results = []

    for sigma_value in SIGMA_SET:
        sigma = sigma_value
        print("\n" + "=" * 72)
        print(f"开始 sigma = {sigma}")
        print("=" * 72)

        for a_min in CREDIT_LIMITS:
            all_results.append(
                solve_credit_limit(a_min)
            )

    return all_results


def save_npz_results(results, save_dir):
    for result in results:
        sigma_value = result["sigma"]
        a_min = result["a_min"]

        sigma_name = str(sigma_value).replace(".", "p")
        a_min_name = str(a_min).replace("-", "minus_").replace(".", "p")

        file_path = (
            save_dir
            / f"result_sigma_{sigma_name}_amin_{a_min_name}.npz"
        )

        np.savez_compressed(
            file_path,
            sigma=sigma_value,
            a_min=a_min,
            q_star=result["q_star"],
            interest_rate=result["interest_rate"],
            Aggre_asset=result["Aggre_asset"],
            a_grid=result["a_grid"],
            policy=result["policy"],
            F=result["F"],
            e_grid=e_grid,
            beta=beta,
            da=da,
            P_trans_matrix=P_trans_matrix,
        )

        print(f"结果已保存：{file_path}")


def save_summary_csv(results, save_dir):
    """Excel 可以直接打开；同时避免额外依赖 pandas/openpyxl。"""
    csv_path = save_dir / "equilibrium_summary.csv"

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sigma",
            "credit_limit_a_min",
            "bond_price_q_star",
            "interest_rate",
            "interest_rate_percent",
            "aggregate_asset",
        ])

        for r in results:
            writer.writerow([
                r["sigma"],
                r["a_min"],
                r["q_star"],
                r["interest_rate"],
                100 * r["interest_rate"],
                r["Aggre_asset"],
            ])

    print(f"汇总表已保存：{csv_path}")


def save_three_line_table(results, save_dir):
    """保存一张论文风格的三线表 PNG。"""
    rows = []
    for r in results:
        rows.append([
            f'{r["sigma"]:.2f}',
            f'{r["a_min"]:g}',
            f'{r["q_star"]:.6f}',
            f'{100 * r["interest_rate"]:.4f}%',
            f'{r["Aggre_asset"]:.2e}',
        ])

    headers = [
        r"$\sigma$",
        "Credit limit $a_{min}$",
        "Price $q^*$",
        "Interest rate $r^*$",
        "Aggregate assets",
    ]

    fig_height = max(3.2, 0.48 * len(rows) + 1.6)
    fig, ax = plt.subplots(figsize=(10.5, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1.0, 1.55)

    n_rows = len(rows)
    n_cols = len(headers)

    # 三线表：只保留表头上边、表头下边、全表底边
    for (row, col), cell in table.get_celld().items():
        cell.set_facecolor("white")
        cell.set_edgecolor("black")
        cell.set_linewidth(0.0)
        cell.visible_edges = ""

        if row == 0:
            cell.get_text().set_weight("bold")
            cell.visible_edges = "TB"
            cell.set_linewidth(1.1)

        if row == n_rows:
            cell.visible_edges = "B"
            cell.set_linewidth(1.1)

    ax.set_title(
        "Equilibrium price and interest rate across borrowing limits",
        fontsize=13,
        pad=16,
    )

    table_path = save_dir / "equilibrium_three_line_table.png"
    fig.savefig(table_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"三线表图片已保存：{table_path}")


def save_comparative_plots(results, save_dir):
    """分别画 q* 和 r*，两条线对应两个 sigma。"""
    x = np.arange(len(CREDIT_LIMITS))
    x_labels = [str(v) for v in CREDIT_LIMITS]

    # Price q*
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for sigma_value in SIGMA_SET:
        subset = [
            r for a_min in CREDIT_LIMITS
            for r in results
            if r["sigma"] == sigma_value and r["a_min"] == a_min
        ]
        ax.plot(
            x,
            [r["q_star"] for r in subset],
            marker="o",
            linewidth=2,
            label=fr"$\sigma={sigma_value}$",
        )

    ax.set_xticks(x, x_labels)
    ax.set_xlabel(r"Credit limit $a_{min}$ (more negative = looser)")
    ax.set_ylabel(r"Equilibrium bond price $q^*$")
    ax.set_title("Bond price across borrowing limits")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(save_dir / "price_by_credit_limit.png", dpi=300)
    plt.close(fig)

    # Interest rate r*
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for sigma_value in SIGMA_SET:
        subset = [
            r for a_min in CREDIT_LIMITS
            for r in results
            if r["sigma"] == sigma_value and r["a_min"] == a_min
        ]
        ax.plot(
            x,
            [100 * r["interest_rate"] for r in subset],
            marker="o",
            linewidth=2,
            label=fr"$\sigma={sigma_value}$",
        )

    ax.set_xticks(x, x_labels)
    ax.set_xlabel(r"Credit limit $a_{min}$ (more negative = looser)")
    ax.set_ylabel(r"Equilibrium interest rate $r^*$ (%)")
    ax.set_title("Interest rate across borrowing limits")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(save_dir / "interest_rate_by_credit_limit.png", dpi=300)
    plt.close(fig)

    print("对比图已保存：price_by_credit_limit.png")
    print("对比图已保存：interest_rate_by_credit_limit.png")


def print_summary(results):
    print("\n" + "=" * 82)
    print("SUMMARY")
    print("=" * 82)
    print(
        f'{"sigma":>8} {"a_min":>10} {"q*":>14} '
        f'{"r* (%)":>14} {"Agg.Asset":>16}'
    )
    print("-" * 82)

    for r in results:
        print(
            f'{r["sigma"]:>8.2f} '
            f'{r["a_min"]:>10g} '
            f'{r["q_star"]:>14.8f} '
            f'{100 * r["interest_rate"]:>14.6f} '
            f'{r["Aggre_asset"]:>16.6e}'
        )

    print("=" * 82)


if __name__ == "__main__":
    save_dir = Path(
        "/Users/hrlo/Desktop/papers/model_results"
    )
    save_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    results = main()

    print_summary(results)
    save_npz_results(results, save_dir)
    save_summary_csv(results, save_dir)
    save_three_line_table(results, save_dir)
    save_comparative_plots(results, save_dir)
