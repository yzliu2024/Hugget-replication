from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# ==========================================================
# 1. 读取结果的文件夹
# ==========================================================

result_dir = Path(
    "/Users/hrlo/Desktop/papers/model_results"
)


# ==========================================================
# 2. 图片保存文件夹
# ==========================================================

figure_dir = Path(
    "/Users/hrlo/Desktop/papers/figures"
)

figure_dir.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# 3. 找到所有 result_*.npz
# ==========================================================

result_files = sorted(
    result_dir.glob("result_*.npz")
)

if len(result_files) == 0:
    raise FileNotFoundError(
        f"在 {result_dir} 中没有找到 result_*.npz"
    )


# ==========================================================
# 4. 逐个读取并画图
# ==========================================================

for file_path in result_files:

    data = np.load(file_path)

    # 读取保存的数据
    a_min = float(data["a_min"])
    q_star = float(data["q_star"])
    Aggre_asset = float(data["Aggre_asset"])

    a_grid = data["a_grid"]
    policy = data["policy"]
    F = data["F"]
    e_grid = data["e_grid"]

    # 文件名使用
    a_min_name = (
        str(a_min)
        .replace("-", "minus_")
        .replace(".0", "")
    )


    # ======================================================
    # Policy Function
    # 横轴：a_t
    # 纵轴：a_{t+1}
    # eh 和 el 放在一张图
    # ======================================================

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        a_grid,
        policy[0:, 0],
        label=f"$e_h={e_grid[0]}$"
    )

    plt.plot(
        a_grid,
        policy[:, 1],
        label=f"$e_l={e_grid[1]}$"
    )

    # 45度线
    plt.plot(
        a_grid,
        a_grid,
        "--",
        label="$a_{t+1}=a_t$"
    )

    plt.xlabel(
        "$a_t$"
    )

    plt.ylabel(
        "$a_{t+1}$"
    )

    plt.title(
        f"Policy Function\n"
        f"$a_{{min}}={a_min:g}$, "
        f"$q^*={q_star:.8f}$"
    )

    plt.legend()

    plt.grid()

    plt.tight_layout()

    plt.savefig(
        figure_dir
        / f"policy_{a_min_name}.png",
        dpi=300,
        bbox_inches="tight"
    )

    # 不使用 plt.show()
    # 否则 Mac 上可能会阻塞程序
    plt.close()


    # ======================================================
    # F Distribution
    # eh 和 el 放在一张图
    # ======================================================

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        a_grid,
        F[:, 0],
        label=f"$e_h={e_grid[0]}$"
    )

    plt.plot(
        a_grid,
        F[:, 1],
        label=f"$e_l={e_grid[1]}$"
    )

    plt.xlabel(
        "$a$"
    )

    plt.ylabel(
        "$F(a,e)$"
    )

    plt.title(
        f"Asset Distribution\n"
        f"$a_{{min}}={a_min:g}$, "
        f"$q^*={q_star:.8f}$"
    )

    plt.legend()

    plt.grid()

    plt.tight_layout()

    plt.savefig(
        figure_dir
        / f"F_{a_min_name}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    # ======================================================
    # 输出信息
    # ======================================================

    print(
        f"已完成："
        f"a_min={a_min:g}, "
        f"q_star={q_star:.8f}, "
        f"Aggre_asset={Aggre_asset}"
    )


print(
    f"\n全部图片已保存到："
    f"{figure_dir}"
)